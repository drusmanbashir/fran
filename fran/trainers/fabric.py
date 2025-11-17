import os
from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any, List, Literal, Optional, Tuple, Union, cast

import lightning as L
import torch
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import FSDPStrategy, Strategy
from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from tqdm import tqdm

SchedulerConfig = Mapping[str, Any]


class TrainerFabric:
    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        max_epochs: Optional[int] = 1,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
        should_train: bool = True,
    ) -> None:
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )
        self.should_train = should_train
        self.global_step = 0
        self.grad_accum_steps: int = grad_accum_steps
        self.current_epoch = 0

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf")
        if not isinstance(limit_val_batches, int):
            assert limit_val_batches == float("inf")

        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency

    def fit(
        self,
        model: L.LightningModule,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        ckpt_path: Optional[str] = None,
    ) -> None:
        self.fabric.launch()

        train_loader = self.fabric.setup_dataloaders(
            train_loader, use_distributed_sampler=self.use_distributed_sampler
        )
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(
                val_loader, use_distributed_sampler=self.use_distributed_sampler
            )

        if isinstance(self.fabric.strategy, FSDPStrategy):
            raise NotImplementedError("BYOT currently does not support FSDP")

        optim_and_sched = model.configure_optimizers()
        if isinstance(optim_and_sched, Mapping):
            optimizer = optim_and_sched["optimizer"]
            scheduler_cfg = optim_and_sched.get("lr_scheduler")
        else:
            optimizer, scheduler_cfg = self._parse_optimizers_schedulers(
                optim_and_sched
            )

        assert optimizer is not None
        model, optimizer = self.fabric.setup(model, optimizer)

        state: dict[str, Any] = {
            "model": model,
            "optim": optimizer,
            "scheduler": scheduler_cfg,
        }

        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)
                if (
                    self.max_epochs is not None
                    and self.current_epoch >= self.max_epochs
                ):
                    self.should_stop = True

        self.fabric.call("on_fit_start")
        while not self.should_stop:
            if self.should_train:
                self.train_loop(
                    model,
                    optimizer,
                    train_loader,
                    limit_batches=self.limit_train_batches,
                    scheduler_cfg=scheduler_cfg,
                )

            if self.should_validate:
                self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)

            if self.should_train:
                self.step_scheduler(
                    model,
                    scheduler_cfg,
                    level="epoch",
                    current_value=self.current_epoch,
                )

            self.current_epoch += 1

            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

            self.save(state)

        self.fabric.call("on_fit_end")
        self.should_stop = False

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[SchedulerConfig] = None,
    ) -> None:
        self.fabric.call("on_train_epoch_start")
        iterable = self.progbar_wrapper(
            train_loader,
            total=min(len(train_loader), int(limit_batches)),
            desc=f"Epoch {self.current_epoch}",
        )

        for batch_idx, batch in enumerate(iterable):
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_train_batch_start", self, batch, batch_idx)

            should_optim_step = self.global_step % self.grad_accum_steps == 0
            if should_optim_step:
                self.fabric.call("on_before_optimizer_step", optimizer)

                optimizer.step(
                    partial(
                        self.training_step,
                        model=model,
                        batch=batch,
                        batch_idx=batch_idx,
                    )
                )
                self.fabric.call("on_before_zero_grad", optimizer)
                optimizer.zero_grad()
            else:
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)

            self.fabric.call(
                "on_train_batch_end", self, self._current_train_return, batch, batch_idx
            )

            if should_optim_step:
                self.step_scheduler(
                    model, scheduler_cfg, level="step", current_value=self.global_step
                )

            self._format_iterable(iterable, self._current_train_return["loss"], "train")
            self.global_step += int(should_optim_step)

            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        self.fabric.call("on_train_epoch_end")

    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ) -> None:
        if val_loader is None:
            return

        if val_loader is not None and not is_overridden(
            "validation_step", _unwrap_objects(model)
        ):
            rank_zero_warn(
                "Your LightningModule does not have a validation_step implemented, "
                "but you passed a validation dataloader. Skipping validation."
            )
            return

        if not is_overridden("on_validation_model_eval", _unwrap_objects(model)):
            model.eval()
        else:
            self.fabric.call("on_validation_model_eval")

        print("==" * 50)
        print("GRAD ENABLED")
        torch.set_grad_enabled(True)

        self.fabric.call("on_validation_epoch_start")

        iterable = self.progbar_wrapper(
            val_loader,
            total=min(len(val_loader), int(limit_batches)),
            desc="Validation",
        )

        for batch_idx, batch in enumerate(iterable):
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            out = self.validation_step(model, batch, batch_idx)
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())
            self.fabric.call("on_validation_batch_end", self, out, batch, batch_idx)
            self._current_val_return = out
            self._format_iterable(iterable, self._current_val_return["loss"], "val")

        self.fabric.call("on_validation_epoch_end")

        if not is_overridden("on_validation_model_train", _unwrap_objects(model)):
            model.train()
        else:
            self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)

    def validation_step(self, model: L.LightningModule, batch: Any, batch_idx: int):
        with autocast():
            out = model.validation_step(batch, batch_idx)
        return out

    def training_step(
        self, model: L.LightningModule, batch: Any, batch_idx: int
    ) -> torch.Tensor:
        with autocast():
            outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(
                batch, batch_idx=batch_idx
            )
            loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

        self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")

        self._current_train_return = apply_to_collection(
            outputs, dtype=torch.Tensor, function=lambda x: x.detach()
        )

        return loss

    def step_scheduler(
        self,
        model: L.LightningModule,
        scheduler_cfg: Optional[SchedulerConfig],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        if scheduler_cfg is None:
            return

        if scheduler_cfg.get("interval", "epoch") != level:
            return

        if current_value % int(scheduler_cfg.get("frequency", 1)) != 0:
            return

        possible_monitor_vals: dict[Optional[str], Optional[torch.Tensor]] = {
            None: None
        }

        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals["train_loss"] = self._current_train_return
        elif isinstance(self._current_train_return, Mapping):
            for k, v in self._current_train_return.get(
                "losses_for_logging", {}
            ).items():
                possible_monitor_vals[f"train_{k}"] = v

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals["val_loss"] = self._current_val_return
        elif isinstance(self._current_val_return, Mapping):
            for k, v in self._current_val_return.get("losses_for_logging", {}).items():
                possible_monitor_vals[f"val_{k}"] = v

        monitor_key = cast(Optional[str], scheduler_cfg.get("monitor"))
        try:
            monitor = possible_monitor_vals[monitor_key]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {monitor_key} is invalid. Possible values are {possible_keys}."
            ) from ex

        model.lr_scheduler_step(scheduler_cfg["scheduler"], monitor)

    @property
    def should_validate(self) -> bool:
        return self.current_epoch % self.validation_frequency == 0

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping[str, Any]], path: str) -> None:
        if state is None:
            state = {}

        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def save(self, state: Optional[Mapping[str, Any]]) -> None:
        if state is None:
            state = {}

        state = dict(state)
        state.update(global_step=self.global_step, current_epoch=self.current_epoch)

        self.fabric.save(
            os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"),
            state,
        )

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))
        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])

    def _parse_optimizers_schedulers(
        self, configure_optim_output: Any
    ) -> Tuple[Optional[Optimizer], Optional[SchedulerConfig]]:
        defaults: dict[str, Any] = {
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        if isinstance(configure_optim_output, Optimizer):
            return configure_optim_output, None

        if isinstance(configure_optim_output, (_LRScheduler, ReduceLROnPlateau)):
            cfg = {**defaults, "scheduler": configure_optim_output}
            return None, cfg

        if isinstance(configure_optim_output, Mapping):
            cfg = {**defaults, **dict(configure_optim_output)}
            return None, cfg

        if isinstance(configure_optim_output, (list, tuple)):
            if len(configure_optim_output) == 1:
                return self._parse_optimizers_schedulers(configure_optim_output[0])
            if len(configure_optim_output) == 2:
                opt, _ = self._parse_optimizers_schedulers(configure_optim_output[0])
                _, sched = self._parse_optimizers_schedulers(configure_optim_output[1])
                return opt, sched

        return None, None

    @staticmethod
    def _format_iterable(
        prog_bar: Any,
        candidates: Optional[
            Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]
        ],
        prefix: str,
    ) -> None:
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(
                candidates, torch.Tensor, lambda x: x.item()
            )
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():  # type: ignore[union-attr]
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)
