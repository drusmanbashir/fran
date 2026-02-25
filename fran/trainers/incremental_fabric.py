# %%
from __future__ import annotations

import os
import ipdb
tr = ipdb.set_trace

from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Literal, Optional, Union, cast

import torch
from lightning_utilities import apply_to_collection
from tqdm import tqdm

import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden

from lightning.pytorch.loggers import Logger
import os
import shutil
from collections.abc import Iterable, Mapping
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union, cast

import lightning as L
import torch
from fastcore.all import in_ipython
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger as FabricLogger
from lightning.fabric.strategies import FSDPStrategy, Strategy
from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.callbacks import BatchSizeFinder
from lightning.pytorch.callbacks import (DeviceStatsMonitor, EarlyStopping,
                                         ModelCheckpoint,
                                         TQDMProgressBar)
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from tqdm import tqdm
from utilz.stringz import headline
from lightning.pytorch.trainer.states import TrainerFn

from fran.callback.incremental import (
    LRFloorStop,
    UpdateDatasetOnPlateau,
    UpdateDatasetOnPlateauFabric,
)
from fran.callback.test import PeriodicTest
from fran.callback.wandb import WandbImageGridCallback, WandbLogBestCkpt
from fran.callback.case_recorder import CaseIDRecorder
from fran.configs.parser import ConfigMaker, parse_neptune_dict
from fran.callback.base import BatchSizeSafetyMargin
from fran.managers.data.incremental import DataManagerDualI, DataManagerMultiI
from fran.managers.project import Project
from fran.managers.unet import UNetManagerIncremental
from fran.managers.wandb import WandbManager
from fran.callback.lr_monitor_minimal import MinimalLearningRateMonitor
from fran.trainers.base import (backup_ckpt, checkpoint_from_model_id,
                                switch_ckpt_keys)

SchedulerConfig = Mapping[str, Any]


class _EpochLoopState:
    def __init__(self) -> None:
        self._batches_that_stepped = 0


class _FitLoopState:
    def __init__(self) -> None:
        self.epoch_loop = _EpochLoopState()


class _TrainerState:
    def __init__(self, fn: Any) -> None:
        self.fn = fn


def _flatten_dict(d: dict, base: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{base}/{k}" if base else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _dm_class_for_periodic_test(periodic_test: int):
    return DataManagerMultiI if int(periodic_test) > 0 else DataManagerDualI


def _dm_class_from_ckpt(ckpt_path: str | Path):
    sd = torch.load(ckpt_path, map_location="cpu")
    hp = sd.get("datamodule_hyper_parameters", {}) or sd.get("hyper_parameters", {})
    return DataManagerMultiI if "keys_test" in hp else DataManagerDualI



class TrainerFabric:
    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[list[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[list[Any], Any]] = None,
        loggers: Optional[Union[Logger, list[Logger]]] = None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
    ) -> None:
        """Exemplary Trainer with Fabric. This is a very simple trainer focused on readablity but with reduced
        featureset. As a trainer with more included features, we recommend using the
        :class:`lightning.pytorch.Trainer`.

        Args:
            accelerator: The hardware to run on. Possible choices are:
                ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
            strategy: Strategy for how to run across multiple devices. Possible choices are:
                ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
            devices: Number of devices to train on (``int``),
                which GPUs to train on (``list`` or ``str``), or ``"auto"``.
                The value applies per node.
            precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
                or bfloat16 precision AMP (``"bf16-mixed"``).
            plugins: One or several custom plugins
            callbacks: A single callback or a list of callbacks. The following hooks are supported:
                - on_train_epoch_start
                - on train_epoch_end
                - on_train_batch_start
                - on_train_batch_end
                - on_before_backward
                - on_after_backward
                - on_before_zero_grad
                - on_before_optimizer_step
                - on_validation_model_eval
                - on_validation_model_train
                - on_validation_epoch_start
                - on_validation_epoch_end
                - on_validation_batch_start
                - on_validation_batch_end

            loggers: A single logger or a list of loggers. See :meth:`~lightning.fabric.fabric.Fabric.log` for more
                information.

            max_epochs: The maximum number of epochs to train
            max_steps: The maximum number of (optimizer) steps to train
            grad_accum_steps: How many batches to process before each optimizer step
            limit_train_batches: Limits the number of train batches per epoch
                If greater than number of batches in the dataloader, this has no effect.
            limit_val_batches: Limits the number of validation batches per epoch.
                If greater than number of batches in the dataloader, this has no effect.
            validation_frequency: How many epochs to run before each validation epoch.
            use_distributed_sampler: Wraps the sampler of each dataloader with a respective distributed-aware sampler
                in case of distributed training.
            checkpoint_dir: Directory to store checkpoints to.
            checkpoint_frequency: How many epochs to run before each checkpoint is written.

        Warning:
            callbacks written for the lightning trainer (especially making assumptions on the trainer), won't work!

        """

        cb_list = [] if callbacks is None else (callbacks if isinstance(callbacks, list) else [callbacks])
        self.callbacks = cb_list
        for cb in self.callbacks:
            try:
                cb._trainer = self
            except Exception:
                pass
        self.checkpoint_callback = next(
            (cb for cb in self.callbacks if isinstance(cb, ModelCheckpoint)), None
        )

        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=None,
            loggers=loggers,
        )
        self.global_step = 0
        self.grad_accum_steps: int = grad_accum_steps
        self.current_epoch = 0

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        # ensures limit_X_batches is either int or inf
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
        self.default_root_dir = checkpoint_dir
        self.loggers = [] if loggers is None else (loggers if isinstance(loggers, list) else [loggers])
        self.logger = self.loggers[0] if self.loggers else None
        self.callback_metrics: dict[str, Any] = {}
        self.progress_bar_metrics: dict[str, Any] = {}
        self.optimizers: list[Optimizer] = []
        self.state = _TrainerState(TrainerFn.FITTING)
        self.fast_dev_run = False
        self.sanity_checking = False
        self.val_check_interval = 1.0
        self.check_val_every_n_epoch = self.validation_frequency
        self.fit_loop = _FitLoopState()
        self.datamodule = None
        self.ckpt_path = None
        self.strategy = self.fabric.strategy
        if not hasattr(self.strategy, "remove_checkpoint"):
            def _remove_checkpoint(filepath: str) -> None:
                try:
                    if filepath and os.path.exists(filepath):
                        os.remove(filepath)
                except Exception:
                    pass
            setattr(self.strategy, "remove_checkpoint", _remove_checkpoint)
        self.accumulate_grad_batches = self.grad_accum_steps
        self.train_dataloader = None
        self.val_dataloaders = None
        self.store_preds = False
        self.active_dataloader_idx = None
        self.current_train_dataloader_idx = 0
        setattr(self.fabric, "store_preds", self.store_preds)

    def _call_callbacks(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, hook_name, None)
            if callable(fn):
                fn(*args, **kwargs)
        # LightningModule.trainer is a Fabric shim under Fabric, so mirror trainer flags to fabric.
        if hasattr(self, "store_preds"):
            setattr(self.fabric, "store_preds", self.store_preds)

    @staticmethod
    def _as_loader_list(
        loader: Optional[Union[torch.utils.data.DataLoader, Sequence[torch.utils.data.DataLoader]]]
    ) -> list[torch.utils.data.DataLoader]:
        if loader is None:
            return []
        if isinstance(loader, (list, tuple)):
            return [dl for dl in loader]
        return [loader]

    def _setup_loader_collection(
        self, loader: Optional[Union[torch.utils.data.DataLoader, Sequence[torch.utils.data.DataLoader]]]
    ) -> Optional[Union[torch.utils.data.DataLoader, list[torch.utils.data.DataLoader]]]:
        if loader is None:
            return None
        loaders = self._as_loader_list(loader)
        configured = [
            self.fabric.setup_dataloaders(dl, use_distributed_sampler=self.use_distributed_sampler)
            for dl in loaders
        ]
        if isinstance(loader, tuple):
            return configured
        if isinstance(loader, list):
            return configured
        return configured[0]

    @staticmethod
    def _safe_len(loader: torch.utils.data.DataLoader) -> Optional[int]:
        try:
            return len(loader)
        except TypeError:
            return None

    @staticmethod
    def _capped_total(loader: torch.utils.data.DataLoader, limit_batches: Union[int, float]) -> Optional[int]:
        n = TrainerFabric._safe_len(loader)
        if n is None:
            if limit_batches == float("inf"):
                return None
            return int(limit_batches)
        return min(n, int(limit_batches)) if limit_batches != float("inf") else n

    @staticmethod
    def _extract_step_metrics(
        outputs: Optional[Union[torch.Tensor, Mapping[str, Any]]], prefix: str
    ) -> dict[str, Any]:
        if outputs is None:
            return {}
        metrics: dict[str, Any] = {}
        if isinstance(outputs, torch.Tensor):
            metrics[f"{prefix}_loss"] = outputs.detach()
            return metrics
        if isinstance(outputs, Mapping):
            for k, v in outputs.items():
                if isinstance(v, (torch.Tensor, float, int)):
                    metrics[f"{prefix}_{k}"] = v.detach() if isinstance(v, torch.Tensor) else v
            nested = outputs.get("losses_for_logging")
            if isinstance(nested, Mapping):
                for k, v in nested.items():
                    if isinstance(v, (torch.Tensor, float, int)):
                        metrics[f"{prefix}_{k}"] = v.detach() if isinstance(v, torch.Tensor) else v
        return metrics

    @staticmethod
    def _extract_model_loss_metrics(model: L.LightningModule, prefix: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        loss_fnc = getattr(model, "loss_fnc", None)
        loss_dict = getattr(loss_fnc, "loss_dict", None)
        if not isinstance(loss_dict, Mapping):
            return metrics
        for key, value in loss_dict.items():
            if "batch" in str(key):
                continue
            metric_name = f"{prefix}_{key}"
            if isinstance(value, torch.Tensor):
                metrics[metric_name] = value.detach()
            elif isinstance(value, (float, int)):
                metrics[metric_name] = float(value)
        return metrics

    def _update_trainer_metrics(
        self,
        outputs: Optional[Union[torch.Tensor, Mapping[str, Any]]],
        prefix: str,
        model: Optional[L.LightningModule] = None,
    ) -> None:
        metrics = self._extract_step_metrics(outputs, prefix)
        if model is not None:
            metrics.update(self._extract_model_loss_metrics(model, prefix))
        if metrics:
            root_device = getattr(getattr(self, "strategy", None), "root_device", "cpu")
            callback_ready: dict[str, Any] = {}
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    callback_ready[k] = v.detach()
                elif isinstance(v, (float, int)):
                    callback_ready[k] = torch.tensor(float(v), device=root_device)
            self.progress_bar_metrics.update(callback_ready)
            self.callback_metrics.update(callback_ready)
            if self.loggers:
                scalar_metrics = apply_to_collection(
                    metrics,
                    torch.Tensor,
                    lambda t: float(t.detach().cpu().item()) if t.numel() == 1 else None,
                )
                scalar_metrics = {
                    k: v for k, v in scalar_metrics.items() if isinstance(v, (float, int))
                }
                if scalar_metrics:
                    for logger in self.loggers:
                        if hasattr(logger, "log_metrics"):
                            logger.log_metrics(scalar_metrics, step=self.global_step)

    @property
    def is_global_zero(self) -> bool:
        return bool(self.fabric.is_global_zero)

    @property
    def global_rank(self) -> int:
        return int(self.fabric.global_rank)

    @property
    def world_size(self) -> int:
        return int(self.fabric.world_size)

    def save_checkpoint(self, filepath: str, weights_only: bool = False) -> None:
        state: dict[str, Any] = {
            "model": self._fitted_model,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
        }
        if not weights_only and self.optimizers:
            state["optim"] = self.optimizers[0]
        self.fabric.save(filepath, state)

    def fit(
        self,
        model: L.LightningModule,
        train_loader: Union[torch.utils.data.DataLoader, Sequence[torch.utils.data.DataLoader]],
        val_loader: Optional[Union[torch.utils.data.DataLoader, Sequence[torch.utils.data.DataLoader]]],
        ckpt_path: Optional[str] = None,
    ):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the LightningModule to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
            train_loader: the training dataloader. Has to be an iterable returning batches.
            val_loader: the validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.

        """
        self.fabric.launch()

        # setup dataloaders
        train_loader = self._setup_loader_collection(train_loader)
        val_loader = self._setup_loader_collection(val_loader)

        # setup model and optimizer
        if isinstance(self.fabric.strategy, L.fabric.strategies.fsdp.FSDPStrategy):
            # currently, there is no way to support fsdp with model.configure_optimizers in fabric
            # as it would require fabric to hold a reference to the model, which we don't want to.
            raise NotImplementedError("BYOT currently does not support FSDP")

        optimizer, scheduler_cfg = self._parse_optimizers_schedulers(model.configure_optimizers())
        assert optimizer is not None
        model, optimizer = self.fabric.setup(model, optimizer)
        try:
            model.setup(stage="fit")
        except TypeError:
            model.setup("fit")
        self._fitted_model = model
        self.optimizers = [optimizer]
        train_loaders = self._as_loader_list(train_loader)
        val_loaders = self._as_loader_list(val_loader)
        self.num_training_batches = sum(self._safe_len(dl) or 0 for dl in train_loaders)
        self.num_val_batches = [self._safe_len(dl) or 0 for dl in val_loaders] if val_loaders else [0]
        self.train_dataloader = train_loader
        self.val_dataloaders = val_loaders

        # assemble state (current epoch and global step will be added in save)
        state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}

        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)

                # check if we even need to train here
                if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                    self.should_stop = True

        self.fabric.call("on_fit_start")
        self._call_callbacks("on_fit_start", self, model)
        self.fabric.call("on_train_start")
        self._call_callbacks("on_train_start", self, model)
        while not self.should_stop:
            self.train_loop(
                model, optimizer, train_loader, limit_batches=self.limit_train_batches, scheduler_cfg=scheduler_cfg
            )

            if self.should_validate:
                self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)

            self.step_scheduler(model, scheduler_cfg, level="epoch", current_value=self.current_epoch)

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

            self.save(state)

        self.fabric.call("on_fit_end")
        self.fabric.call("on_train_end")
        self._call_callbacks("on_train_end", self, model)
        self._call_callbacks("on_fit_end", self, model)
        # reset for next fit call
        self.should_stop = False

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: Union[torch.utils.data.DataLoader, Sequence[torch.utils.data.DataLoader]],
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]] = None,
    ):
        """The training loop running a single training epoch.

        Args:
            model: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater than the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`
                for supported values.

        """
        self.fabric.call("on_train_epoch_start")
        self._call_callbacks("on_train_epoch_start", self, model)
        train_loaders = self._as_loader_list(train_loader)
        active_idx = getattr(self, "active_dataloader_idx", None)
        if active_idx is None:
            loader_items = list(enumerate(train_loaders))
        else:
            active_idx = int(active_idx)
            if active_idx < 0 or active_idx >= len(train_loaders):
                raise IndexError(
                    f"active_dataloader_idx={active_idx} out of range for {len(train_loaders)} train loaders"
                )
            loader_items = [(active_idx, train_loaders[active_idx])]

        for dataloader_idx, single_loader in loader_items:
            iterable = self.progbar_wrapper(
                single_loader,
                total=self._capped_total(single_loader, limit_batches),
                desc=f"Epoch {self.current_epoch}/train{dataloader_idx}",
            )

            for batch_idx, batch in enumerate(iterable):
                # end epoch if stopping training completely or max batches for this epoch reached
                if self.should_stop or batch_idx >= limit_batches:
                    break

                self.current_train_dataloader_idx = int(dataloader_idx)
                self.fabric.call("on_train_batch_start", batch, batch_idx)
                self._call_callbacks("on_train_batch_start", self, model, batch, batch_idx)

                # check if optimizer should step in gradient accumulation
                should_optim_step = self.global_step % self.grad_accum_steps == 0
                if should_optim_step:
                    # currently only supports a single optimizer
                    self.fabric.call("on_before_optimizer_step", optimizer)
                    self._call_callbacks("on_before_optimizer_step", self, model, optimizer)

                    # optimizer step runs train step internally through closure
                    optimizer.step(
                        partial(
                            self.training_step,
                            model=model,
                            batch=batch,
                            batch_idx=batch_idx,
                            dataloader_idx=dataloader_idx,
                        )
                    )
                    self.fabric.call("on_before_zero_grad", optimizer)
                    self._call_callbacks("on_before_zero_grad", self, model, optimizer)

                    optimizer.zero_grad()

                else:
                    # gradient accumulation -> no optimizer step
                    self.training_step(
                        model=model,
                        batch=batch,
                        batch_idx=batch_idx,
                        dataloader_idx=dataloader_idx,
                    )

                self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)
                self._call_callbacks("on_train_batch_end", self, model, self._current_train_return, batch, batch_idx)
                train_prefix = "train" if dataloader_idx == 0 else f"train{dataloader_idx}"
                self._update_trainer_metrics(self._current_train_return, train_prefix, model)
                if dataloader_idx == 0:
                    self._update_trainer_metrics(self._current_train_return, "train0", model)

                # this guard ensures, we only step the scheduler once per global step
                if should_optim_step:
                    self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

                # add output values to progress bar
                self._format_iterable(iterable, self._current_train_return, train_prefix)

                # only increase global step if optimizer stepped
                self.global_step += int(should_optim_step)
                self.fit_loop.epoch_loop._batches_that_stepped = self.global_step

                # stopping criterion on step level
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    self.should_stop = True
                    break

        self.fabric.call("on_train_epoch_end")
        self._call_callbacks("on_train_epoch_end", self, model)

    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[Union[torch.utils.data.DataLoader, Sequence[torch.utils.data.DataLoader]]],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop running a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.

        """
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        # no validation but warning if val_loader was passed, but validation_step not implemented
        if val_loader is not None and not is_overridden("validation_step", _unwrap_objects(model)):
            L.fabric.utilities.rank_zero_warn(
                "Your LightningModule does not have a validation_step implemented, "
                "but you passed a validation dataloder. Skipping Validation."
            )
            return

        if not is_overridden("on_validation_model_eval", _unwrap_objects(model)):
            model.eval()
        else:
            self.fabric.call("on_validation_model_eval")  # calls `model.eval()`

        torch.set_grad_enabled(False)
        self.fabric.call("on_validation_start")
        self._call_callbacks("on_validation_start", self, model)

        self.fabric.call("on_validation_epoch_start")
        self._call_callbacks("on_validation_epoch_start", self, model)

        val_loaders = self._as_loader_list(val_loader)
        for dataloader_idx, single_loader in enumerate(val_loaders):
            iterable = self.progbar_wrapper(
                single_loader,
                total=self._capped_total(single_loader, limit_batches),
                desc=f"Validation/val{dataloader_idx}",
            )

            for batch_idx, batch in enumerate(iterable):
                # end epoch if stopping training completely or max batches for this epoch reached
                if self.should_stop or batch_idx >= limit_batches:
                    break

                self.fabric.call("on_validation_batch_start", batch, batch_idx)
                self._call_callbacks("on_validation_batch_start", self, model, batch, batch_idx, dataloader_idx)

                out = model.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
                # avoid gradients in stored/accumulated values -> prevents potential OOM
                out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

                self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
                self._call_callbacks("on_validation_batch_end", self, model, out, batch, batch_idx, dataloader_idx)
                self._current_val_return = out
                val_prefix = "val" if dataloader_idx == 0 else f"val{dataloader_idx}"
                self._update_trainer_metrics(self._current_val_return, val_prefix, model)
                if dataloader_idx == 0:
                    self._update_trainer_metrics(self._current_val_return, "val0", model)
                self._format_iterable(iterable, self._current_val_return, val_prefix)

        self.fabric.call("on_validation_epoch_end")
        self._call_callbacks("on_validation_epoch_end", self, model)
        self.fabric.call("on_validation_end")
        self._call_callbacks("on_validation_end", self, model)

        if not is_overridden("on_validation_model_train", _unwrap_objects(model)):
            model.train()
        else:
            self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)

    def training_step(
        self, model: L.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(
            batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )

        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

        self.fabric.call("on_before_backward", loss)
        self._call_callbacks("on_before_backward", self, model, loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")
        self._call_callbacks("on_after_backward", self, model)

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())

        return loss

    def step_scheduler(
        self,
        model: L.LightningModule,
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """

        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals: dict[Optional[str], Optional[torch.Tensor]] = {None: None}
        for key, value in self.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                possible_monitor_vals[key] = value
            elif isinstance(value, (float, int)):
                possible_monitor_vals[key] = torch.tensor(float(value), device=self.strategy.root_device)
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals["train_loss"] = self._current_train_return
        elif isinstance(self._current_train_return, Mapping):
            for k, v in self._current_train_return.items():
                if isinstance(v, torch.Tensor):
                    possible_monitor_vals["train_" + k] = v
                elif isinstance(v, (float, int)):
                    possible_monitor_vals["train_" + k] = torch.tensor(float(v), device=self.strategy.root_device)

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals["val_loss"] = self._current_val_return
        elif isinstance(self._current_val_return, Mapping):
            for k, v in self._current_val_return.items():
                if isinstance(v, torch.Tensor):
                    possible_monitor_vals["val_" + k] = v
                elif isinstance(v, (float, int)):
                    possible_monitor_vals["val_" + k] = torch.tensor(float(v), device=self.strategy.root_device)

        try:
            monitor = possible_monitor_vals[cast(Optional[str], scheduler_cfg["monitor"])]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler_cfg['monitor']} is invalid. Possible values are {possible_keys}."
            ) from ex

        # rely on model hook for actual step
        model.lr_scheduler_step(scheduler_cfg["scheduler"], monitor)

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.current_epoch % self.validation_frequency == 0

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], path: str) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from

        """
        if state is None:
            state = {}

        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def save(self, state: Optional[Mapping]) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        if state is None:
            state = {}

        state.update(global_step=self.global_step, current_epoch=self.current_epoch)

        self.fabric.save(os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"), state)

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints

        """
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))

        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])

    def _parse_optimizers_schedulers(
        self, configure_optim_output
    ) -> tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[Mapping[str, Union[_LRScheduler, ReduceLROnPlateau, bool, str, int]]],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        """
        _lr_sched_defaults = {"interval": "epoch", "frequency": 1, "monitor": "val_loss"}

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        if isinstance(configure_optim_output, (_LRScheduler, ReduceLROnPlateau)):
            _lr_sched_defaults.update(scheduler=configure_optim_output)
            return None, _lr_sched_defaults

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            # {"optimizer": ..., "lr_scheduler": ...} shape from LightningModule.configure_optimizers
            if "optimizer" in configure_optim_output:
                opt = configure_optim_output.get("optimizer")
                if not isinstance(opt, L.fabric.utilities.types.Optimizable):
                    return None, None

                lr_conf = configure_optim_output.get("lr_scheduler")
                if lr_conf is None:
                    return opt, None

                cfg = dict(_lr_sched_defaults)
                monitor = configure_optim_output.get("monitor")

                if isinstance(lr_conf, Mapping):
                    cfg.update(lr_conf)
                    if monitor is not None and "monitor" not in cfg:
                        cfg["monitor"] = monitor
                    return opt, cfg

                if isinstance(lr_conf, (_LRScheduler, ReduceLROnPlateau)):
                    cfg["scheduler"] = lr_conf
                    if monitor is not None:
                        cfg["monitor"] = monitor
                    return opt, cfg

                return opt, None

            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(isinstance(_opt_cand, L.fabric.utilities.types.Optimizable) for _opt_cand in configure_optim_output):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            if all(
                isinstance(_lr_cand, (_LRScheduler, ReduceLROnPlateau, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return None, self._parse_optimizers_schedulers(configure_optim_output[0])[1]

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    @staticmethod
    def _format_iterable(
        prog_bar, candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]], prefix: str
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)

class IncrementalTrainerMinimal:
    def __init__(
        self,
        project_title,
        configs,
        run_name=None,
        ckpt_path: Optional[str | Path] = None,
    ):
        self.project = Project(project_title=project_title)
        self.configs = configs
        self.run_name = run_name
        if ckpt_path is not None:
            self.ckpt = Path(ckpt_path)
        else:
            self.ckpt = None if run_name is None else checkpoint_from_model_id(run_name)

        self.periodic_test = 0
        self._log_incremental_to_wandb = True
        self.start_n = 40
        self._checkpoint_dir: Optional[Path] = None
        self.qc_configs(configs)

    def _resolve_checkpoint_dir(self, logger=None) -> Path:
        run_token = None
        if logger is not None and hasattr(logger, "experiment"):
            exp = logger.experiment
            run_token = getattr(exp, "id", None) or getattr(exp, "name", None)
        run_token = run_token or self.run_name
        if run_token:
            return Path(self.project.checkpoints_parent_folder) / str(run_token) / "checkpoints"
        return Path(self.project.checkpoints_parent_folder) / "checkpoints"

    def setup(
        self,
        batch_size=None,
        logging_freq=25,
        lr=None,
        devices=1,
        compiled=None,
        wandb=True,
        profiler=False,
        periodic_test: int = 0,
        cbs=None,
        tags=None,
        description="",
        epochs=600,
        batchsize_finder=False,
        override_dm_checkpoint=False,
        early_stopping=False,
        early_stopping_monitor="val0_loss",
        early_stopping_mode="min",
        early_stopping_patience=30,
        early_stopping_min_delta=0.0,
        lr_floor=None,
        start_n: int = 40,
        wandb_grid_epoch_freq: int = 5,
        log_incremental_to_wandb: bool = True,
    ):
        self.periodic_test = int(periodic_test)
        self.start_n = int(start_n)
        self._log_incremental_to_wandb = bool(log_incremental_to_wandb)
        cbs = list(cbs or [])
        tags = list(tags or [])

        self.maybe_alter_configs(batch_size, compiled)
        self.set_lr(lr)
        self.set_strategy(devices)
        self.init_dm_unet(
            batch_size=batch_size, override_dm_checkpoint=override_dm_checkpoint
        )

        callbacks, logger, profiler_obj = self.init_cbs(
            cbs=cbs,
            wandb=wandb,
            batchsize_finder=batchsize_finder,
            periodic_test=self.periodic_test,
            profiler=profiler,
            tags=tags,
            description=description,
            early_stopping=early_stopping,
            early_stopping_monitor=early_stopping_monitor,
            early_stopping_mode=early_stopping_mode,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            lr_floor=lr_floor,
            wandb_grid_epoch_freq=int(wandb_grid_epoch_freq),
        )
        self.logger = logger
        self._checkpoint_dir = self._resolve_checkpoint_dir(logger=self.logger)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.D.prepare_data()
        self.D.setup(stage="fit")

        self.train_loader = self.D.train_dataloader()
        self.val_loader = self.D.val_dataloader()

        self.trainer = TrainerFabric(
            callbacks=callbacks,
            loggers=logger,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=devices if torch.cuda.is_available() else 1,
            precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
            max_epochs=epochs,
            limit_train_batches=float("inf"),
            limit_val_batches=float("inf"),
            checkpoint_dir=str(self._checkpoint_dir),
            strategy=self.strategy if torch.cuda.is_available() else "auto",
        )
        if wandb and len(getattr(self.trainer.fabric, "_loggers", [])) == 0:
            raise RuntimeError(
                "wandb=True but no Fabric logger is attached. Check logger wiring in setup()."
            )

    def fit(self):
        # Refresh loaders from datamodule at fit time to match current incremental state.
        train_loader = self.D.train_dataloader()
        val_loader = self.D.val_dataloader()
        self.trainer.datamodule = self.D
        self.trainer.logger = self.logger
        self.trainer.loggers = [] if self.logger is None else [self.logger]
        if self.ckpt is not None and Path(self.ckpt).is_file():
            try:
                sd = torch.load(self.ckpt, map_location="cpu")
                if isinstance(sd, dict):
                    gs = sd.get("global_step", None)
                    ep = sd.get("epoch", sd.get("current_epoch", None))
                    if gs is not None:
                        self.trainer.global_step = int(gs)
                        self.trainer.fit_loop.epoch_loop._batches_that_stepped = int(gs)
                    if ep is not None:
                        self.trainer.current_epoch = int(ep)
            except Exception:
                pass
        self.trainer.fit(
            model=self.N,
            train_loader=train_loader,
            val_loader=val_loader,
            ckpt_path=str(self.ckpt) if self.ckpt is not None else None,
        )

    def init_cbs(
        self,
        cbs,
        wandb,
        batchsize_finder,
        periodic_test,
        profiler,
        tags,
        description="",
        early_stopping=False,
        early_stopping_monitor="val0_loss",
        early_stopping_mode="min",
        early_stopping_patience=30,
        early_stopping_min_delta=0.0,
        lr_floor=None,
        wandb_grid_epoch_freq: int = 5,
    ):
        logger = None
        if wandb:
            logger = WandbManager(
                project=self.project,
                run_id=self.run_name,
                log_model_checkpoints=False,
                tags=tags,
                notes=description,
            )
            if hasattr(logger, "experiment"):
                exp = logger.experiment
                self.run_name = getattr(exp, "id", None) or getattr(exp, "name", None) or self.run_name
        checkpoint_dir = self._resolve_checkpoint_dir(logger=logger)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if batchsize_finder:
            cbs += [
                BatchSizeFinder(batch_arg_name="batch_size", mode="binsearch"),
                BatchSizeSafetyMargin(buffer=1),
            ]
        if periodic_test > 0:
            cbs += [PeriodicTest(every_n_epochs=periodic_test, limit_batches=50)]

        cbs += [
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                save_last=True,
                monitor="val0_loss",
                every_n_epochs=10,
                filename="{epoch}-{val0_loss:.2f}",
                enable_version_counter=True,
                auto_insert_metric_name=True,
            ),
            TQDMProgressBar(refresh_rate=3),
            UpdateDatasetOnPlateauFabric(
                monitor="val0_loss",
                n_samples_to_add=30,
                target_label=1,
                loss_threshold=0.5,
                log_to_wandb=bool(self._log_incremental_to_wandb),
            ),
            CaseIDRecorder(
                freq=1,
                window_epochs=1,
                include_train=True,
                log_wandb=bool(wandb),
            ),
        ]

        if early_stopping:
            cbs += [
                EarlyStopping(
                    monitor=early_stopping_monitor,
                    mode=early_stopping_mode,
                    patience=int(early_stopping_patience),
                    min_delta=float(early_stopping_min_delta),
                )
            ]
        if lr_floor is not None:
            cbs += [LRFloorStop(min_lr=lr_floor)]

        if wandb:
            dm_cfg = {
                "dataset_params": parse_neptune_dict(
                    deepcopy(self.D.configs["dataset_params"])
                ),
                "plan_train": parse_neptune_dict(
                    deepcopy(self.D.configs["plan_train"])
                ),
                "plan_valid": parse_neptune_dict(
                    deepcopy(self.D.configs["plan_valid"])
                ),
            }
            flat_cfg = _flatten_dict(dm_cfg, base="configs/datamodule")
            logger.experiment.config.update(flat_cfg, allow_val_change=True)
            cbs += [
                MinimalLearningRateMonitor(logging_interval="epoch"),
                WandbImageGridCallback(
                    classes=self.configs["model_params"]["out_channels"],
                    patch_size=self.configs["plan_train"]["patch_size"],
                    epoch_freq=max(1, int(wandb_grid_epoch_freq)),
                ),
                WandbLogBestCkpt(),
            ]

        if profiler:
            _ = AdvancedProfiler(dirpath=self.project.log_folder, filename="profiler")
            cbs += [DeviceStatsMonitor(cpu_stats=True)]

        return cbs, logger, profiler

    def init_dm(self):
        cache_rate = self.configs["dataset_params"]["cache_rate"]
        ds_type = self.configs["dataset_params"]["ds_type"]
        dm_cls = _dm_class_for_periodic_test(self.periodic_test)
        return dm_cls(
            self.project.project_title,
            configs=self.configs,
            batch_size=self.configs["dataset_params"]["batch_size"],
            cache_rate=cache_rate,
            device=self.configs["dataset_params"].get("device", "cuda"),
            ds_type=ds_type,
            start_n=self.start_n,
        )

    def load_dm(self, batch_size=None, override_dm_checkpoint=False):
        if override_dm_checkpoint:
            sd = torch.load(self.ckpt, map_location="cpu")
            backup_ckpt(self.ckpt)
            sd["datamodule_hyper_parameters"]["configs"] = self.configs
            headline("Overriding datamodule checkpoint.")
            out_fname = (self.run_name or "run") + ".ckpt"
            backup_path = Path(self.project.log_folder) / out_fname
            shutil.copy(self.ckpt, backup_path)
            torch.save(sd, self.ckpt)

        dm_from_ckpt = _dm_class_from_ckpt(self.ckpt)
        dm_wanted = _dm_class_for_periodic_test(self.periodic_test)
        dm_cls = dm_from_ckpt

        D = dm_cls.load_from_checkpoint(
            self.ckpt,
            project_title=self.project.project_title,
            batch_size=batch_size,
            map_location="cpu",
        )
        if batch_size:
            D.configs["dataset_params"]["batch_size"] = int(batch_size)
        if (dm_from_ckpt is DataManagerDualI) and (dm_wanted is DataManagerMultiI):
            headline(
                "Checkpoint datamodule is Dual (no test). Keeping checkpoint DM for compatibility."
            )
        return D

    def init_dm_unet(self, batch_size=None, override_dm_checkpoint=False):
        if self.ckpt:
            self.D = self.load_dm(
                batch_size=batch_size, override_dm_checkpoint=override_dm_checkpoint
            )
            headline(
                "Loading configs from checkpoint. Use override_dm_checkpoint=True to force trainer configs."
            )
            self.configs["dataset_params"] = self.D.configs["dataset_params"]
            for key in list(self.configs.keys()):
                if key in self.D.configs:
                    self.configs[key] = self.D.configs[key]
            self.N = self.load_model()
            self.configs["model_params"] = self.N.model_params
        else:
            self.D = self.init_dm()
            self.N = self.init_model()

    def init_model(self):
        return UNetManagerIncremental(
            project_title=self.project.project_title,
            configs=self.configs,
            lr=self.lr,
            sync_dist=False,
        )

    def load_model(self, map_location="cpu", **kwargs):
        try:
            model = UNetManagerIncremental.load_from_checkpoint(
                self.ckpt, map_location=map_location, strict=True, **kwargs
            )
        except RuntimeError:
            switch_ckpt_keys(self.ckpt)
            model = UNetManagerIncremental.load_from_checkpoint(
                self.ckpt, map_location=map_location, strict=True, **kwargs
            )
        return model

    def set_strategy(self, devices):
        if devices in (-1, "auto", None):
            n_gpus = torch.cuda.device_count()
            norm_devices = max(1, n_gpus)
        elif isinstance(devices, int):
            norm_devices = max(1, devices)
        elif isinstance(devices, (list, tuple)):
            norm_devices = max(1, len(devices))
        else:
            raise ValueError("devices must be int, list/tuple, -1, 'auto', or None")

        if norm_devices <= 1:
            strategy = "auto"
        else:
            strategy = "ddp_notebook" if in_ipython() else "ddp"

        self.devices = norm_devices
        self.strategy = strategy

    def set_lr(self, lr):
        if lr and not self.ckpt:
            self.lr = lr
        elif lr and self.ckpt:
            self.lr = lr
            sd = torch.load(self.ckpt, map_location="cpu")
            for g in sd["optimizer_states"][0]["param_groups"]:
                g["lr"] = float(self.lr)
            sd["lr_schedulers"][0]["_last_lr"] = [float(self.lr)]
            headline(f"Overriding checkpoint LR with {self.lr}")
            torch.save(sd, self.ckpt)
        elif lr is None and self.ckpt:
            state_dict = torch.load(self.ckpt, weights_only=False, map_location="cpu")
            self.lr = state_dict["lr_schedulers"][0]["_last_lr"][0]
        else:
            self.lr = self.configs["model_params"]["lr"]

    def maybe_alter_configs(self, batch_size, compiled):
        if batch_size is not None:
            self.configs["dataset_params"]["batch_size"] = int(batch_size)
        if compiled is not None:
            self.configs["model_params"]["compiled"] = bool(compiled)

    def qc_configs(self, configs: dict):
        ratios = configs["dataset_params"]["fgbg_ratio"]
        labels_all = configs["plan_train"]["labels_all"]
        if isinstance(ratios, list):
            assert len(ratios) == len(
                labels_all
            ), f"Class ratios ({len(ratios)}) do not match labels ({len(labels_all)})"
        else:
            assert isinstance(
                ratios, (int, float)
            ), "If no list is provided, fgbg_ratio must be int/float"


class IncrementalTrainer(IncrementalTrainerMinimal):
    pass


# %%

if __name__ == "__main__":
    run_fit = True
    device_id = 0
    batch_size = 4
    epochs = 600
    wandb = True
    run_name =None

    project = Project("nodes")
    cfg_maker = ConfigMaker(project)
    cfg_maker.setup(4)
    conf = cfg_maker.configs
    conf["dataset_params"]["cache_rate"] = 0.0
    conf["dataset_params"]["fold"] = 0

    trainer = IncrementalTrainerMinimal(project.project_title, conf, run_name=run_name)
    trainer.setup(
        compiled=False,
        batch_size=batch_size,
        devices=[device_id],
        epochs=epochs,
        batchsize_finder=False,
        profiler=False,
        wandb=wandb,
        tags=[],
        description="Partially trained up to 100 epochs",
        start_n=10,
    )
    trainer.N.compiled = False
    trainer.fit()
# %%
