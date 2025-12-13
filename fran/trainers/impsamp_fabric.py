# %%
from fran.managers import Project
from tqdm.auto import tqdm as pbar
from pathlib import Path
from pytorch_grad_cam import (
    GradCAM,
)
import ipdb

from fran.trainers.base import checkpoint_from_model_id
from fran.configs.parser import ConfigMaker
from utilz.string import info_from_filename
tr = ipdb.set_trace

from fran.managers import UNetManagerCraig
import os
from torch.cuda.amp import autocast
from monai.transforms import Decollated
from monai.transforms.io.array import SaveImage
import pandas as pd
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from tqdm import tqdm
import warnings

import shutil
from fran.trainers.impsamp import (
    fix_dict_keys,
    resolve_datamanager,
)
from fran.transforms.imageio import TorchWriter
from fran.transforms.inferencetransforms import ToCPUd
from utilz.imageviewers import ImageMaskViewer


def init_unet_trainer(project, config, lr):
    N = UNetManagerCraig(
        project=project,
        config=config,
        lr=lr,
    )
    return N
 

def load_unet_trainer(ckpt, project, config, lr, **kwargs):
    try:
        N = UNetManagerCraig.load_from_checkpoint(
            ckpt,
            project=project,
            dataset_params=config["dataset_params"],
            lr=lr,
            **kwargs,
        )
        print("Model loaded from checkpoint: ", ckpt)
    except:
        tr()
        ckpt_dict = torch.load(ckpt, weights_only=False)
        state_dict = ckpt_dict["state_dict"]
        ckpt_state = state_dict["state_dict"]
        ckpt_state_updated = fix_dict_keys(ckpt_state, "model", "model._orig_mod")
        # print(ckpt_state_updated.keys())
        state_dict_neo = state_dict.copy()
        state_dict_neo["state_dict"] = ckpt_state_updated
        ckpt_old = ckpt.str_replace("_bkp", "")
        ckpt_old = ckpt.str_replace(".ckpt", ".ckpt_bkp")
        torch.save(state_dict_neo, ckpt)
        shutil.move(ckpt, ckpt_old)

        N = UNetManagerCraig.load_from_checkpoint(
            ckpt,
            project=project,
            dataset_params=config["dataset_params"],
            lr=lr,
            **kwargs,
        )
    return N


class StoreInfo:
    def __init__(self, freq=1, run_name=None):

        # freq : decides every n batches to skip. Default: 1, i.e., every batch is stored
        self.freq = freq
        self.output_fldr = Path("/s/fran_storage/grads") / run_name
        self.dfs_train = []
        self.dfs_val = []

    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        # pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:

        if trainer.current_epoch == (trainer.max_epochs - 1):
            if self.freq % batch_idx == 0:
                self.dfs_train.append(self._common(outputs, batch))

    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        # pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:

        self.dfs_val.append(self._common(outputs, batch))

    def on_fit_end(self):
        pass

    def _common(self, outputs, batch):
        grad_L_z = outputs[1]
        # Gi_inside = model.grad_L_x * model.grad_sigma_z[0]
        # Gi_inside_normed_batch = [torch.linalg.norm(G) for G in Gi_inside]
        # # Gi_inside_normed = torch.stack(Gi_inside_normed_batch)
        # Gi_inside_normed.shape
        # L_rho = 5
        ks = batch["image"].meta["filename_or_obj"]
        # Gi = Gi_inside_normed_batch * L_rho
        if isinstance(ks, str):
            bs = 1
        else:
            bs = len(ks)
        if self.output_fldr is not None:
            batch["grad"] = grad_L_z
            self._save_tensors(batch)
        grad_L_z_batched = grad_L_z.reshape(bs, -1)
        grad_L_z_normed = torch.linalg.norm(grad_L_z_batched, dim=1)
        grads_listed = grad_L_z_normed.tolist()
        fns = batch["image"].meta["filename_or_obj"]
        dici = {"grad_norm": grads_listed, "fns": fns}
        df = pd.DataFrame(dici)
        return df

    def _save_tensors(self, batch):
        C = ToCPUd(keys=["image", "grad"])
        D = Decollated(keys=["image", "grad"], detach=True)
        batch = C(batch)
        mini_batches = D(batch)
        for i, item in enumerate(mini_batches):
            SI = SaveImage(
                output_ext="pt",
                output_dir=self.output_fldr,
                output_postfix=str(i),
                output_dtype="float32",
                writer=TorchWriter,
                separate_folder=False,
            )
            SG = SaveImage(
                output_ext="pt",
                output_dir=self.output_fldr,
                output_postfix=f"grad_{str(i)}",
                output_dtype="float32",
                writer=TorchWriter,
                separate_folder=False,
            )
            image, grad = item["image"], item["grad"]
            SI(image)
            SG(grad)
        #
        #
        # fns = img_tensor_batch.meta['filename_or_obj']
        # for img_tnsr,grad_tnsr,fn in zip(img_tensor_batch,grad_tensor_batch,fns):
        #     img_tnsr.meta['filename_or_obj'] = fn
        #     grad_tnsr.meta = img_tnsr.meta
        #     SI(img_tnsr)
        #     SG(grad_tnsr)


# customizing hooks because fabric treats model as a callback
class UNetManagerCraig(UNetManagerCraig):
    def on_train_batch_start(self, trainer, batch, batch_idx):
        pass

    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        # pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        pass

    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        # pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        pass


class Trainer:
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
        should_train=True,
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

    def fit(
        self,
        model: L.LightningModule,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
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
        train_loader = self.fabric.setup_dataloaders(
            train_loader, use_distributed_sampler=self.use_distributed_sampler
        )
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(
                val_loader, use_distributed_sampler=self.use_distributed_sampler
            )

        # setup model and optimizer
        if isinstance(self.fabric.strategy, L.fabric.strategies.fsdp.FSDPStrategy):
            # currently, there is no way to support fsdp with model.configure_optimizers in fabric
            # as it would require fabric to hold a reference to the model, which we don't want to.
            raise NotImplementedError("BYOT currently does not support FSDP")

        optimizer, scheduler_cfg = model.configure_optimizers().values()
        assert optimizer is not None
        model, optimizer = self.fabric.setup(model, optimizer)

        # assemble state (current epoch and global step will be added in save)
        state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}

        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)

                # check if we even need to train here
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

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

            self.save(state)

        # reset for next fit call
        self.fabric.call("on_fit_end")
        self.should_stop = False

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ] = None,
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
        iterable = self.progbar_wrapper(
            train_loader,
            total=min(len(train_loader), limit_batches),
            desc=f"Epoch {self.current_epoch}",
        )

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break
            self.fabric.call("on_train_batch_start", self, batch, batch_idx)
            # check if optimizer should step in gradient accumulation
            should_optim_step = self.global_step % self.grad_accum_steps == 0
            if should_optim_step:
                # currently only supports a single optimizer
                self.fabric.call("on_before_optimizer_step", optimizer)

                # optimizer step runs train step internally through closure
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
                # gradient accumulation -> no optimizer step
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)
            self.fabric.call(
                "on_train_batch_end", self, self._current_train_return, batch, batch_idx
            )
            # this guard ensures, we only step the scheduler once per global step
            if should_optim_step:
                self.step_scheduler(
                    model, scheduler_cfg, level="step", current_value=self.global_step
                )

            self._format_iterable(iterable, self._current_train_return[0], "train")
            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)
            # stopping criterion on step level
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break
        self.fabric.call("on_train_epoch_end")

    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
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
        if val_loader is not None and not is_overridden(
            "validation_step", _unwrap_objects(model)
        ):
            L.fabric.utilities.rank_zero_warn(
                "Your LightningModule does not have a validation_step implemented, "
                "but you passed a validation dataloder. Skipping Validation."
            )
            return

        if not is_overridden("on_validation_model_eval", _unwrap_objects(model)):
            # pass
            model.eval()
        else:
            self.fabric.call("on_validation_model_eval")  # calls `model.eval()`

        print("==" * 50)
        print("GRAD ENABLED")
        torch.set_grad_enabled(True)

        self.fabric.call("on_validation_epoch_start")

        iterable = self.progbar_wrapper(
            val_loader, total=min(len(val_loader), limit_batches), desc="Validation"
        )

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            out = self.validation_step(model, batch, batch_idx)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())
            self.fabric.call("on_validation_batch_end", self, out, batch, batch_idx)
            self._current_val_return = out
            self._format_iterable(iterable, self._current_val_return[0], "val")

        self.fabric.call("on_validation_epoch_end")

        if not is_overridden("on_validation_model_train", _unwrap_objects(model)):
            model.train()
        else:
            self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)

    def validation_step(self, model: L.LightningModule, batch: Any, batch_idx: int):
        """The default validation step. Override if you need to do anything extra"""
        with autocast():

            out = model.validation_step(batch, batch_idx)
        return out

    def training_step(
        self, model: L.LightningModule, batch: Any, batch_idx: int
    ) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        with autocast():
            outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(
                batch, batch_idx=batch_idx
            )
            loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

        self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(
            outputs, dtype=torch.Tensor, function=lambda x: x.detach()
        )

        return loss

    def step_scheduler(
        self,
        model: L.LightningModule,
        scheduler_cfg: Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ],
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
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update("train_loss", self._current_train_return)

        if isinstance(self._current_train_return, tuple):
            possible_monitor_vals.update("train_loss", self._current_train_return[0])
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update(
                {
                    "train_" + k: v
                    for k, v in self._current_train_return["losses_for_logging"].items()
                }
            )

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update("val_loss", self._current_val_return)

        if isinstance(self._current_val, tuple):
            possible_monitor_vals.update("val_loss", self._current_val_return[0])
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update(
                {
                    "val_" + k: v
                    for k, v in self._current_val_return["losses_for_logging"].items()
                }
            )

        try:
            monitor = possible_monitor_vals[
                cast(Optional[str], scheduler_cfg["monitor"])
            ]
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

        self.fabric.save(
            os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"),
            state,
        )

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

    def _parse_optimizers_schedulers(self, configure_optim_output) -> Tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        """
        _lr_sched_defaults = {
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        if isinstance(configure_optim_output, L.fabric.utilities.types.LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(
                isinstance(_opt_cand, L.fabric.utilities.types.Optimizable)
                for _opt_cand in configure_optim_output
            ):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            if all(
                isinstance(_lr_cand, (L.fabric.utilities.types.LRScheduler, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return (
                        None,
                        self._parse_optimizers_schedulers(configure_optim_output[0])[1],
                    )

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
        prog_bar,
        candidates: Optional[
            Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]
        ],
        prefix: str,
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(
                candidates, torch.Tensor, lambda x: x.item()
            )
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)


# %%

# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR>
if __name__ == "__main__":

    import torch

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")

    fn_results = "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/results/summary_LITS-933.xlsx"
    df_res = pd.read_excel(fn_results)
    from fran.utils.common import *

    project_title = "litsmc"
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_config_wholeimage.xlsx"
    )
    configuration_filename = "/s/fran_storage/projects/litsmc/experiment_config.xlsx"
    configuration_filename = None

    config = ConfigMaker(proj, ).config

    # conf['model_params']['lr']=1e-3

    # conf['dataset_params']['plan']=5
    device_id = 0
    # run_name = "LITS-1007"
    # device_id = 0
    run_totalseg = "LITS-1025"
    # run_litsmc = "LITS-1018"
    run_litsmc = "LITS-1131"
    run_empty = None
    run_name = run_empty
    bs = 1  # 5 is good if LBD with 2 samples per case
    run_name =run_litsmc
    if run_name is not None:
        ckpt = checkpoint_from_model_id(run_name, sort_method="last")
    else:
        ckpt = None
    cbs = []
    cbs = [StoreInfo(run_name="neo" if run_name is None else run_name, freq=5)]
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = False
    tags = []
    description = f""
    config["dataset_params"]["batch_size"] = bs
# %%
# SECTION:-------------------- Dataloaders-------------------------------------------------------------------------------------- <CR> <CR> <CR>

    DMClass = resolve_datamanager(config["plan_valid"]["mode"])
    if ckpt:
        # D = DMClass.load_from_checkpoint(ckpt, project=proj)
        N = load_unet_trainer(ckpt=ckpt, project=proj, config=config, lr=1e-2)
    else:
        N = init_unet_trainer(proj, config, lr=config["model_params"]["lr"])
    D = DMClass(
            proj,
            
            config=config,
            batch_size=config["dataset_params"]["batch_size"],
            cache_rate=0,
           
        )
    N.create_loss_fnc()
# %%
#SECTION:-------------------- From Folder--------------------------------------------------------------------------------------
# %%
    fldr = Path("/s/xnat_shadow/crc/tensors/lbd_plan3")
    fn_stats =fldr/("stats.csv")
    D = DMClass.from_folder(data_folder=fldr, project=proj,config=config,batch_size=2,split="train")
    D.prepare_data()
    D.setup()
    dl = D.dl

    N.lr = 1e-2
    devices=[1]
    N.model.to('cuda')
# %%
    def stat_dict(grad_L_z,suffix=""):
        return        {
            'max'+suffix: grad_L_z.max().item(),
            'min'+suffix: grad_L_z.min().item(), 
            'mean'+suffix:grad_L_z.mean().item(),
            'sum'+suffix: grad_L_z.sum().item(),
            'std'+suffix: grad_L_z.std().item(),
            'norm'+suffix:torch.linalg.norm(grad_L_z).item(),
        }

#SECTION:-------------------- Loops--------------------------------------------------------------------------------------
    all_stats=[]
# %%
    for i, batch in pbar(enumerate(dl)):
        batch['image'] =batch['image'].to('cuda')
        batch['lm'] =batch['lm'].to('cuda')

        inputs, target = batch["image"], batch["lm"]
        pred = N.forward(
            inputs
        )  # N.pred so that NeptuneImageGridCallback can use it

        pred, target = N.maybe_apply_ds_scales(pred, target)
        loss = N.loss_fnc(pred, target)
        loss_dict = N.loss_fnc.loss_dict
        grad_L_z = N.loss_fnc.grad_L_z
        grad_L_z_ch23 = N.loss_fnc.grad_L_z_ch23
        fn = inputs.meta['filename_or_obj']
        cid = info_from_filename(Path(fn).name,False)['case_id']
        inputs.meta
        stats_grad_L_z = stat_dict(grad_L_z)
        stats_ch23 = stat_dict(grad_L_z_ch23,suffix="_ch23")
        info = {
            'fn': inputs.meta['filename_or_obj'],
            'case_id':cid,
        }
        final_dict = stats_grad_L_z | stats_ch23 | info


        all_stats.append(final_dict)
    # print(stats)
    df = pd.DataFrame(all_stats)
    df =df.merge(df_res[['case_id','dsc','lesions_gt','gt_vol_tot']], on='case_id',how='left')
    df.to_csv(fn_stats,index=False)
# %%
    n=0
    ImageMaskViewer([inputs[n][0].detach().cpu(),pred[0][n][3].detach().cpu()])
# %%

    preds[0].shape
    n=1
# %%
# SECTION:-------------------- Train-------------------------------------------------------------------------------------- <CR>
    Tm = Trainer(callbacks=cbs, precision="bf16-mixed", should_train=False,devices = devices)
# %%
    Tm.fit(model=N, train_loader=train_dl, val_loader=val_dl)
# SECTION:-------------------- GRADCAM-------------------------------------------------------------------------------------- <CR>

# %%
    class CAMTarget:
        def __call__(self, model_output):
            output = model_output[0]

            output_tumour = model_output[:, 2, :]
            return output_tumour.sum()

# %%

    iteri = iter(val_dl)
    batch = next(iteri)
    input_tensor = batch["image"][3:7, :]
    target_layers = [N.model.tu[-1]]
    # targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
# %%

    input_tensor = input_tensor.to("cuda")
    with autocast():

        output = N.model(input_tensor)
    targets = [CAMTarget()]
# %%
    with GradCAM(
        model=N.model,
        target_layers=target_layers,
    ) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
# %%
    ind = 3
    ImageMaskViewer([input_tensor[ind, 0], grayscale_cam[ind]])

# %%
    S = cbs[0]
    dfs_train = pd.concat(S.dfs_train)
    dfs_train.to_csv("train_prefitted.csv", index=False)

    dfs_val = pd.concat(S.dfs_val)
    dfs_val.to_csv("val_prefitted.csv", index=False)
# %%
    iteri = iter(train_dl)
    batch = next(iteri)
    print(batch["image"].meta["filename_or_obj"])
# %%

    limit_batches: Union[int, float] = float("inf")
    val_loader = val_dl
    iterable = Tm.progbar_wrapper(
        val_loader, total=min(len(val_loader), limit_batches), desc="Validation"
    )
    for batch_idx, batch in enumerate(iterable):
        batch = next(iter(iterable))
        batch_idx = 0
        batch["image"] = batch["image"].to("cuda:0")
        # end epoch if stopping training completely or max batches for this epoch reached
        if Tm.should_stop or batch_idx >= limit_batches:
            break

        Tm.fabric.call("on_validation_batch_start", batch, batch_idx)

        out = model.validation_step(batch, batch_idx)
        # avoid gradients in stored/accumulated values -> prevents potential OOM
        out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

        Tm.fabric.call("on_validation_batch_end", out, batch, batch_idx)
        Tm._current_val_return = out

        Tm._format_iterable(iterable, Tm._current_val_return, "val")
# %%
    A = np.array([[1, 2, 3]])  # 1x3 matrix
    B = np.array([[4, 5, 6],   # 3x3 matrix
                  [7, 8, 9],
                  [10, 11, 12]])

# Performing the matrix multiplication
    A*B
    result = np.matmul(A, B)
# %%
# %%
#SECTION:-------------------- TROUBLESHOOTING--------------------------------------------------------------------------------------
# %%
    accelerator = "auto"
    ckpt_path=None
    strategy = "auto" 
    precision = "bf16-mixed"
    plugins = None
    loggers = None
    grad_accum_steps = 1
    max_epochs = 1
    max_steps = None
    limit_train_batches = float("inf")
    limit_val_batches = float("inf")
    validation_frequency = 1
    use_distributed_sampler = True
    checkpoint_dir = "./checkpoints"
    checkpoint_frequency = 1
    callbacks=cbs
    precision="bf16-mixed"

    devices = [0]
    should_train=False
# %%
    # Initialize Fabric
    Tm.fabric = L.Fabric(
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        precision=precision,
        plugins=plugins,
        callbacks=callbacks,
        loggers=loggers,
    )
    
    # Set trainer attributes
    Tm.should_train = should_train
    Tm.global_step = 0
    Tm.grad_accum_steps = grad_accum_steps
    Tm.current_epoch = 0
    Tm.max_epochs = max_epochs
    Tm.max_steps = max_steps
    Tm.should_stop = False
    Tm.limit_train_batches = limit_train_batches
    Tm.limit_val_batches = limit_val_batches
    Tm.validation_frequency = validation_frequency
    Tm.use_distributed_sampler = use_distributed_sampler
    Tm._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
    Tm._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}
    Tm.checkpoint_dir = checkpoint_dir
    Tm.checkpoint_frequency = checkpoint_frequency

# %%

    model =N
    val_loader=val_dl
    train_loader=train_dl
    Tm.fabric.launch()
    # setup dataloaders
    train_loader = Tm.fabric.setup_dataloaders(
        train_loader, use_distributed_sampler=Tm.use_distributed_sampler
    )
    if val_loader is not None:
        val_loader = Tm.fabric.setup_dataloaders(
            val_loader, use_distributed_sampler=Tm.use_distributed_sampler
        )

# %%
    iteri =iter(train_loader)
    batch = next(iteri)
    batch['image'].dtype
# %%
    # setup model and optimizer
    if isinstance(Tm.fabric.strategy, L.fabric.strategies.fsdp.FSDPStrategy):
        # currently, there is no way to support fsdp with model.configure_optimizers in fabric
        # as it would require fabric to hold a reference to the model, which we don't want to.
        raise NotImplementedError("BYOT currently does not support FSDP")

    optimizer, scheduler_cfg = model.configure_optimizers().values()
    assert optimizer is not None
    model, optimizer = Tm.fabric.setup(model, optimizer)

    # assemble state (current epoch and global step will be added in save)
    state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}

    # load last checkpoint if available
    if ckpt_path is not None and os.path.isdir(ckpt_path):
        latest_checkpoint_path = Tm.get_latest_checkpoint(Tm.checkpoint_dir)
        if latest_checkpoint_path is not None:
            Tm.load(state, latest_checkpoint_path)

            # check if we even need to train here
            if (
                Tm.max_epochs is not None
                and Tm.current_epoch >= Tm.max_epochs
            ):
                Tm.should_stop = True

    Tm.fabric.call("on_fit_start")
    while not Tm.should_stop:
        # if Tm.should_train:
        #     Tm.train_loop(
        #         model,
        #         optimizer,
        #         train_loader,
        #         limit_batches=Tm.limit_train_batches,
        #         scheduler_cfg=scheduler_cfg,
        #     )

        if Tm.should_validate:
            Tm.val_loop(model, val_loader, limit_batches=Tm.limit_val_batches)

        if Tm.should_train:
            Tm.step_scheduler(
                model,
                scheduler_cfg,
                level="epoch",
                current_value=Tm.current_epoch,
            )

        Tm.current_epoch += 1

        # stopping condition on epoch level
        if Tm.max_epochs is not None and Tm.current_epoch >= Tm.max_epochs:
            Tm.should_stop = True

        Tm.save(state)

    # reset for next fit call
    Tm.fabric.call("on_fit_end")
    Tm.should_stop = False


# %%
    model = Tm.model
    val_loader
    limit_batches=Tm.limit_val_batches

    # no validation but warning if val_loader was passed, but validation_step not implemented
    if val_loader is not None and not is_overridden(
        "validation_step", _unwrap_objects(model)
    ):
        L.fabric.utilities.rank_zero_warn(
            "Your LightningModule does not have a validation_step implemented, "
            "but you passed a validation dataloder. Skipping Validation."
        )

    if not is_overridden("on_validation_model_eval", _unwrap_objects(model)):
        # pass
        model.eval()
    else:
        Tm.fabric.call("on_validation_model_eval")  # calls `model.eval()`

    print("==" * 50)
    print("GRAD ENABLED")
    torch.set_grad_enabled(True)

    Tm.fabric.call("on_validation_epoch_start")

    iterable = Tm.progbar_wrapper(
        val_loader, total=min(len(val_loader), limit_batches), desc="Validation"
    )

    for batch_idx, batch in enumerate(iterable):
        # end epoch if stopping training completely or max batches for this epoch reached
        if Tm.should_stop or batch_idx >= limit_batches:
            break

        Tm.fabric.call("on_validation_batch_start", batch, batch_idx)

        out = Tm.validation_step(model, batch, batch_idx)
        # avoid gradients in stored/accumulated values -> prevents potential OOM
        out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())
        Tm.fabric.call("on_validation_batch_end", Tm, out, batch, batch_idx)
        Tm._current_val_return = out
        Tm._format_iterable(iterable, Tm._current_val_return[0], "val")

    Tm.fabric.call("on_validation_epoch_end")

    if not is_overridden("on_validation_model_train", _unwrap_objects(model)):
        model.train()
    else:
        Tm.fabric.call("on_validation_model_train")
    torch.set_grad_enabled(True)


# %%
