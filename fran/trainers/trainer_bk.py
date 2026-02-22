from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from lightning.pytorch import Trainer as TrainerL
from lightning.pytorch.callbacks import (
    BatchSizeFinder,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.profilers import AdvancedProfiler

from fran.callback.base import BatchSizeSafetyMargin
from fran.callback.incremental import LRFloorStop
from fran.callback.test import PeriodicTest
from fran.callback.wandb import WandbImageGridCallback, WandbLogBestCkpt
from fran.configs.parser import parse_neptune_dict
from fran.managers.wandb import WandbManager
from fran.trainers.trainer import Trainer
from utilz.stringz import headline


def _flatten_dict(d: dict, base: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{base}/{k}" if base else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


class TrainerBK(Trainer):
    """Trainer variant with W&B logging/callback plumbing."""

    def _ensure_local_ckpt_on_wandb_resume(self, logger: WandbManager | None) -> None:
        """
        If a W&B run is resumed, ensure trainer also resumes from a local checkpoint.
        This prevents silently resuming metrics/logging while restarting model weights.
        """
        if logger is None or not self.run_name:
            return

        # Already good.
        if self.ckpt is not None and Path(self.ckpt).exists():
            headline(f"W&B resume: using local checkpoint {self.ckpt}")
            return

        wb_ckpt = logger.model_checkpoint
        if wb_ckpt:
            wb_ckpt_path = Path(wb_ckpt)
            if wb_ckpt_path.exists():
                self.ckpt = wb_ckpt_path
                headline(f"W&B resume: using checkpoint from summary {self.ckpt}")
                return

        # Try to mirror/download and re-read summary path.
        try:
            logger.download_checkpoints()
            wb_ckpt = logger.model_checkpoint
            if wb_ckpt and Path(wb_ckpt).exists():
                self.ckpt = Path(wb_ckpt)
                headline(f"W&B resume: downloaded local checkpoint {self.ckpt}")
                return
        except Exception as e:
            headline(f"W&B resume: checkpoint sync attempt failed: {e}")

        raise RuntimeError(
            "W&B run resume requested, but no local checkpoint is available. "
            "Refusing to continue to avoid resuming logs without resuming model state."
        )

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
        cbs=[],
        tags=[],
        description="",
        epochs=600,
        batchsize_finder=False,
        override_dm_checkpoint=False,
        early_stopping=False,
        early_stopping_monitor="val0_loss_dice",
        early_stopping_mode="min",
        early_stopping_patience=30,
        early_stopping_min_delta=0.0,
        lr_floor=None,
        wandb_grid_epoch_freq: int = 5,
    ):
        self.periodic_test = int(periodic_test)
        self.maybe_alter_configs(batch_size, compiled)
        self.set_lr(lr)

        has_cuda = torch.cuda.is_available()
        if has_cuda:
            self.set_strategy(devices)
            trainer_devices = devices
            accelerator = "gpu"
            strategy = self.strategy
        else:
            self.devices = 1
            self.sync_dist = False
            self.strategy = "auto"
            trainer_devices = 1
            accelerator = "cpu"
            strategy = "auto"

        self.init_dm_unet(epochs, batch_size, override_dm_checkpoint)

        # Keep loop/step state consistent on resumed runs.
        # BatchSizeFinder runs probe fits and restores a temp checkpoint,
        # which resets progress counters (e.g., epoch shown as 1).
        if self.ckpt is not None and batchsize_finder:
            headline(
                "Resumed run detected: disabling BatchSizeFinder to preserve checkpoint epoch/step state."
            )
            batchsize_finder = False

        cbs, logger, profiler = self.init_cbs(
            cbs=cbs,
            neptune=wandb,
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
        self._ensure_local_ckpt_on_wandb_resume(logger)
        self.D.prepare_data()

        self.trainer = TrainerL(
            callbacks=cbs,
            accelerator=accelerator,
            devices=trainer_devices,
            precision="bf16-mixed" if has_cuda else 32,
            profiler=profiler,
            logger=logger,
            max_epochs=epochs,
            log_every_n_steps=logging_freq,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            default_root_dir=self.project.checkpoints_parent_folder,
            strategy=strategy,
        )

    def init_cbs(
        self,
        cbs,
        neptune,
        batchsize_finder,
        periodic_test,
        profiler,
        tags,
        description="",
        early_stopping=False,
        early_stopping_monitor="val0_loss_dice",
        early_stopping_mode="min",
        early_stopping_patience=30,
        early_stopping_min_delta=0.0,
        lr_floor=None,
        wandb_grid_epoch_freq: int = 5,
    ):
        if batchsize_finder==True:
            cbs += [
                BatchSizeFinder(batch_arg_name="batch_size", mode="binsearch"),
                BatchSizeSafetyMargin(buffer=1),
            ]

        if periodic_test > 0:
            cbs += [PeriodicTest(every_n_epochs=periodic_test, limit_batches=50)]

        cbs += [
            ModelCheckpoint(
                save_last=True,
                monitor="val0_loss",
                every_n_epochs=10,
                filename="{epoch}-{val_loss:.2f}",
                enable_version_counter=True,
                auto_insert_metric_name=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=3),
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

        logger = None
        if neptune:
            logger = WandbManager(
                project=self.project,
                run_id=self.run_name,
                log_model_checkpoints=False,
                tags=tags,
                notes=description,
            )
            dm_cfg = {
                "dataset_params": parse_neptune_dict(deepcopy(self.D.configs["dataset_params"])),
                "plan_train": parse_neptune_dict(deepcopy(self.D.configs["plan_train"])),
                "plan_valid": parse_neptune_dict(deepcopy(self.D.configs["plan_valid"])),
            }
            flat_cfg = _flatten_dict(dm_cfg, base="configs/datamodule")
            logger.experiment.config.update(flat_cfg, allow_val_change=True)
            cbs += [
                WandbImageGridCallback(
                    classes=self.configs["model_params"]["out_channels"],
                    patch_size=self.configs["plan_train"]["patch_size"],
                    epoch_freq=max(1, int(wandb_grid_epoch_freq)),
                ),
                WandbLogBestCkpt(),
            ]

        if profiler:
            profiler = AdvancedProfiler(dirpath=self.project.log_folder, filename="profiler")
            cbs += [DeviceStatsMonitor(cpu_stats=True)]
        else:
            profiler = None

        return cbs, logger, profiler
