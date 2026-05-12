
# %%
import uuid
from copy import deepcopy
from os import path
from pathlib import Path
from typing import Any, Optional

from fran.callback.base import BatchSizeSafetyMargin
from fran.callback.case_recorder import CaseIDRecorder, infer_labels_and_update_out_channels
from fran.callback.wandb.wandb import WandbLogBestCkpt
from fran.managers.data.main import DataManagerDual
from fran.managers.data.run_through import DataManagerRT, DataManagerRTBTfms
from fran.trainers.trainer import Trainer, TrainerL
from fran.utils.batch_size_scaling import _adjust_batch_size, _reset_progress, _try_loop_run
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.logger import DummyLogger
from lightning.pytorch.utilities.memory import garbage_collection_cuda, is_oom_error


def _rt_reset_dataloaders(trainer: TrainerL) -> None:
    loop = trainer._active_loop
    assert loop is not None
    loop._combined_loader = None
    loop.setup_data()


def _rt_scale_batch_dump_params(trainer: TrainerL) -> dict[str, Any]:
    loop = trainer._active_loop
    assert loop is not None
    return {
        "loggers": trainer.loggers,
        "callbacks": trainer.callbacks,
        "max_steps": trainer.max_steps,
        "limit_train_batches": trainer.limit_train_batches,
        "loop_state_dict": deepcopy(loop.state_dict()),
    }


def _rt_scale_batch_reset_params(trainer: TrainerL, steps_per_trial: int) -> None:
    trainer.logger = DummyLogger() if trainer.logger is not None else None
    trainer.callbacks = []
    trainer.limit_train_batches = 1.0
    trainer.fit_loop.epoch_loop.max_steps = steps_per_trial


def _rt_scale_batch_restore_params(trainer: TrainerL, params: dict[str, Any]) -> None:
    trainer.loggers = params["loggers"]
    trainer.callbacks = params["callbacks"]

    loop = trainer._active_loop
    assert loop is not None
    loop.epoch_loop.max_steps = params["max_steps"]
    trainer.limit_train_batches = params["limit_train_batches"]
    loop.load_state_dict(deepcopy(params["loop_state_dict"]))
    loop.restarting = False
    _rt_reset_dataloaders(trainer)
    loop.reset()


def _run_power_scaling_rt(
    trainer: TrainerL,
    new_size: int,
    batch_arg_name: str,
    max_trials: int,
    params: dict[str, Any],
) -> int:
    any_success = False
    for _ in range(max_trials):
        garbage_collection_cuda()
        _reset_progress(trainer)
        try:
            _try_loop_run(trainer, params)
            new_size, changed = _adjust_batch_size(
                trainer, batch_arg_name, factor=2.0, desc="succeeded"
            )
            if not changed:
                break
            _rt_reset_dataloaders(trainer)
            any_success = True
        except RuntimeError as exception:
            if is_oom_error(exception):
                garbage_collection_cuda()
                new_size, _ = _adjust_batch_size(
                    trainer, batch_arg_name, factor=0.5, desc="failed"
                )
                _rt_reset_dataloaders(trainer)
                if any_success:
                    break
            else:
                raise
    return new_size


def _run_binary_scaling_rt(
    trainer: TrainerL,
    new_size: int,
    batch_arg_name: str,
    max_trials: int,
    params: dict[str, Any],
) -> int:
    low = 1
    high = None
    count = 0
    while True:
        garbage_collection_cuda()
        _reset_progress(trainer)
        try:
            _try_loop_run(trainer, params)
            count += 1
            if count > max_trials:
                break
            low = new_size
            if high:
                if high - low <= 1:
                    break
                midval = (high + low) // 2
                new_size, changed = _adjust_batch_size(
                    trainer, batch_arg_name, value=midval, desc="succeeded"
                )
            else:
                new_size, changed = _adjust_batch_size(
                    trainer, batch_arg_name, factor=2.0, desc="succeeded"
                )
            if not changed:
                break
            _rt_reset_dataloaders(trainer)
        except RuntimeError as exception:
            if is_oom_error(exception):
                garbage_collection_cuda()
                high = new_size
                midval = (high + low) // 2
                new_size, _ = _adjust_batch_size(
                    trainer, batch_arg_name, value=midval, desc="failed"
                )
                _rt_reset_dataloaders(trainer)
                if high - low <= 1:
                    break
            else:
                raise
    return new_size


def _scale_batch_size_rt(
    trainer: TrainerL,
    mode: str,
    steps_per_trial: int,
    init_val: int,
    max_trials: int,
    batch_arg_name: str,
) -> int | None:
    if trainer.fast_dev_run:
        return None

    ckpt_path = path.join(
        trainer.default_root_dir, f".scale_batch_size_{uuid.uuid4()}.ckpt"
    )
    trainer.save_checkpoint(ckpt_path)
    params = _rt_scale_batch_dump_params(trainer)
    try:
        _rt_scale_batch_reset_params(trainer, steps_per_trial)

        if trainer.progress_bar_callback:
            trainer.progress_bar_callback.disable()

        new_size, _ = _adjust_batch_size(trainer, batch_arg_name, value=init_val)
        if mode == "power":
            new_size = _run_power_scaling_rt(
                trainer, new_size, batch_arg_name, max_trials, params
            )
        else:
            new_size = _run_binary_scaling_rt(
                trainer, new_size, batch_arg_name, max_trials, params
            )

        garbage_collection_cuda()
        return new_size
    finally:
        _rt_scale_batch_restore_params(trainer, params)

        if trainer.progress_bar_callback:
            trainer.progress_bar_callback.enable()

        trainer._checkpoint_connector.restore(ckpt_path)
        trainer.strategy.remove_checkpoint(ckpt_path)


class BatchSizeFinderRT(Callback):
    """Run-through batch size finder that probes training only."""

    SUPPORTED_MODES = ("power", "binsearch")

    def __init__(
        self,
        mode: str = "power",
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
        batch_arg_name: str = "batch_size",
    ) -> None:
        mode = mode.lower()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"`mode` should be either of {self.SUPPORTED_MODES}")
        self.optimal_batch_size = init_val
        self._mode = mode
        self._steps_per_trial = steps_per_trial
        self._init_val = init_val
        self._max_trials = max_trials
        self._batch_arg_name = batch_arg_name

    def scale_batch_size(self, trainer, pl_module) -> None:
        restore = trainer._checkpoint_connector.restore

        def restore_trusted_temp_checkpoint(checkpoint_path=None, weights_only=None):
            return restore(checkpoint_path, weights_only=False)

        trainer._checkpoint_connector.restore = restore_trusted_temp_checkpoint
        try:
            new_size = _scale_batch_size_rt(
                trainer=trainer,
                mode=self._mode,
                steps_per_trial=self._steps_per_trial,
                init_val=self._init_val,
                max_trials=self._max_trials,
                batch_arg_name=self._batch_arg_name,
            )
        finally:
            trainer._checkpoint_connector.restore = restore
        self.optimal_batch_size = int(new_size)
        trainer.datamodule.batch_size = int(new_size)
        pl_module.batch_size = int(new_size)

    def on_fit_start(self, trainer, pl_module) -> None:
        self.scale_batch_size(trainer, pl_module)


class CaseIDRecorderRT(CaseIDRecorder):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if (epoch > 0 and epoch % self.freq == 0) or self.incrementing is True:
            self.dfs["epoch"] = epoch
            if self.incrementing is False:
                self._store(trainer, "train", self.loss_dicts_train, epoch)
            else:
                self._store(trainer, "train2", self.loss_dicts_train2, epoch)
            trainer.dfs = self.dfs
            self.reset()


class WandbLogBestCkptRT(WandbLogBestCkpt):
    def on_train_epoch_end(self, trainer, pl_module):
        ckpt_best = trainer.checkpoint_callback.best_model_path
        ckpt_last = trainer.checkpoint_callback.last_model_path
        run = trainer.logger.experiment
        run.summary.update(
            {
                "training/last_model_path": ckpt_last,
                "training/best_model_path": ckpt_best,
            }
        )


class TrainerRT(Trainer):
    """Compatibility shim for run-through training on top of the base Trainer."""

    def __init__(
        self,
        project_title,
        configs,
        run_name=None,
        ckpt: Optional[str | Path] = None,
    ):
        """Initialize the base Trainer in run-through mode."""
        super().__init__(
            project_title=project_title,
            configs=configs,
            run_name=run_name,
            ckpt=ckpt,
            run_through=True,
        )

    def setup(
        self,
        batch_size=None,
        train_indices=None,
        logging_freq=25,
        lr=None,
        devices=1,
        compiled=None,
        wandb=True,
        profiler=False,
        debug: bool = False,
        cbs=[],
        tags=[],
        description="",
        epochs=600,
        batchsize_finder=False,
        override_dm_checkpoint=False,
        lr_floor=None,
        wandb_grid_epoch_freq: int = 5,
        permanent_checkpoint_every_n_epochs: int = 100,
        batch_tfms: bool = False,
        val_device: str = "cuda",
    ):
        """Run-through setup without validation or early-stopping public arguments. 
        no dual-ssd option"""

        return super().setup(
            batch_size=batch_size,
            train_indices=train_indices,
            logging_freq=logging_freq,
            lr=lr,
            devices=devices,
            compiled=compiled,
            wandb=wandb,
            profiler=profiler,
            debug=debug,
            cbs=cbs,
            tags=tags,
            description=description,
            epochs=epochs,
            batchsize_finder=batchsize_finder,
            override_dm_checkpoint=override_dm_checkpoint,
            early_stopping=False,
            lr_floor=lr_floor,
            wandb_grid_epoch_freq=wandb_grid_epoch_freq,
            permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
            dual_ssd=False,
            batch_tfms=batch_tfms,
            val_device=val_device,
        )

    def init_cbs(
        self,
        cbs,
        wandb,
        batchsize_finder,
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
        permanent_checkpoint_every_n_epochs: int = 100,
    ):
        cbs, logger, profiler = super().init_cbs(
            cbs=cbs,
            wandb=wandb,
            batchsize_finder=False,
            profiler=profiler,
            tags=tags,
            description=description,
            early_stopping=early_stopping,
            early_stopping_monitor=early_stopping_monitor,
            early_stopping_mode=early_stopping_mode,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            lr_floor=lr_floor,
            wandb_grid_epoch_freq=wandb_grid_epoch_freq,
            permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
        )
        if batchsize_finder:
            cbs[1:1] = [
                BatchSizeFinderRT(batch_arg_name="batch_size", mode="binsearch"),
                BatchSizeSafetyMargin(),
            ]
        return cbs, logger, profiler

    def resolve_orchestrator_class(self, batch_tfms: Optional[bool] = None):
            return DataManagerRTBTfms if batch_tfms else DataManagerRT


    def init_dm(self):
        dm_class = self.resolve_orchestrator_class(batch_tfms=self.batch_tfms)
        dm =dm_class(
                project_title=self.project.project_title,
                configs=self.configs,
                batch_size=self.configs["dataset_params"]["batch_size"],
                cache_rate=self.configs["dataset_params"]["cache_rate"],
                device=self.configs["dataset_params"].get("device", "cuda"),
                ds_type=self.configs["dataset_params"].get("ds_type"),
                train_indices=self.train_indices,
                debug=self.debug,
                batch_tfms=self.batch_tfms,
            )
        
        labels_all = self.configs["plan_train"].get("labels_all")
        if not labels_all:
            infer_labels_and_update_out_channels(dm=dm, configs=self.configs)
        return dm

# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
if __name__ == "__main__":
    from fran.configs.parser import ConfigMaker
    from fran.managers import Project

    P = Project("kits23")
    C = ConfigMaker(P)
    C.setup(2)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0
    run_name= "KITS23-SIRIG"
    
# %%

    Tm = TrainerRT(
        project_title=P.project_title,
        configs=conf,
        run_name=None,
    )
    Tm.setup(
        compiled=False,
        train_indices=8,
        batch_size=2,
        devices=[0],
        epochs=20,
        batchsize_finder=False,
        profiler=False,
        wandb=True,
        tags=["scratch", "runthrough"],
        description="trainer_rt scratch run",
    )
# %%
    Tm.fit()

    def setup(
        self,
        batch_size=None,
        train_indices=None,
        logging_freq=25,
        lr=None,
        devices=1,
        compiled=None,
        wandb=True,
        profiler=False,
        debug: bool = False,
        cbs=[],
        tags=[],
        description="",
        epochs=600,
        batchsize_finder=False,
        override_dm_checkpoint=False,
        wandb_grid_epoch_freq: int = 5,
        permanent_checkpoint_every_n_epochs: int = 100,
        dual_ssd: bool = False,
        batch_tfms: bool = False,
    ):
        """Run-through setup excludes validation and early-stopping kwargs."""
        return super().setup(
            batch_size=batch_size,
            train_indices=train_indices,
            logging_freq=logging_freq,
            lr=lr,
            devices=devices,
            compiled=compiled,
            wandb=wandb,
            profiler=profiler,
            debug=debug,
            cbs=cbs,
            tags=tags,
            description=description,
            epochs=epochs,
            batchsize_finder=batchsize_finder,
            override_dm_checkpoint=override_dm_checkpoint,
            early_stopping=False,
            wandb_grid_epoch_freq=wandb_grid_epoch_freq,
            permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
            dual_ssd=dual_ssd,
            batch_tfms=batch_tfms,
        )
