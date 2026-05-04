from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
from fran.callback.base import BatchSizeSafetyMargin
from fran.callback.case_recorder import (
    CaseIDRecorder,
    infer_labels_and_update_out_channels,
)
from fran.callback.debug_epoch_limit import DebugEpochBatchLimit
from fran.callback.incremental import LRFloorStop
from fran.callback.wandb.wandb import WandbImageGridCallback, WandbLogBestCkpt
from fran.configs.parser import normalize_logging_payload
from fran.managers.data.training import DataManagerDual, DataManagerPatch
from fran.managers.wandb.wandb import WandbManager
from fran.trainers.trainer import Trainer, _flatten_dict
from lightning.pytorch import Trainer as TrainerL
from lightning.pytorch.callbacks import (
    BatchSizeFinder,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.profilers import AdvancedProfiler
from utilz.cprint import cprint
from utilz.stringz import headline


def _train_metric_name(metric: str) -> str:
    if isinstance(metric, str) and metric.startswith("val"):
        return "train" + metric[3:]
    return metric


class CaseIDRecorderRT(CaseIDRecorder):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if (epoch > 0 and epoch % self.freq == 0) or self.incrementing == True:
            self.dfs["epoch"] = epoch
            if self.incrementing == False:
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


class DataManagerRT(DataManagerDual):
    def __init__(
        self,
        project_title,
        configs: dict,
        batch_size: int,
        manager_class,
        cache_rate=0.0,
        device="cuda",
        ds_type=None,
        data_folder: Optional[str | Path] = None,
        train_indices=None,
        debug=False,
        dual_ssd=True,
    ):
        self.manager_class = manager_class
        super().__init__(
            project_title,
            configs,
            batch_size,
            cache_rate,
            device,
            ds_type,
            True,
            data_folder,
            manager_class,
            None,
            train_indices,
            None,
            1.0,
            debug,
            dual_ssd,
        )

    def _build_managers(self):
        self.train_manager = self.manager_class(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="all",
            device=self.device,
            ds_type=self.ds_type,
            data_folder=self.data_folder,
            debug=self.debug,
            dual_ssd=self.dual_ssd,
        )

    def _iter_managers(self):
        return (self.train_manager,)

    def prepare_data(self):
        self._build_managers()
        self._call_prepare_data()
        if self.train_indices is not None:
            cprint(
                f"Limiting training dataset size to{self.train_indices}", color="yellow"
            )
            self.train_manager.select_cases_from_inds(self.train_indices)
            self.train_manager.data = self.train_manager.create_staged_data_dicts(
                self.train_manager.cases
            )

    def val_dataloader(self):
        self._validation_crash()

    def state_dict(self) -> dict:
        return {
            "batch_size": int(self._batch_size),
            "train_indices": self.train_indices,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not state_dict:
            return
        if "batch_size" in state_dict:
            self._batch_size = int(state_dict["batch_size"])
        if "train_indices" in state_dict:
            self.train_indices = state_dict["train_indices"]

    @property
    def valid_ds(self):
        self._validation_crash()

    @property
    def valid_manager(self):
        self._validation_crash()

    @DataManagerDual.batch_size.setter
    def batch_size(self, v: int) -> None:
        v = int(v)
        if v == self._batch_size:
            return
        self._batch_size = v
        if hasattr(self, "train_manager"):
            self.train_manager.batch_size = v
            self.train_manager.set_effective_batch_size()
            self.train_manager.create_train_dataloader()

    def _validation_crash(self):
        raise RuntimeError("DataManagerRT has no validation path")


class TrainerRT(Trainer):
    def monitor_metric_name(self, metric: str) -> str:
        return _train_metric_name(metric)

    def init_trainer(self, epochs):
        if "scheduler_monitor" in self.configs["model_params"]:
            self.configs["model_params"]["scheduler_monitor"] = (
                self.monitor_metric_name(
                    self.configs["model_params"]["scheduler_monitor"]
                )
            )
        return super().init_trainer(epochs)

    def load_trainer(self, map_location="cpu", **kwargs):
        N = super().load_trainer(map_location=map_location, **kwargs)
        N.monitor = self.monitor_metric_name(N.monitor)
        if "scheduler_monitor" in N.model_params:
            N.model_params["scheduler_monitor"] = self.monitor_metric_name(
                N.model_params["scheduler_monitor"]
            )
        return N

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
        early_stopping=False,
        early_stopping_monitor="val0_loss_dice",
        early_stopping_mode="min",
        early_stopping_patience=30,
        early_stopping_min_delta=0.0,
        lr_floor=None,
        wandb_grid_epoch_freq: int = 5,
        permanent_checkpoint_every_n_epochs: int = 100,
        dual_ssd: bool = False,
    ):
        if isinstance(train_indices, str):
            train_indices = train_indices.strip()
            if train_indices == "" or train_indices.lower() in {"none", "null"}:
                train_indices = None

        self.train_indices = train_indices
        self.debug = bool(debug)
        self.dual_ssd = bool(dual_ssd)
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
        self.D.prepare_data()
        self.D.setup(stage="fit")

        if self.ckpt is not None and batchsize_finder:
            headline(
                "Resumed run detected: disabling BatchSizeFinder to preserve checkpoint epoch/step state."
            )
            batchsize_finder = False

        cbs, logger, profiler = self.init_cbs(
            cbs=cbs,
            wandb=wandb,
            batchsize_finder=batchsize_finder,
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
            permanent_checkpoint_every_n_epochs=int(
                permanent_checkpoint_every_n_epochs
            ),
        )
        self._ensure_local_ckpt_on_wandb_resume(logger)

        self.trainer = TrainerL(
            callbacks=cbs,
            accelerator=accelerator,
            devices=trainer_devices,
            precision="bf16-mixed" if has_cuda else 32,
            profiler=profiler,
            logger=logger,
            max_epochs=epochs,
            check_val_every_n_epoch=1,
            limit_val_batches=0,
            log_every_n_steps=logging_freq,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            default_root_dir=self.project.checkpoints_parent_folder,
            strategy=strategy,
        )

    def init_dm(self):
        cache_rate = self.configs["dataset_params"]["cache_rate"]
        ds_type = self.configs["dataset_params"]["ds_type"]
        dm = DataManagerRT(
            project_title=self.project.project_title,
            configs=self.configs,
            batch_size=self.configs["dataset_params"]["batch_size"],
            manager_class=self.resolve_datamanager(self.configs["plan_train"]["mode"]),
            cache_rate=cache_rate,
            device=self.configs["dataset_params"].get("device", "cuda"),
            ds_type=ds_type,
            train_indices=self.train_indices,
            debug=self.debug,
            dual_ssd=self.dual_ssd,
        )
        labels_all = self.configs["plan_train"].get("labels_all")
        if not labels_all:
            infer_labels_and_update_out_channels(dm=dm, configs=self.configs)
        return dm

    def init_dm_unet(self, epochs, batch_size, override_dm_checkpoint=False):
        if self.ckpt:
            headline(
                "Run-through resume: loading model checkpoint and rebuilding datamodule from current config."
            )
            self.D = self.init_dm()
            self.N = self.load_trainer()
            self.configs["model_params"] = self.N.model_params
        else:
            self.D = self.init_dm()
            self.N = self.init_trainer(epochs)
        print("Data Manager initialized.\n {}".format(self.D))

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
        checkpoint_monitor = self.monitor_metric_name("val0_loss")
        early_stopping_monitor = self.monitor_metric_name(early_stopping_monitor)
        checkpoint_filename = f"{{epoch}}-{{{checkpoint_monitor}:.2f}}"

        cbs = [
            CaseIDRecorderRT(
                vip_label=self.configs["plan_train"].get("vip_label", 1), freq=2
            )
        ]
        if batchsize_finder == True:
            cbs += [
                BatchSizeFinder(batch_arg_name="batch_size", mode="binsearch"),
                BatchSizeSafetyMargin(),
            ]

        if self.debug == True:
            cbs += [DebugEpochBatchLimit(n=10)]

        cbs += [
            ModelCheckpoint(
                save_top_k=2,
                save_last=True,
                monitor=checkpoint_monitor,
                every_n_epochs=10,
                filename=checkpoint_filename,
                enable_version_counter=True,
                auto_insert_metric_name=True,
                save_on_train_epoch_end=True,
            ),
            ModelCheckpoint(
                save_top_k=-1,
                save_last=True,
                every_n_epochs=int(permanent_checkpoint_every_n_epochs),
                filename="epoch{epoch:04d}-snapshot",
                enable_version_counter=False,
                auto_insert_metric_name=False,
                save_on_train_epoch_end=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        if early_stopping:
            cbs += [
                EarlyStopping(
                    monitor=early_stopping_monitor,
                    mode=early_stopping_mode,
                    patience=int(early_stopping_patience),
                    min_delta=float(early_stopping_min_delta),
                    check_on_train_epoch_end=True,
                )
            ]

        if lr_floor is not None:
            cbs += [LRFloorStop(min_lr=lr_floor)]

        logger = None
        if wandb:
            logger = WandbManager(
                project=self.project,
                run_id=self.run_name,
                log_model_checkpoints=False,
                tags=tags,
                notes=description,
            )
            dm_cfg = {
                "dataset_params": normalize_logging_payload(
                    deepcopy(self.D.configs["dataset_params"])
                ),
                "plan_train": normalize_logging_payload(
                    deepcopy(self.D.configs["plan_train"])
                ),
            }
            if "plan_valid" in self.D.configs:
                dm_cfg["plan_valid"] = normalize_logging_payload(
                    deepcopy(self.D.configs["plan_valid"])
                )
            flat_cfg = _flatten_dict(dm_cfg, base="configs/datamodule")
            logger.experiment.config.update(flat_cfg, allow_val_change=True)
            if getattr(self.D, "train1_indices", None) is not None:
                logger.experiment.config.update(
                    {"configs/datamodule/train1_indices": list(self.D.train1_indices)},
                    allow_val_change=True,
                )
            cbs += [
                WandbImageGridCallback(
                    classes=self.configs["model_params"]["out_channels"],
                    patch_size=self.configs["plan_train"]["patch_size"],
                    epoch_freq=max(1, int(wandb_grid_epoch_freq)),
                ),
                WandbLogBestCkptRT(),
            ]

        if profiler:
            profiler = AdvancedProfiler(
                dirpath=self.project.log_folder, filename="profiler"
            )
            cbs += [DeviceStatsMonitor(cpu_stats=True)]
        else:
            profiler = None

        return cbs, logger, profiler

    def resolve_datamanager(self, mode: str):
        if mode == "sourcepbd":
            return DataManagerPatch
        return super().resolve_datamanager(mode)


# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
# %%
if __name__ == "__main__":
    from fran.configs.parser import ConfigMaker
    from fran.managers import Project
    from fran.managers.project import Project
    from fran.utils.common import *
    from utilz.helpers import pp

    P = Project("lidc")
    P = Project("totalseg")
    P = Project("kits23")
    C = ConfigMaker(P)
    C.setup(2)

    conf = C.configs
    print(conf["model_params"])

    planT = conf["plan_train"]
    pp(planT)

    planT["mode"]

    bs = 6
    device_id = 0
    batchsize_finder = True
    batchsize_finder = False
    wandb = False
    wandb = True
    override_dm = True
    override_dm = False

    run_name = "KITS23-SIRIG"
    run_name = None
    tags = []
    description = f""
    conf["dataset_params"]["fold"] = 0
    lr = None
    debug_ = False
    profiler = False
    compiled = False
    cbs = []
    wandb_grid_epoch_freq = 20
    train_indices = 40

# %%
    Tm = TrainerRT(P.project_title, conf, run_name)
    Tm.setup(
        compiled=compiled,
        train_indices=train_indices,
        cbs=cbs,
        debug=debug_,
        batch_size=bs,
        devices=[device_id],
        epochs=600 if profiler == False else 1,
        batchsize_finder=batchsize_finder,
        profiler=profiler,
        wandb=wandb,
        wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        tags=tags,
        description=description,
    )

# %%
    Tm.fit()
# %%
# %%
# SECTION:-------------------- TS--------------------------------------------------------------------------------------
    N = Tm.N
    D = Tm.D
    D.setup()
    D.prepare_data()
    tmt = D.train_manager

    tmt.collate_fn
    tmt.batch_size
    tmt.prepare_data()
    tmt.setup()
    dl2 = tmt.dl
    iteri2 = iter(dl2)

    for i, batch in enumerate(iteri2):
        print(batch["image"].shape)

    patch_overlap = 0
    mode = "constant"
    device = "cpu"
    sw_device = "cuda:1"
    bs = 1

# %%


