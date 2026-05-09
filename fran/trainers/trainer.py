import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from utilz.helpers import in_ipython
from fran.callback.base import BatchSizeSafetyMargin
from fran.callback.case_recorder import (
    CaseIDRecorder,
    infer_labels_and_update_out_channels,
)
from fran.callback.debug_epoch_limit import DebugEpochBatchLimit
from fran.callback.incremental import LRFloorStop
from fran.callback.wandb.wandb import WandbImageGridCallback, WandbLogBestCkpt
from fran.configs.helpers import normalize_logging_payload
from fran.managers import Project
from fran.managers.data.batch_tfms import DataManagerDualBTfms
from fran.managers.data.dualssd import (
    DataManagerDualSSD,
    DataManagerDualSSDBTfms,
    dual_ssd_manager_class,
)
from fran.managers.data.main import (
    DataManagerDual,
    DataManagerRBD,
    DataManagerLBD,
    DataManagerPatch,
    DataManagerSource,
    DataManagerWhole,
)
from fran.managers.unet import UNetManager
from fran.managers.wandb.wandb import WandbManager
from fran.trainers.base import (
    backup_ckpt,
    checkpoint_epoch,
    checkpoint_from_model_id,
    switch_ckpt_keys,
)
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


def _flatten_dict(d: dict, base: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{base}/{k}" if base else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


class FranBatchSizeFinder(BatchSizeFinder):
    """
    Use Lightning's stock batch-size finder, but restore the temporary
    probe checkpoint with weights_only=False so PyTorch 2.6+ can unpickle
    the trusted local temp checkpoint it just wrote.
    """

    def scale_batch_size(self, trainer, pl_module) -> None:
        restore = trainer._checkpoint_connector.restore

        def restore_trusted_temp_checkpoint(checkpoint_path=None, weights_only=None):
            return restore(checkpoint_path, weights_only=False)

        trainer._checkpoint_connector.restore = restore_trusted_temp_checkpoint
        try:
            super().scale_batch_size(trainer, pl_module)
        finally:
            trainer._checkpoint_connector.restore = restore


class Trainer:
    """Trainer variant with W&B logging/callback plumbing."""

    def __init__(
        self,
        project_title,
        configs,
        run_name=None,
        ckpt: Optional[str | Path] = None,
        run_through: bool = False,
    ):
        self.project = Project(project_title=project_title)
        self.configs = configs
        self.run_name = run_name
        self.run_through = bool(run_through)
        assert not (run_name and ckpt), "Cannot specify both run_name and ckpt"
        if ckpt is not None:
            # Future work: infer run_name from the ckpt folder layout plus project_title.
            raise NotImplementedError(
                "Explicit ckpt resume for Trainer is not implemented yet."
            )
        self.ckpt = None if run_name is None else checkpoint_from_model_id(run_name)
        self.qc_configs(configs, self.project)
        self.batch_tfms = False

    def monitor_metric_name(self, metric: str) -> str:
        """Map validation monitors onto training metrics for run-through fits."""
        if self.run_through and isinstance(metric, str) and metric.startswith("val"):
            return "train" + metric[3:]
        return metric

    def run_through_helpers(self):
        """Lazy imports for run-through support."""
        from fran.managers.data.run_through import DataManagerRT
        from fran.trainers.trainer_rt import CaseIDRecorderRT, WandbLogBestCkptRT

        return SimpleNamespace(
            DataManagerRT=DataManagerRT,
            CaseIDRecorderRT=CaseIDRecorderRT,
            WandbLogBestCkptRT=WandbLogBestCkptRT,
        )

    def apply_monitor_metric_name(self, model: UNetManager) -> None:
        """Align model and scheduler monitors with the configured training mode."""
        model.monitor = self.monitor_metric_name(model.monitor)
        if "scheduler_monitor" in model.model_params:
            model.model_params["scheduler_monitor"] = self.monitor_metric_name(
                model.model_params["scheduler_monitor"]
            )

    def available_checkpoint_epochs(self) -> list[tuple[int, Path]]:
        if self.ckpt is not None:
            checkpoints_dir = Path(self.ckpt).parent
        elif self.run_name is not None:
            checkpoints_dir = Path(
                checkpoint_from_model_id(self.run_name, normalize_keys=False)
            ).parent

        ckpts = []
        for ckpt in checkpoints_dir.glob("*.ckpt"):
            epoch = checkpoint_epoch(ckpt)
            ckpts.append((int(epoch), ckpt))
        ckpts = sorted(ckpts, key=lambda item: (item[0], item[1].name))
        return ckpts

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
        train_indices=None,
        val_indices=None,
        val_sampling: float = 1.0,
        logging_freq=25,
        lr=None,
        devices=1,
        compiled=None,
        wandb=True,
        profiler=False,
        debug: bool = False,
        val_every_n_epochs: int = 2,
        cbs=[],
        tags=[],
        description="",
        epochs=600,
        batchsize_finder=False,
        override_dm_checkpoint=False,
        early_stopping=True,
        early_stopping_monitor="val0_loss_dice",
        early_stopping_mode="min",
        early_stopping_patience=30,
        early_stopping_min_delta=0.0,
        lr_floor=None,
        wandb_grid_epoch_freq: int = 5,
        permanent_checkpoint_every_n_epochs: int = 100,
        dual_ssd: bool = False,
        batch_tfms: bool = False,
    ):
        effective_val_every_n_epochs = 1 if self.run_through else int(val_every_n_epochs)
        if self.run_through is False:
            assert (
                effective_val_every_n_epochs % 2 == 0
            ), "val_every_n_epochs must be an even integer"
        if isinstance(train_indices, str):
            train_indices = train_indices.strip()
            if train_indices == "" or train_indices.lower() in {"none", "null"}:
                train_indices = None
        self.val_every_n_epochs = effective_val_every_n_epochs
        if self.run_through:
            early_stopping_patience_epochs = int(early_stopping_patience)
        else:
            early_stopping_patience_epochs = int(
                early_stopping_patience / self.val_every_n_epochs
            )
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.val_sampling = float(val_sampling)
        self.debug = bool(debug)
        self.dual_ssd = bool(dual_ssd)
        self.batch_tfms = bool(batch_tfms)
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
        # infer_labels_and_update_out_channels(
        #     dm=self.D,
        #     configs=self.configs,
        #     pl_module=self.N,
        #     trainer=self,
        # )

        # Keep loop/step state consistent on resumed runs.
        # BatchSizeFinder runs probe fits and restores a temp checkpoint,
        # which resets progress counters (e.g., epoch shown as 1).
        if self.ckpt is not None and batchsize_finder:
            headline(
                "BatchSizeFinder was requested, but this run is resuming from an existing checkpoint "
                f"({self.ckpt}). To preserve resumed epoch/global_step continuity and avoid mixing "
                "BatchSizeFinder probe steps into the same W&B run history, batchsize_finder is being "
                "force-disabled for this run."
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
            early_stopping_patience=early_stopping_patience_epochs,
            early_stopping_min_delta=early_stopping_min_delta,
            lr_floor=lr_floor,
            wandb_grid_epoch_freq=int(wandb_grid_epoch_freq),
            permanent_checkpoint_every_n_epochs=int(
                permanent_checkpoint_every_n_epochs
            ),
        )
        self._ensure_local_ckpt_on_wandb_resume(logger)

        trainer_kwargs = dict(
            callbacks=cbs,
            accelerator=accelerator,
            devices=trainer_devices,
            precision="bf16-mixed" if has_cuda else 32,
            profiler=profiler,
            logger=logger,
            max_epochs=epochs,
            check_val_every_n_epoch=self.val_every_n_epochs,
            log_every_n_steps=logging_freq,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            default_root_dir=self.project.checkpoints_parent_folder,
            strategy=strategy,
        )
        if self.run_through:
            trainer_kwargs["limit_val_batches"] = 0
            trainer_kwargs["check_val_every_n_epoch"] = 1
        self.trainer = TrainerL(**trainer_kwargs)

    def init_dm(self):
        """Build the training datamodule for standard or run-through training."""
        cache_rate = self.configs["dataset_params"]["cache_rate"]
        ds_type = self.configs["dataset_params"]["ds_type"]
        self.configs["plan_train"]["val_every_n_epochs"] = self.val_every_n_epochs
        if self.run_through:
            rt = self.run_through_helpers()
            manager_class = self.resolve_datamanager(self.configs["plan_train"]["mode"])
            if self.dual_ssd:
                manager_class = dual_ssd_manager_class(manager_class)
            dm = rt.DataManagerRT(
                project_title=self.project.project_title,
                configs=self.configs,
                batch_size=self.configs["dataset_params"]["batch_size"],
                manager_class=manager_class,
                cache_rate=cache_rate,
                device=self.configs["dataset_params"].get("device", "cuda"),
                ds_type=ds_type,
                train_indices=self.train_indices,
                debug=self.debug,
            )
        else:
            manager_class_train = self.resolve_datamanager(
                self.configs["plan_train"]["mode"]
            )
            manager_class_valid = self.resolve_datamanager(
                self.configs["plan_valid"]["mode"]
            )
            dm_class = self.resolve_orchestrator_class()
            dm = dm_class(
                project_title=self.project.project_title,
                configs=self.configs,
                batch_size=self.configs["dataset_params"]["batch_size"],
                cache_rate=cache_rate,
                device=self.configs["dataset_params"].get("device", "cuda"),
                ds_type=ds_type,
                manager_class_train=manager_class_train,
                manager_class_valid=manager_class_valid,
                train_indices=self.train_indices,
                val_indices=self.val_indices,
                val_sampling=self.val_sampling,
                batch_tfms=self.batch_tfms,
            )

        labels_all = self.configs["plan_train"].get("labels_all")
        if not labels_all:
            infer_labels_and_update_out_channels(dm=dm, configs=self.configs)
        return dm

    def load_dm(self, batch_size=None, override_dm_checkpoint=False):
        sd = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        if override_dm_checkpoint:
            backup_ckpt(self.ckpt)
            sd["datamodule_hyper_parameters"]["configs"] = self.configs
            headline(
                "Overriding datamodule checkpoint. only Configs are overridden, not train / val_indices or other params"
            )
            out_fname = self.run_name + ".ckpt"
            bckup_ckpt = Path(self.project.log_folder) / out_fname
            shutil.copy(self.ckpt, bckup_ckpt)
            torch.save(sd, self.ckpt)

        hp = sd["datamodule_hyper_parameters"]
        batch_tfms = bool(hp["batch_tfms"]) if "batch_tfms" in hp else self.batch_tfms
        dm_class = self.resolve_orchestrator_class(batch_tfms=batch_tfms)
        D = dm_class.load_from_checkpoint(
            self.ckpt,
            project_title=self.project.project_title,
            batch_size=batch_size,
            val_sampling=self.val_sampling,
            map_location="cpu",
            weights_only=False,
        )
        if batch_size:
            D.configs["dataset_params"]["batch_size"] = int(batch_size)
        return D

    def init_dm_unet(self, epochs, batch_size, override_dm_checkpoint=False):
        """Initialize the datamodule/model pair for fresh or resumed training."""
        if self.ckpt:
            if self.run_through:
                if override_dm_checkpoint:
                    headline(
                        "Run-through resume ignores override_dm_checkpoint because the datamodule is rebuilt from current Trainer configs."
                    )
                headline(
                    "Run-through resume: loading model checkpoint and rebuilding datamodule from current config."
                )
                self.D = self.init_dm()
                self.N = self.load_trainer()
                self.configs["model_params"] = self.N.model_params
            else:
                self.D = self.load_dm(
                    batch_size=batch_size,
                    override_dm_checkpoint=override_dm_checkpoint,
                )
                headline(
                    "Loading configs from checkpoints. If you want to override them with Trainer configs, set override_dm_checkpoint=True.\nRegardless, train and val indices are fixed for this run in the ckpt."
                )
                self.configs["dataset_params"] = self.D.configs["dataset_params"]
                self.train_indices, self.val_indices = (
                    self.D.train_indices,
                    self.D.val_indices,
                )
                missing_keys = []
                for key in self.configs.keys():
                    try:
                        self.configs[key] = self.D.configs[key]
                    except KeyError:
                        missing_keys.append(key)
                if len(missing_keys) > 0:
                    headline(
                        f"Missing keys: {missing_keys}.. If any are critical, please check the datamodule checkpoint."
                    )
                self.N = self.load_trainer()
                self.configs["model_params"] = self.N.model_params
        else:
            self.D = self.init_dm()
            self.N = self.init_trainer(epochs)
        print("Data Manager initialized.\n {}".format(self.D))

    def resolve_orchestrator_class(self, batch_tfms: Optional[bool] = None):
        if batch_tfms is None:
            batch_tfms = self.batch_tfms
        if self.dual_ssd:
            return DataManagerDualSSDBTfms if batch_tfms else DataManagerDualSSD
        return DataManagerDualBTfms if batch_tfms else DataManagerDual

    def set_lr(self, lr):
        if self.ckpt:
            if lr is not None:
                cprint(
                    "Trainer resume ignores explicit lr={} and keeps checkpoint LR.".format(
                        lr
                    ),
                    color="yellow",
                )
            self.state_dict = torch.load(
                self.ckpt, weights_only=False, map_location="cpu"
            )
            self.lr = self.state_dict["lr_schedulers"][0]["_last_lr"][0]
        elif lr is not None:
            self.lr = float(lr)
        else:
            self.lr = self.configs["model_params"]["lr"]

    def init_cbs(
        self,
        cbs,
        wandb,
        batchsize_finder,
        profiler,
        tags,
        description="",
        early_stopping=True,
        early_stopping_monitor="val0_loss_dice",
        early_stopping_mode="min",
        early_stopping_patience=30,
        early_stopping_min_delta=0.0,
        lr_floor=None,
        wandb_grid_epoch_freq: int = 5,
        permanent_checkpoint_every_n_epochs: int = 100,
    ):
        """Build callbacks, logger, and profiler for the current training mode."""
        checkpoint_monitor = self.monitor_metric_name("val0_loss")
        early_stopping_monitor = self.monitor_metric_name(early_stopping_monitor)
        checkpoint_filename = f"{{epoch}}-{{{checkpoint_monitor}:.2f}}"
        case_id_recorder_cls = CaseIDRecorder
        wandb_best_ckpt_cls = WandbLogBestCkpt
        batch_size_finder_cls = FranBatchSizeFinder
        checkpoint_kwargs = {}
        early_stopping_kwargs = {}
        if self.run_through:
            rt = self.run_through_helpers()
            case_id_recorder_cls = rt.CaseIDRecorderRT
            wandb_best_ckpt_cls = rt.WandbLogBestCkptRT
            batch_size_finder_cls = BatchSizeFinder
            checkpoint_kwargs["save_on_train_epoch_end"] = True
            early_stopping_kwargs["check_on_train_epoch_end"] = True

        cbs = [
            case_id_recorder_cls(
                vip_label=self.configs["plan_train"].get("vip_label", 1), freq=20
            )
        ]
        if batchsize_finder == True:
            cbs += [
                batch_size_finder_cls(batch_arg_name="batch_size", mode="binsearch"),
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
                **checkpoint_kwargs,
            ),
            ModelCheckpoint(  # 2nd checkpointer
                save_top_k=-1,
                save_last=True,
                every_n_epochs=int(permanent_checkpoint_every_n_epochs),
                filename="epoch{epoch:04d}-snapshot",
                enable_version_counter=False,
                auto_insert_metric_name=False,
                **checkpoint_kwargs,
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
                    **early_stopping_kwargs,
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
                wandb_best_ckpt_cls(),
            ]

        if profiler:
            profiler = AdvancedProfiler(
                dirpath=self.project.log_folder, filename="profiler"
            )
            cbs += [DeviceStatsMonitor(cpu_stats=True)]
        else:
            profiler = None

        return cbs, logger, profiler

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
            sync_dist = False
        else:
            strategy = "ddp_notebook" if in_ipython() else "ddp"
            sync_dist = True

        self.devices = norm_devices
        self.sync_dist = sync_dist
        self.strategy = strategy

    def maybe_alter_configs(self, batch_size, compiled):
        if batch_size is not None:
            self.configs["dataset_params"]["batch_size"] = int(batch_size)
        if compiled is not None:
            self.configs["model_params"]["compiled"] = bool(compiled)

    def qc_configs(self, configs, project):
        # ratios = configs["plan_train"]["fgbg_ratio"]
        ratios = configs["dataset_params"]["fgbg_ratio"]
        assert isinstance(ratios, int | float | list), (
            "If no list is provided, fgbg_ratio must be an integer"
        )

    def get_callback(self, cb=CaseIDRecorder):
        trn = getattr(self, "trainer", None)
        if trn is None:
            print("Trainer is not initialized yet. No callbacks available.")
            return
        cbs = trn.callbacks
        cbb = [c for c in cbs if isinstance(c, cb)][0]
        if len(cbb) == 0:
            print (cbs)
            raise ValueError(f"Callback {cb} not found in trainer callbacks")
        return cbb


    def heuristic_batch_size(self):
        raise NotImplementedError

    def init_trainer(self, epochs):
        """Create a fresh UNetManager with monitor names normalized for this mode."""
        if "scheduler_monitor" in self.configs["model_params"]:
            self.configs["model_params"]["scheduler_monitor"] = (
                self.monitor_metric_name(
                    self.configs["model_params"]["scheduler_monitor"]
                )
            )

        N = UNetManager(
            project_title=self.project.project_title,
            configs=self.configs,
            lr=self.lr,
            sync_dist=self.sync_dist,
        )
        self.apply_monitor_metric_name(N)
        return N

    def load_trainer(self, map_location="cpu", **kwargs):
        """Load a UNetManager checkpoint and normalize its monitor names."""
        weights_only = False
        try:
            N = UNetManager.load_from_checkpoint(
                self.ckpt,
                map_location=map_location,
                strict=True,
                weights_only=weights_only,
                **kwargs,
            )
        except RuntimeError:
            switch_ckpt_keys(self.ckpt)
            N = UNetManager.load_from_checkpoint(
                self.ckpt,
                map_location=map_location,
                strict=True,
                weights_only=weights_only,
                **kwargs,
            )
        print("Model loaded from checkpoint: ", self.ckpt)
        self.apply_monitor_metric_name(N)
        return N

    def resolve_datamanager(self, mode: str):
        """Resolve the manager class for the configured training mode."""
        if mode == "pbd":
            DMClass = DataManagerPatch
        elif mode == "source":
            DMClass = DataManagerSource
        elif mode == "sourcepbd" and self.run_through:
            DMClass = DataManagerPatch
        elif mode == "whole":
            DMClass = DataManagerWhole
        elif mode == "lbd":
            DMClass = DataManagerLBD
        elif mode == "rbd":
            DMClass = DataManagerRBD
        else:
            raise NotImplementedError(
                "Mode {} is not supported for datamanager".format(mode)
            )
        return DMClass

    def fit(self):
        try:
            self.trainer.fit(
                model=self.N,
                datamodule=self.D,
                ckpt_path=self.ckpt,
                weights_only=False,
            )
        except KeyboardInterrupt:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass
            raise

    def best_available_checkpoint(self) -> Optional[Path]:
        model_ckpts = [
            cb for cb in self.trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        if not model_ckpts:
            return None
        best = model_ckpts[0].best_model_path
        if best:
            return Path(best)
        last = model_ckpts[0].last_model_path
        return Path(last) if last else None


# %%
# SECTION: -------------------- SETUP-------------------------------------------------------------------------------------- P = Project("nodes") <CR> <CR> <CR> <CR> <CR>
if __name__ == "__main__":
    from fran.configs.parser import ConfigMaker
    from fran.utils.common import *
    from utilz.helpers import pp

    P = Project("lidc")
    P = Project("kits23")
    P = Project("totalseg")
    C = ConfigMaker(P)
    C.setup(3)

    conf = C.configs
    print(conf["model_params"])

    planT = conf["plan_train"]
    planV = conf["plan_valid"]
    pp(planT)

    planT["mode"]
    # add_plan_to_db(plan,"/r/datasets/preprocessed/totalseg/lbd/spc_100_100_100_plan5",P.db)
    # if (lm==3).any():
    #     print("Bad values 3 ->0")
    #     lm[lm==3]=1
    #     torch.save(lm, bad_case_fn)
    #
    # find_matching_fn(Path(bad_names[0])[0],fixed, tags=["all"])

    # fldr= DS.kits23.folder
    # fn = fldr/("label_analysis/lesion_stats.csv")
    # df = pd.read_csv(fn)
    # counts = df.groupby("case_id").size()
    # counts2 = counts.sort_values(ascending=False)
    # bb= counts2.index[:200]
# SECTION:-------------------- TRAINING-------------------------------------------------------------------------------------- <CR> <CR> <CR> devices = 2 <CR> <CR> <CR> <CR> <CR> <CR>
# %%
    bs = 16
    device_id = 0
    batchsize_finder = True
    batchsize_finder = False
    batch_tfms=True
    wandb = False
    wandb = True
    override_dm = False
    override_dm = True

    # run_name = "KITS23-SIRIG"
    run_name = None
    run_name = "TOTALSEG-NJUGU"
    tags = []
    description = f""
    conf["dataset_params"]["fold"] = 0
    lr = None
    debug_ = False
    profiler = False
    compiled = False
    cbs = []
    wandb_grid_epoch_freq = 20
    val_every_n_epochs = 2
    train_indices = None
# %%
# SECTION:--------------------  TRAINING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    Tm = Trainer(P.project_title, conf, run_name)
# %%
    Tm.setup(
        compiled=compiled,
        train_indices=train_indices,
        val_every_n_epochs=val_every_n_epochs,
        val_sampling=1.0,
        batch_tfms=batch_tfms,
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
    # model(inputs)
# %%
    N = Tm.N
    D = Tm.D
    D.setup()
    D.prepare_data()
    tmt = D.train_manager
    tmv = D.valid_manager

# %%
    tmt.collate_fn
    tmv.collate_fn
    tmv.batch_size
    tmv.prepare_data()
# %%
    tmv.setup()
    dl = tmv.dl
    iteri = iter(dl)
# %%
    for i, batch in enumerate(iteri):
        print(batch["image"].shape)

# %%
    ds = tmv.ds
    ds[0]
    dici = ds[0]
    dici2 = ds[1]
# %%
    tmt.setup()
    dl2 = tmt.dl
    iteri2 = iter(dl2)
# %%
    for i, batch in enumerate(iteri2):
        print(batch["image"].shape)
# %%

    patch_overlap = 0
    mode = "constant"
    device = "cpu"
    sw_device = "cuda:1"
    bs = 1  # start lower if you are hitting OOM
# %%
# %%
