# %%
import shutil
from copy import deepcopy

import ipdb
from fastcore.all import in_ipython
from lightning.pytorch import Trainer as TrainerL
from lightning.pytorch.profilers import AdvancedProfiler
from tqdm.auto import tqdm as pbar
from utilz.string import headline

from fran.configs.parser import ConfigMaker, parse_neptune_dict
from fran.managers.data.training import DataManagerMulti
# from fran.callback.modelcheckpoint import ModelCheckpointUB
from fran.managers.project import Project
from fran.managers.unet import UNetManager
from fran.trainers.base import (backup_ckpt, checkpoint_from_model_id,
                                switch_ckpt_keys, write_normalized_ckpt)

tr = ipdb.set_trace

import os
from pathlib import Path

import psutil
import torch._dynamo

from fran.callback.nep import NeptuneImageGridCallback, NeptuneLogBestCkpt
from fran.managers.data.training import (DataManagerBaseline, DataManagerLBD,
                                         DataManagerPatch, DataManagerSource,
                                         DataManagerWhole, DataManagerWID)

torch._dynamo.config.suppress_errors = True
import warnings

from lightning.pytorch.callbacks import (DeviceStatsMonitor,
                                         LearningRateMonitor, ModelCheckpoint,
                                         TQDMProgressBar)

from fran.managers.nep import NeptuneManager

try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass

import torch


def safe_log_dict(exp, base_path: str, d: dict):
    """
    Recursively log a nested dict into Neptune experiment,
    key by key with try/except so one bad key doesn't stop the rest.
    """
    for k, v in d.items():
        path = f"{base_path}/{k}" if base_path else k
        try:
            if isinstance(v, dict):
                safe_log_dict(exp, path, v)
            else:
                exp[path].assign(v)
        except Exception as e:
            print(f"[Neptune logging skipped] {path}: {e}")


class Trainer:
    def __init__(self, project_title, configs, run_name=None):
        self.project = Project(project_title=project_title)
        self.configs = configs
        self.run_name = run_name
        self.ckpt = None if run_name is None else checkpoint_from_model_id(run_name)
        self.qc_configs(configs, self.project)

    def setup(
        self,
        batch_size=None,
        logging_freq=25,
        lr=None,
        devices=1,
        compiled=None,
        neptune=True,
        profiler=False,
        tags=[],
        description="",
        epochs=600,
        batchsize_finder=False,
        override_dm_checkpoint=False,
    ):
        """
        if override_dm_checkpoint=True, will use Trainer configs instead of DM checkpoint loaded configs
        """

        self.maybe_alter_configs(batch_size, batchsize_finder, compiled)
        self.set_lr(lr)
        self.set_strategy(devices)
        self.init_dm_unet(epochs, batch_size, override_dm_checkpoint)
        cbs, logger, profiler = self.init_cbs(neptune, profiler, tags, description)
        self.D.prepare_data()

        # if self.configs["model_params"]["compiled"] == True:
        #     self.N.model = torch.compile(self.N.model, dynamic=True)
        # self.N = torch.compile(self.N)

        self.trainer = TrainerL(
            callbacks=cbs,
            accelerator="gpu",
            devices=devices,
            precision="bf16-mixed",
            profiler=profiler,
            logger=logger,
            max_epochs=epochs,
            log_every_n_steps=logging_freq,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            default_root_dir=self.project.checkpoints_parent_folder,
            strategy=self.strategy,
            # strategy='ddp_find_unused_parameters_true'
        )

    def init_dm_unet(self, epochs, batch_size, override_dm_checkpoint=False):
        if self.ckpt:
            self.D = self.load_dm(
                batch_size=batch_size, override_dm_checkpoint=override_dm_checkpoint
            )
            # if override_dm_checkpoint == True:
            #     self.D.configs = self.configs
            #     self.D.save_hyperparameters(
            #         {
            #             "project_title": self.project.project_title,
            #             "configs": self.D.configs,
            #         },
            #         logger=False,
            #     )
            # else:
            headline(
                "Loading configs from checkpoints. If you want to override them with Trainer configs, set override_dm_checkpoint=True"
            )
            self.configs["dataset_params"] = self.D.configs["dataset_params"]
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

    def set_lr(self, lr):
        if lr and not self.ckpt:
            self.lr = lr
        elif lr and self.ckpt:
            self.lr = lr
            sd = torch.load(self.ckpt, map_location="cpu")

            for g in sd["optimizer_states"][0]["param_groups"]:
                g["lr"] = float(self.lr)

            sd["lr_schedulers"][0]["_last_lr"] = [float(self.lr)]

            headline(
                "Warning: Overriding CKPT learning rate with Trainer configs: {}".format(
                    self.lr
                )
            )
            torch.save(sd, self.ckpt)

        elif lr is None and self.ckpt:
            self.state_dict = torch.load(
                self.ckpt, weights_only=False, map_location="cpu"
            )
            self.lr = self.state_dict["lr_schedulers"][0]["_last_lr"][0]
        else:
            self.lr = self.configs["model_params"]["lr"]

    def init_cbs(self, neptune, profiler, tags, description):
        cbs = [
            ModelCheckpoint(
                save_last=True,
                monitor="val_loss",
                every_n_epochs=10,
                # mode="min",
                filename="{epoch}-{val_loss:.2f}",
                enable_version_counter=True,
                auto_insert_metric_name=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=3),
        ]
        if neptune == True:
            logger = NeptuneManager(
                project=self.project,
                run_id=self.run_name,
                log_model_checkpoints=False,  # Update to True to log model checkpoints
                tags=tags,
                description=description,
                capture_stdout=False,
                capture_stderr=False,
                capture_traceback=True,
                capture_hardware_metrics=True,
            )
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
            # Write to a clear namespace and also register as "hyperparams" so it’s prominent:
            safe_log_dict(logger.experiment, "configs/datamodule", dm_cfg)
            # logger.log_hyperparams(
            #     {"dm/plan_train/patch_size": dm_cfg["plan_train"]["patch_size"]}
            # )
            logger.experiment.wait()
            N = NeptuneImageGridCallback(
                classes=self.configs["model_params"]["out_channels"],
                patch_size=self.configs["plan_train"]["patch_size"],
                epoch_freq=5,  # skip how many epochs.
            )
            N2 = NeptuneLogBestCkpt()

            cbs += [N, N2]
        else:
            logger = None

        if profiler == True:
            profiler = AdvancedProfiler(
                dirpath=self.project.log_folder, filename="profiler"
            )
            # profiler ='simple'
            cbs += [DeviceStatsMonitor(cpu_stats=True)]
        else:
            profiler = None

        return cbs, logger, profiler

    def set_strategy(self, devices):
        """
        Normalize devices and pick a Lightning strategy.
        Returns (strategy_string_or_None, normalized_devices:int, sync_dist:bool)
        """
        # normalize devices
        if devices in (-1, "auto", None):
            n_gpus = torch.cuda.device_count()
            norm_devices = max(1, n_gpus)
        elif isinstance(devices, int):
            norm_devices = max(1, devices)
        elif isinstance(devices, (list, tuple)):
            norm_devices = max(1, len(devices))
        else:
            raise ValueError("devices must be int, list/tuple, -1, 'auto', or None")

        # detect notebook (safer than truthy get_ipython alone)

        if norm_devices <= 1:
            strategy = "auto"  # let PL pick single-process defaults
            sync_dist = False
        else:
            if in_ipython():
                # works in Jupyter; avoid fork-based DDP
                strategy = "ddp_notebook"
            else:
                # standard multi-GPU
                strategy = "ddp"
            sync_dist = True

        # store and return
        self.devices = norm_devices
        self.sync_dist = sync_dist
        self.strategy = strategy

    def maybe_alter_configs(self, batch_size, batchsize_finder, compiled):
        if batch_size is not None:
            self.configs["dataset_params"]["batch_size"] = int(batch_size)

        if batchsize_finder is True:
            self.configs["dataset_params"]["batch_size"] = self.heuristic_batch_size()

        if compiled is not None:
            self.configs["model_params"]["compiled"] = bool(compiled)

    def qc_configs(self, configs, project):
        ratios = configs["dataset_params"]["fgbg_ratio"]
        configs["model_params"]["out_channels"] - 1
        labels_all = configs["plan_train"]["labels_all"]
        if isinstance(ratios, list):
            assert (
                a := (len(ratios)) == (b := len(labels_all))
            ), "Class ratios {0} do not match number of labels in dataset {1}".format(
                a, b
            )
        else:
            assert isinstance(
                ratios, int
            ), "If no list is provided, fgbg_ratio must be an integer"

    def heuristic_batch_size(self):
        total_gb = psutil.virtual_memory().total / 1e9
        if total_gb < 15:
            return 6
        elif total_gb < 32:
            return 8
        elif total_gb < 48:
            return 20
        else:
            return 48

    def init_dm(self):

        cache_rate = self.configs["dataset_params"]["cache_rate"]
        ds_type = self.configs["dataset_params"]["ds_type"]
        D = DataManagerMulti(
            self.project.project_title,
            configs=self.configs,
            batch_size=self.configs["dataset_params"]["batch_size"],
            cache_rate=cache_rate,
            ds_type=ds_type,
        )

        return D

    def init_trainer(self, epochs):
        N = UNetManager(
            project_title=self.project.project_title,
            configs=self.configs,
            lr=self.lr,
            sync_dist=self.sync_dist,
        )
        return N

    def load_trainer(self, map_location="cpu", **kwargs):
        try:
            N = UNetManager.load_from_checkpoint(
                self.ckpt, map_location=map_location, strict=True, **kwargs
            )
            # N = UNetManager.load_from_checkpoint(
            #     self.ckpt,
            #     map_location="cpu",
            #     **kwargs,
            # )

        except RuntimeError:
            switch_ckpt_keys(self.ckpt)
            N = UNetManager.load_from_checkpoint(
                self.ckpt, map_location=map_location, strict=True, **kwargs
            )

        print("Model loaded from checkpoint: ", self.ckpt)
        return N

    def load_dm(self, batch_size=None, override_dm_checkpoint=False):
        if override_dm_checkpoint == True:
            sd = torch.load(self.ckpt, map_location="cpu")
            backup_ckpt(self.ckpt)
            sd["datamodule_hyper_parameters"]["configs"] = self.configs
            headline("Overriding datamodule checkpoint.")
            out_fname = self.run_name + ".ckpt"
            bckup_ckpt = Path(self.project.log_folder) / (out_fname)
            print("Datamodule checkpoint has been overwritten with Trainer configs.")
            print("A copy of original ckpt is stored at: ", bckup_ckpt)
            shutil.copy(self.ckpt, bckup_ckpt)
            torch.save(sd, self.ckpt)
        D = DataManagerMulti.load_from_checkpoint(
            self.ckpt,
            project_title=self.project.project_title,
            batch_size=batch_size,
            map_location="cpu",
        )
        if batch_size:
            # project_title = self.project.project_title
            D.configs["dataset_params"]["batch_size"] = batch_size
            # D.save_hyperparameters('project_title', 'configs',logger=False) # logger = False otherwise it clashes with UNet Manager
        return D

    def resolve_datamanager(self, mode: str):
        if mode == "patch":
            DMClass = DataManagerPatch
        elif mode == "source":
            DMClass = DataManagerSource
        elif mode == "whole":
            DMClass = DataManagerWhole
        elif mode == "lbd":
            DMClass = DataManagerLBD
        elif mode == "pbd":
            DMClass = DataManagerWID
        elif mode == "baseline":
            DMClass = DataManagerBaseline
        else:
            raise NotImplementedError(
                "Mode {} is not supported for datamanager".format(mode)
            )
        return DMClass

    def fit(self):
        self.trainer.fit(model=self.N, datamodule=self.D, ckpt_path=self.ckpt)


# %%


if __name__ == "__main__":
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- - <CR>
# %%

    # CODE: Project or configs should be the only arg not both
    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")

    proj_nodes = Project(project_title="nodes")
    proj_tsl = Project(project_title="totalseg")
    proj_litsmc = Project(project_title="litsmc")
    conf_litsmc = ConfigMaker(
        proj_litsmc,
    ).configs
    conf_nodes = ConfigMaker(
        proj_nodes,
    ).configs
    conf_tsl = ConfigMaker(
        proj_tsl,
    ).configs

    # conf['model_params']['lr']=1e-3
    conf_litsmc["dataset_params"]["cache_rate"]
    # run_name = "LITS-1007"
    device_id = 0
    run_tsl = "LITS-1120"
    run_nodes = "LITS-1110"
    run_none = None
    run_litsmc = "LITS-1131"
    bs = 10  # is good if LBD with 2 samples per case
    # run_name ='LITS-1003'
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    tags = []
    description = f"Partially trained up to 100 epochs"
# %%
# SECTION:-------------------- TOTALSEG TRAINING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    run_name = run_tsl

    run_name = run_none
    conf = conf_tsl
    proj = "totalseg"
# %%
    Tm = Trainer(proj, conf, run_name)
# %%
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=600 if profiler == False else 1,
        batchsize_finder=batch_finder,
        profiler=profiler,
        neptune=neptune,
        tags=tags,
        description=description,
    )
# %%
    # Tm.D.batch_size=8
    Tm.N.compiled = compiled
# %%
    Tm.fit()
    # model(inputs)
# %%

    conf["dataset_params"]["ds_type"]
    conf["dataset_params"]["cache_rate"]
# %%
# SECTION:-------------------- LITSMC -------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>

    run_name = run_litsmc
    run_name = run_none
    conf = conf_litsmc
    proj = "litsmc"
    conf["dataset_params"]["cache_rate"] = 0.5
# %%
    Tm = Trainer(proj, conf, run_name)
# %%
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=600 if profiler == False else 1,
        batchsize_finder=batch_finder,
        profiler=profiler,
        neptune=neptune,
        tags=tags,
        description=description,
    )
# %%
    # Tm.D.batch_size=8
    Tm.N.compiled = compiled
# %%
    Tm.fit()
    # model(inputs)
# %%
# SECTION:-------------------- NODES-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    run_name = run_nodes
    run_name = None
    conf = conf_nodes
    proj = "nodes"

# %%
    Tm = Trainer(proj, conf, run_name)
# %%
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=600 if profiler == False else 1,
        batchsize_finder=batch_finder,
        profiler=profiler,
        neptune=neptune,
        tags=tags,
        description=description,
    )
# %%
    # Tm.D.batch_size=8
    Tm.N.compiled = compiled
    Tm.fit()
# %%

# SECTION:-------------------- TROUBLESHOOTING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>

    Tm.D.prepare_data()
    Tm.D.setup()
    Tm.D.train_manager.keys_tr
    dl = Tm.D.train_dataloader()
    dlv = Tm.D.valid_dataloader()
    iteri = iter(dl)
    b = next(iteri)
# %%

    D = Tm.D
    dlt = D.train_dataloader()
    dlv = D.valid_dataloader()
    ds = Tm.D.valid_ds
    ds = Tm.D.train_ds
    dat = ds[0]
# %%

    cache_rate = 0
    ds_type = Tm.configs["dataset_params"]["ds_type"]
    ds_type = "cache"
    D = DataManagerMulti(
        Tm.project,
        configs=Tm.configs,
        batch_size=Tm.configs["dataset_params"]["batch_size"],
        cache_rate=cache_rate,
        ds_type=ds_type,
    )
    D.prepare_data()
    D.setup()

# %%

    for i, bb in pbar(enumerate(ds)):
        lm = bb[0]["lm"]
        print(lm.meta["filename_or_obj"])
# %%
    ds = Tm.D.train_ds
    dici = ds.data[0]
    dat = ds[0]
# %%
    tm = Tm.D.train_manager

    tm.tfms_list
# %%

    dici = tm.tfms_list[0](dici)
    dici = tm.tfms_list[1](dici)
    dici = tm.tfms_list[2](dici)
    dici = tm.tfms_list[3](dici)
    tm.tfms_list[3]
    tm.tfms_list[4]
    dici = tm.tfms_list[4](dici)

# %%
    dl = Tm.D.train_dataloader()
    dlv = Tm.D.valid_dataloader()
    iteri = iter(dlt)
    # Tm.N.model.to('cpu')
# %%
    while iter:
        batch = next(iteri)
        print(batch["image"].dtype)
# %%
# %%
