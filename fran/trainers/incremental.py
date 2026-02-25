# %%
import shutil
from utilz.helpers import set_autoreload
from typing import Optional

import ipdb
from fastcore.all import in_ipython
from tqdm.auto import tqdm as pbar
from utilz.stringz import headline

from fran.callback.case_recorder import CaseIDRecorder
from fran.callback.incremental import UpdateDatasetOnPlateau
from fran.callback.test import PeriodicTest
from fran.configs.parser import ConfigMaker
# from fran.callback.modelcheckpoint import ModelCheckpointUB
from fran.managers.project import Project
from fran.managers.unet import UNetManager
from fran.trainers.base import (backup_ckpt, checkpoint_from_model_id,
                                switch_ckpt_keys)
from fran.trainers.trainer_bk import TrainerBK

tr = ipdb.set_trace

import os
from pathlib import Path

import torch._dynamo

from fran.managers.data.incremental import (DataManagerBaselineI,
                                            DataManagerDualI, DataManagerLBDI,
                                            DataManagerMultiI,
                                            DataManagerPatchI,
                                            DataManagerSourceI,
                                            DataManagerWholeI, DataManagerWIDI)

torch._dynamo.config.suppress_errors = True

from lightning.pytorch.callbacks import ModelCheckpoint

try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass

import torch


def _dm_class_for_periodic_test(periodic_test: int):
    return DataManagerMultiI if int(periodic_test) > 0 else DataManagerDualI


def _dm_class_from_ckpt(ckpt_path: str | Path):
    """
    Decide which DM class the checkpoint expects, using datamodule hyperparams.
    Minimises crash risk when periodic_test differs from the run that created the ckpt.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    hp = sd.get("datamodule_hyper_parameters", {}) or sd.get("hyper_parameters", {})
    # your DMs store keys_test only on Multi
    return DataManagerMultiI if "keys_test" in hp else DataManagerDualI


class IncrementalTrainer (TrainerBK):
    def __init__(self, project_title, configs, run_name=None, ckpt_path: Optional[str | Path] = None):
        self.project = Project(project_title=project_title)
        self.configs = configs
        self.run_name = run_name
        if ckpt_path is not None:
            self.ckpt = Path(ckpt_path)
        else:
            self.ckpt = None if run_name is None else checkpoint_from_model_id(run_name)
        self.qc_configs(configs, self.project)

        self.periodic_test = 0  # default

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
        start_n: int = 40,
        wandb_grid_epoch_freq: int = 5,
        log_incremental_to_wandb: bool = True,
    ):
        self.start_n = start_n
        self._log_incremental_to_wandb = bool(log_incremental_to_wandb)
        super().setup(
            batch_size=batch_size,
            logging_freq=logging_freq,
            lr=lr,
            devices=devices,
            compiled=compiled,
            wandb=wandb,
            profiler=profiler,
            periodic_test=periodic_test,
            cbs=cbs,
            tags=tags,
            description=description,
            epochs=epochs,
            batchsize_finder=batchsize_finder,
            override_dm_checkpoint=override_dm_checkpoint,
            early_stopping=early_stopping,
            early_stopping_monitor=early_stopping_monitor,
            early_stopping_mode=early_stopping_mode,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            lr_floor=lr_floor,
            wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        )

    def init_dm(self):
        cache_rate = self.configs["dataset_params"]["cache_rate"]
        ds_type = self.configs["dataset_params"]["ds_type"]

        DM = _dm_class_for_periodic_test(self.periodic_test)
        return DM(
            self.project.project_title,
            configs=self.configs,
            batch_size=self.configs["dataset_params"]["batch_size"],
            cache_rate=cache_rate,
            device=self.configs["dataset_params"].get("device", "cuda"),
            ds_type=ds_type,
            start_n = self.start_n
        )

    def load_dm(self, batch_size=None, override_dm_checkpoint=False):
        if override_dm_checkpoint:
            sd = torch.load(self.ckpt, map_location="cpu")
            backup_ckpt(self.ckpt)
            sd["datamodule_hyper_parameters"]["configs"] = self.configs
            headline("Overriding datamodule checkpoint.")
            out_fname = self.run_name + ".ckpt"
            bckup_ckpt = Path(self.project.log_folder) / out_fname
            shutil.copy(self.ckpt, bckup_ckpt)
            torch.save(sd, self.ckpt)

        # Prefer the class the checkpoint was created with.
        DM_from_ckpt = _dm_class_from_ckpt(self.ckpt)
        DM_wanted = _dm_class_for_periodic_test(self.periodic_test)

        # If they disagree, do NOT force DM_wanted; load what the ckpt expects.
        # That avoids crashes from missing stored hyperparams / attributes.
        DM = DM_from_ckpt

        D = DM.load_from_checkpoint(
            self.ckpt,
            project_title=self.project.project_title,
            batch_size=batch_size,
            map_location="cpu",
        )

        if batch_size:
            D.configs["dataset_params"]["batch_size"] = int(batch_size)

        # Optional: warn (but don’t crash) if user asked for periodic testing but ckpt is Dual
        if (DM_from_ckpt is DataManagerDualI) and (DM_wanted is DataManagerMultiI):
            headline(
                "Note: checkpoint datamodule is Dual (no test). periodic_test>0 was requested, "
                "but DM is loaded from checkpoint to avoid incompatibility."
            )

        return D



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
        log_incremental_to_wandb: bool | None = None,
    ):
        cbs, logger, profiler = super().init_cbs(
            cbs=cbs,
            neptune=neptune,
            batchsize_finder=batchsize_finder,
            periodic_test=periodic_test,
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
        )
        if log_incremental_to_wandb is None:
            log_incremental_to_wandb = getattr(self, "_log_incremental_to_wandb", False)

        cbs += [
            CaseIDRecorder(freq=10),
            UpdateDatasetOnPlateau(
                n_samples_to_add=30,
                log_to_wandb=bool(log_incremental_to_wandb),
            ),
            # UpdateDatasetOnEMAMomentum(
            #     n_samples_to_add=10,
            #     log_to_wandb=bool(log_incremental_to_wandb),
            # ),
        ]
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

    def maybe_alter_configs(self, batch_size, compiled):
        if batch_size is not None:
            self.configs["dataset_params"]["batch_size"] = int(batch_size)

        # if batchsize_finder is True:
        #     self.configs["dataset_params"]["batch_size"] = self.heuristic_batch_size()

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
        raise NotImplementedError



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

  
    def resolve_datamanager(self, mode: str):
        if mode == "patch":
            DMClass = DataManagerPatchI
        elif mode == "source":
            DMClass = DataManagerSourceI
        elif mode == "sourcepatch":
            DMClass = DataManagerPatchI
        elif mode == "whole":
            DMClass = DataManagerWholeI
        elif mode == "lbd":
            DMClass = DataManagerLBDI
        elif mode == "pbd":
            DMClass = DataManagerWIDI
        elif mode == "baseline":
            DMClass = DataManagerBaselineI
        else:
            raise NotImplementedError(
                "Mode {} is not supported for datamanager".format(mode)
            )
        return DMClass

    def fit(self):
        self.trainer.fit(model=self.N, datamodule=self.D, ckpt_path=self.ckpt)

    def best_available_checkpoint(self) -> Optional[Path]:
        """
        Prefer best checkpoint; fallback to last checkpoint.
        """
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

if __name__ == "__main__":
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- - <CR> <CR>
    set_autoreload()
    from fran.utils.common import *
    P = Project("nodes")
    # P.add_data([DS.totalseg])
    C = ConfigMaker(P )
    C.setup(4)
    C.plans
    conf = C.configs
    print(conf["model_params"])

    planT = conf['plan_train']
    planV = conf["plan_valid"]
    pp(planT)

    print(planT['mode'])
    # add_plan_to_db(plan,"/r/datasets/preprocessed/totalseg/lbd/spc_100_100_100_plan5",P.db)
    # if (lm==3).any():
    #     print("Bad values 3 ->0")
    #     lm[lm==3]=1
    #     torch.save(lm, bad_case_fn)
    #
    # find_matching_fn(Path(bad_names[0])[0],fixed, tags=["all"])
# %%
# SECTION:-------------------- COnfirm plans exist--------------------------------------------------------------------------------------

    devices= [1]
    bs = 4

    # run_name ='LITS-1285'
    compiled = False
    profiler = False
    # NOTE: if wandb = False, should store checkpoint locally
    batch_finder = False
    wandb = False
    override_dm = False
    tags = []
    description = f"Partially trained up to 100 epochs"

    conf['plan_train']

    cbs = [PeriodicTest(every_n_epochs=1,limit_batches=50)]

    conf["dataset_params"]["cache_rate"]=0.0
    print(conf['model_params']['out_channels'])
    

    conf['dataset_params']['cache_rate']

# %
    conf["dataset_params"]["fold"]=0
    run_name=None
    lr= 1e-2
#SECTION:-------------------- TRAIN--------------------------------------------------------------------------------------
    Tm = IncrementalTrainer    (P.project_title, conf, run_name)
    device_id = 0
    batchsize_finder=False
    wandb=False
    epochs =50
# %%
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=epochs,
        batchsize_finder=batchsize_finder,
        profiler=profiler,
        wandb=wandb,
        tags=tags,
        description=description,
        start_n=10
    )
# %%
    # Tm.D.batch_size=8
    # model(inputs)
    Tm.fit()
# %%

    conf["dataset_params"]["ds_type"]
    conf["dataset_params"]["cache_rate"]
# %%
# SECTION:-------------------- LITSMC -------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>

# SECTION:-------------------- TROUBLESHOOTING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>

    Tm.D.prepare_data()
    Tm.D.setup()
    Tm.D.train_manager.keys_tr
    dlv = Tm.D.valid_dataloader()
    dl = Tm.D.train_dataloader()
    iteri = iter(dl)
    b = next(iteri)
# %%
    N = Tm.N
    aa = N._common_step(b,0)


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
    D = DataManagerMultiI(
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
    trainer = Tm.trainer
    model = Tm.N
    dlv = Tm.D.train_manager.dl2
    trainer.validate(model=model, dataloaders= dlv)
    trainer.test(model=model, dataloaders= dlv)

    model.loss_dict_full
    cir = [cb for cb in trainer.callbacks if isinstance(cb, CaseIDRecorder)][0]

# %%
    iteri = iter(dlv)
    
    batch = next(iteri)
    batch['image'].meta['filename_or_obj']
# %%
    tmg  = Tm.D.train_manager
    tmg.data_df.columns
    tmg.setup(stage="fit")
    df = tmg.data_df
    df.loc[df["used_in_training"]==True, "image"]
    tmg.create_incremental_dataloaders()
    len(tmg.ds[0])
# %%
