# %%
import shutil
from tqdm.auto import tqdm as pbar
from utilz.cprint import cprint
from utilz.helpers import info_from_filename, set_autoreload
from typing import Optional

import ipdb
from fastcore.all import in_ipython
from tqdm.auto import tqdm as pbar
from utilz.stringz import headline

from fran.callback.case_recorder import CaseIDRecorder
from fran.callback.base import BatchSizeSafetyMargin
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
                                            DataManagerPatchI,
                                            DataManagerSourceI,
                                            DataManagerWholeI, DataManagerWIDI)

torch._dynamo.config.suppress_errors = True

from lightning.pytorch.callbacks import BatchSizeFinder, ModelCheckpoint

try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass

import torch


class IncrementalTrainer (TrainerBK):
    def __init__(self, project_title, configs, run_name=None, ckpt_path: Optional[str | Path] = None, debug=False):
        self.debug=debug
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
        train1_indices:int|list = 50,
        data_increment_size:int=30,
        dice_loss_threshold: float = 0.5,
        logging_freq=25,
        lr=None,
        devices=1,
        compiled=None,
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
        wandb=True,
        wandb_grid_epoch_freq: int = 5,
        log_incremental_to_wandb: bool = True,
    ):
        self.train1_indices = train1_indices
        self._log_incremental_to_wandb = bool(log_incremental_to_wandb)
        self.data_increment_size = data_increment_size
        self.dice_loss_threshold = dice_loss_threshold
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
            wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        )

    def init_dm(self):
        cache_rate = self.configs["dataset_params"]["cache_rate"]
        ds_type = self.configs["dataset_params"]["ds_type"]

        
        return DataManagerDualI(
            self.project.project_title,
            configs=self.configs,
            batch_size=self.configs["dataset_params"]["batch_size"],
            cache_rate=cache_rate,
            device=self.configs["dataset_params"].get("device", "cuda"),
            ds_type=ds_type,
            train1_indices = self.train1_indices
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



        D = DataManagerDualI.load_from_checkpoint(
            self.ckpt,
            project_title=self.project.project_title,
            batch_size=batch_size,
            map_location="cpu",
        )

        if batch_size:
            D.configs["dataset_params"]["batch_size"] = int(batch_size)


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
        wandb,
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
            wandb=wandb,
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
            CaseIDRecorder(freq=5),
            UpdateDatasetOnPlateau(
                monitor = "train_loss_dice_epoch",
                log_to_wandb=bool(log_incremental_to_wandb), 
                debug=self.debug,
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

    def fit(self, max_restarts= 3):
            increment_size = self.data_increment_size
            unused_samples = len(self.D.train_df[self.D.train_df["used_in_training"]==False])
            while unused_samples>0:
                if not hasattr(self,"D"):
                    self.D = self.init_dm()
                self.trainer.fit(model=self.N, datamodule=self.D, ckpt_path=self.ckpt)
                # Only use finder/safety callbacks on the very first fit invocation.
                self.remove_cbs([BatchSizeFinder, BatchSizeSafetyMargin])
                # self.trainer.fit(model=self.N, datamodule=self.D)
                dm = self.D
                dlv = self.D.train2_dataloader()
                model = self.N
                self.trainer.validate(model=model, dataloaders= dlv)
                df = self.trainer.dfs["train2"]
                dice_loss  =df.groupby("case_id")["loss_dice"].median()
                dice_loss = dice_loss.sort_values(ascending=False)
                dice_loss = dice_loss[:increment_size]
                dice_loss = dice_loss[dice_loss>self.dice_loss_threshold]
                worst_case_ids = dice_loss.index
                if len(worst_case_ids) == 0:
                    cprint("No more samples to add", color="blue", bold=True)
                    dsc_fn = self.project.log_folder/("left_over.csv")
                    print("Saving left over case_ids to file: ", dsc_fn)
                    dice_loss.to_csv(dsc_fn)
                    break
                
                indices_new = dm.train_df[dm.train_df["case_id"].isin(worst_case_ids)].index
                dm.update_dataframe_indices(indices_new)
                # self.trainer.save_checkpoint(str(self.ckpt))
                unused_samples = len(self.D.train_df[self.D.train_df["used_in_training"]==False])
                used_indices = self.D.train_df[self.D.train_df["used_in_training"]==True].index
                self.train1_indices = list(used_indices)
                increment_size = min(self.data_increment_size, unused_samples)
                cprint(len(dm.train_df[dm.train_df["used_in_training"]==True]), color="blue", bold=True)
                del self.D
            if not hasattr(self,"D"):
                self.D = self.init_dm()
            cprint("Full compliment data in used. Removing UpdateDatasetOnPlateau callback", color="blue", bold=True)
            self.remove_cbs([UpdateDatasetOnPlateau])
            cprint("Training will now continue for all  epochs", color="blue", bold=True)

            self.trainer.fit(model=self.N, datamodule=self.D)


    def remove_cbs(self, cb_classes):
        if not isinstance(cb_classes, (list, tuple, set)):
            cb_classes = [cb_classes]
        cb_classes = tuple(cb_classes)
        matches = [cb for cb in self.trainer.callbacks if isinstance(cb, cb_classes)]
        if not matches:
            return
        self.trainer.callbacks = [
            cb for cb in self.trainer.callbacks if not isinstance(cb, cb_classes)
        ]

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

    device_id = 1
    bs = 4

    # run_name ='LITS-1285'
    compiled = False
    profiler = False
    # NOTE: if wandb = False, should store checkpoint locally
    batchsize_finder = True
    wandb = True
    override_dm = False
    tags = []
    description = f"Partially trained up to 100 epochs"

    conf['plan_train']

    cbs = [PeriodicTest(every_n_epochs=1,limit_batches=50)]

    conf["dataset_params"]["cache_rate"]=0.0
    print(conf['model_params']['out_channels'])
    

    conf['dataset_params']['cache_rate']

    conf["dataset_params"]["fold"]=0
    run_name="NODES-0087"
    run_name=None
    lr= 1e-2
    debug=True
    debug=False
# %%
#SECTION:-------------------- TRAIN--------------------------------------------------------------------------------------
    Tm = IncrementalTrainer    (P.project_title, conf, run_name, debug=debug)
    epochs =400
    train_init_indices= 30
# %%
    Tm.setup(
        train1_indices=train_init_indices,
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        data_increment_size=20,
        epochs=epochs,
        batchsize_finder=batchsize_finder,
        profiler=profiler,
        wandb=wandb,
        tags=tags,
        description=description,
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
    Tm.D.setup("fit")
    Tm.D.train_manager2.keys
    Tm.D.train_manager2.collate_fn
    dlv = Tm.D.valid_dataloader()
    dl = Tm.D.train_dataloader()
    iteri = iter(dl)
    b = next(iteri)
    ds = Tm.D.train_manager1.ds
    ds[0]
    tmt = Tm.D.train_manager1

    unused_samples = len(Tm.D.train_df[Tm.D.train_df["used_in_training"]==False])
# %%

    dm = trainer.datamodule
    dl = dm.train_manager.dl2
    cprint("Running a validation epoch on remaining training data", color="yellow", bold=True)
    trainer.validate(model=pl_module, dataloaders=dl)
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
    iteri = iter(dlt)
    # Tm.N.model.to('cpu')
# %%

    trainer = Tm.trainer
    cir = [cb for cb in trainer.callbacks if isinstance(cb, CaseIDRecorder)][0]
    cir.dfs['train'].to_csv("train1.csv")
    cir.dfs['valid'].to_csv("valid1.csv")

    cir.reset()
    
    fig = cir.create_plotly(cir.dfs['train'])
    fig.show()
# %%
    df = cir.dfs['train']
    df.columns

    df = cir.dfs["valid"]
# %%
    ld = cir.loss_dicts_train
    [a.keys() for a in ld]
# %%
    df2 = pd.DataFrame(ld)
    df2.to_csv("tmp.csv")

    mini_df= cir.create_limited_df(ld)

# %%

    import re
    dft = mini_df
    batch_vars  = [var for var in dft.columns if re.search(r"batch.*id", var)] 
# %%
    dfs = []
    num_batches = len(batch_vars)
    for n in range(num_batches):
        batch_var = "batch"+str(n)+"_"
        df1 = dft.loc[:,dft.columns.str.contains(batch_var)]
        df1.columns= df1.columns.str.replace(batch_var, "")
        print(df1.columns)
        dfs.append(df1)
    df_final = pd.concat(dfs, axis=0)
    df_final.dropna(inplace=True)
    print(df_final)
# %%
    df_final = cir.pivot_batch_cols(mini_df)
    df_final.to_csv("tmp3.csv")
# %%
    # Tm.D.prepare_data()
    # Tm.D.setup(stage="fit")
# %%
    dm = Tm.D
    n_samples = 10
    dlv = Tm.D.train2_dataloader()
    model = Tm.N
    trainer.validate(model=model, dataloaders= dlv)
    dl1 = Tm.D.train_manager.dl
    df = trainer.dfs["train2"]
    dice_loss  =df.groupby("case_id")["loss_dice"].median()
    dice_loss = dice_loss.sort_values(ascending=False)
    print(dice_loss)
    worst_case_ids = dice_loss[:n_samples].index
    indices_new = dm.train_df[dm.train_df["case_id"].isin(worst_case_ids)].index
    dm.update_dataframe_indices(indices_new)
    len(dm.train_df[dm.train_df["used_in_training"]==True])
    Tm.fit()
# %%
    cir.dfs['train2'].to_csv("train2.csv")
# %%
    for b in pbar(dl1):
        meta  = b["image"].meta
        fns = meta["filename_or_obj"]
        for fn in fns:
            fn_name = fn.split("/")[-1]
            cids1.append(info_from_filename(fn_name, full_caseid=True)["case_id"])
    cids1 = set(cids1)
# %%
    cids2=[]
    dl2 = Tm.D.train_manager.dl2
    for b in pbar(dl2):
        meta  = b["image"].meta
        fns = meta["filename_or_obj"]
        for fn in fns:
            fn_name = fn.split("/")[-1]
            cids2.append(info_from_filename(fn_name, full_caseid=True)["case_id"])
    cids2 = set(cids2)
# %%
    cids1 = set(cids1)
    cids2 = set(cir.dfs['train']['caseid'])
    cids3 = set(cir.dfs['valid']['caseid'])
    cids1 == cids2

    iteri = iter(dlv)
    cir.dfs["train2"]
    cids = set(cir.dfs['train2']['caseid'])
    cids2 = set(cir.dfs['train']['caseid'])
    cids3 = set(cir.dfs['valid']['caseid'])

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

    Tm.trainer.fit(model=Tm.N, datamodule=Tm.D, ckpt_path=Tm.ckpt)
    D = Tm.D
    dl = D.train_dataloader()
    batch = next(iter(dl))
    batch['image'].shape
# %%

    Tm.trainer.logger.save_dir
    Tm.trainer.save_checkpoint("tmp.ckpt")
# %%
    
    Tm.init_dm()

    Tm.D.hparams["train1_indices"] = Tm.D.train1_indices
    Tm.D.hparams.keys()
    Tm.D.hparams["train1_indices"]
# %%p""
# %%

    used_indices = Tm.D.train_df[Tm.D.train_df["used_in_training"]==True].index
# %%
