# %%
import shutil

from monai.transforms.io.dictionary import LoadImaged
from torchinfo import summary
from fran.transforms.imageio import TorchReader
from fran.transforms.misc_transforms import LoadDict
from fran.utils.common import common_vars_filename
import ipdb
tr = ipdb.set_trace

from label_analysis.overlap import get_ipython
import numpy as np
from typing import Union
from pathlib import Path
from fastcore.basics import store_attr
from label_analysis.merge import load_dict
from monai.transforms.croppad.dictionary import RandCropByPosNegLabeld, ResizeWithPadOrCropd
from monai.transforms.utility.dictionary import EnsureChannelFirstd

import psutil
import random
import torch._dynamo
from fran.callback.nep import NeptuneImageGridCallback

from fran.evaluation.losses import CombinedLoss, DeepSupervisionLoss
from fran.managers.data import ( DataManagerLBD, DataManagerPatch,
                                 DataManagerSource)
from fran.utils.fileio import load_yaml
from fran.utils.imageviewers import ImageMaskViewer

torch._dynamo.config.suppress_errors = True
from fran.managers.nep import NeptuneManager
import itertools as il
import operator
import warnings
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ( LearningRateMonitor,
                                         TQDMProgressBar, DeviceStatsMonitor)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fran.architectures.create_network import (create_model_from_conf, 
                                               pool_op_kernels_nnunet)
import torch.nn.functional as F
from fran.transforms.spatialtransforms import one_hot
try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass

import torch
from lightning.pytorch import LightningModule, Trainer





def fix_dict_keys(input_dict, old_string, new_string):
    output_dict = {}
    for key in input_dict.keys():
        neo_key = key.replace(old_string, new_string)
        output_dict[neo_key] = input_dict[key]
    return output_dict


def checkpoint_from_model_id(model_id):
    common_paths = load_yaml(common_vars_filename)
    fldr = Path(common_paths["checkpoints_parent_folder"])
    all_fldrs = [
        f for f in fldr.rglob("*{}/checkpoints".format(model_id)) if f.is_dir()
    ]
    if len(all_fldrs) == 1:
        fldr = all_fldrs[0]
    else:
        print(
            "no local files. Model may be on remote path. use download_neptune_checkpoint() "
        )
        tr()

    list_of_files = list(fldr.glob("*"))
    ckpt = max(list_of_files, key=lambda p: p.stat().st_ctime)
    return ckpt


# from fran.managers.base import *


# class NeptuneCallback(Callback):
# def on_train_epoch_start(self, trainer, pl_module):
#     trainer.logger.experiment["training/epoch"] = trainer.current_epoch

class UNetTrainer(LightningModule):
    def __init__(
        self,
        project,
        dataset_params,
        model_params,
        loss_params,
        max_epochs=1000,
        lr=None,
        sync_dist=False,
    ):
        super().__init__()
        self.lr = lr if lr else model_params["lr"]
        store_attr()
        self.save_hyperparameters("model_params", "loss_params")
        self.model, self.loss_fnc = self.create_model()

    def _common_step(self, batch, batch_idx):
        if not hasattr(self, "batch_size"):
            self.batch_size = batch["image"].shape[0]
        inputs, target = batch["image"], batch["lm"]
        pred = self.forward(
            inputs
        )  # self.pred so that NeptuneImageGridCallback can use it

        pred, target = self.maybe_apply_ds_scales(pred, target)
        loss = self.loss_fnc(pred, target)
        loss_dict = self.loss_fnc.loss_dict
        self.maybe_store_preds(pred)
        return loss, loss_dict

    def maybe_store_preds(self, pred):
        if hasattr(self.trainer, "store_preds") and self.trainer.store_preds == True:
            if isinstance(pred, Union[tuple, list]):
                self.pred = [p.detach().cpu() for p in pred]
            else:
                self.pred = pred.detach().cpu()

    def maybe_apply_ds_scales(self, pred, target):
        if isinstance(pred, list) and isinstance(target, torch.Tensor):
            target_listed = []
            for s in self.deep_supervision_scales:
                if all([i == 1 for i in s]):
                    target_listed.append(target)
                else:
                    size = [
                        int(np.round(ss * aa)) for ss, aa in zip(s, target.shape[2:])
                    ]
                    target_downsampled = F.interpolate(
                        target, size=size, mode="nearest"
                    )
                    target_listed.append(target_downsampled)
            target = target_listed
        return pred, target

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self._common_step(batch, batch_idx)
        self.log_losses(loss_dict, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self._common_step(batch, batch_idx)
        self.log_losses(loss_dict, prefix="val")
        return loss

    def log_losses(self, loss_dict, prefix):
        metrics = [
            "loss",
            "loss_ce",
            "loss_dice",
            "loss_dice_label1",
            "loss_dice_label2",
        ]
        metrics = [me for me in metrics if me in loss_dict.keys()]
        renamed = [prefix + "_" + nm for nm in metrics]
        logger_dict = {
            neo_key: loss_dict[key] for neo_key, key in zip(renamed, metrics)
        }
        self.log_dict(
            logger_dict,
            logger=True,
            batch_size=self.batch_size,
            sync_dist=self.sync_dist,
        )
        # self.log(prefix + "_" + "loss_dice", loss_dict["loss_dice"], logger=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=30)
        output = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_dice",
                "frequency": 2
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        return output

    def forward(self, inputs):
        return self.model(inputs)

        model = create_model_from_conf(self.model_params, self.dataset_params)
    def create_model(self):
        if (
            self.model_params["arch"] == "DynUNet"
            or self.model_params["arch"] == "DynUNet_UB"
            or self.model_params["arch"] == "nnUNet"
        ):
            if (
                self.model_params["arch"] == "DynUNet"
                or self.model_params["arch"] == "DynUNet_UB"
            ):
                num_pool = 4  # this is a hack i am not sure if that's the number of pools . this is just to equalize len(mask) and len(pred)
                ds_factors = list(
                    il.accumulate(
                        [1]
                        + [
                            2,
                        ]
                        * (num_pool - 1),
                        operator.truediv,
                    )
                )
                ds = [1, 1, 1]
                self.deep_supervision_scales = list(
                    map(
                        lambda list1, y: [x * y for x in list1],
                        [
                            ds,
                        ]
                        * num_pool,
                        ds_factors,
                    )
                )

            else:
                num_pool = 5

                self.net_num_pool_op_kernel_sizes = pool_op_kernels_nnunet(
                    self.dataset_params["patch_size"]
                )
                self.deep_supervision_scales = [[1, 1, 1]] + list(
                    list(i)
                    for i in 1
                    / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0)
                )[:-1]

            loss_func = DeepSupervisionLoss(
                levels=num_pool,
                deep_supervision_scales=self.deep_supervision_scales,
                fg_classes=self.model_params["out_channels"] - 1,
            )

        else:
            loss_func = CombinedLoss(
                **self.loss_params, fg_classes=self.model_params["out_channels"] - 1
            )
        return model, loss_func

def update_nep_run_from_config(nep_run, config):
    for key, value in config.items():
        nep_run[key] = value
    return nep_run


def maybe_ddp(devices):
    if devices == 1 or isinstance(devices, Union[list, str, tuple]):
        return "auto"
    ip = get_ipython()
    if ip:
        print("Using interactive-shell ddp strategy")
        return "ddp_notebook"
    else:
        print("Using non-interactive shell ddp strategy")
        return "ddp"


class TrainingManager:
    def __init__(self, project, configs, run_name=None):
        store_attr()
        self.ckpt = None if run_name is None else checkpoint_from_model_id(run_name)
        self.qc_configs(configs,project)

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
        cache_rate = 0.0
    ):
        self.maybe_alter_configs(batch_size,batchsize_finder,compiled)
        self.set_lr(lr)
        self.set_strategy(devices)
        self.init_dm_unet(epochs)
        cbs, logger, profiler = self.init_cbs(neptune,profiler,tags, description)
        self.D.prepare_data()

        if self.configs["model_params"]["compiled"] == True:
            self.N = torch.compile(self.N)

        self.trainer = Trainer(
            callbacks=cbs,
            accelerator="gpu",
            devices=devices,
            # precision="16-mixed",
            precision="bf16",
            profiler = profiler,
            logger=logger,
            max_epochs=epochs,
            log_every_n_steps=logging_freq,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            default_root_dir=self.project.checkpoints_parent_folder,
            strategy=self.strategy
            # strategy='ddp_find_unused_parameters_true'
        )

    def init_dm_unet(self,epochs):
        if self.ckpt:
            self.D = self.load_dm()
            self.configs['dataset_params'] = self.D.dataset_params
            self.N = self.load_trainer()
        
        else:
            self.D = self.init_dm(cache_rate)
            self.N = self.init_trainer( epochs )


    def set_lr(self,lr):
        if lr and not self.ckpt:
            self.lr  = lr
        elif lr and self.ckpt:
            self.state_dict = torch.load(self.ckpt)
            self.state_dict["lr_schedulers"][0]["_last_lr"][0] = lr
            torch.save(self.state_dict, self.ckpt)

        elif lr is None and self.ckpt:
            self.state_dict = torch.load(self.ckpt)
            self.lr = self.state_dict["lr_schedulers"][0]["_last_lr"][0]
        else:
            self.lr  = self.configs["model_params"]["lr"]
    

    def init_cbs(self,neptune,profiler,tags,description):
        cbs = [
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
                capture_stdout=True,
                capture_stderr=True,
                capture_traceback=True,
                capture_hardware_metrics=True,
            )
            N = NeptuneImageGridCallback(
                classes=self.configs["model_params"]["out_channels"],
                patch_size=self.configs["dataset_params"]["patch_size"],
            )

            cbs += [
                N
            ]
        else:
            logger = None

        if profiler==True:
            # profiler = AdvancedProfiler(dirpath=self.project.log_folder, filename="profiler")
            profiler ='simple'
            cbs+=[DeviceStatsMonitor(cpu_stats=True)]
        else:
            profiler = None

        return cbs, logger, profiler


    def set_strategy(self,devices):
        self.strategy = maybe_ddp(devices)
        if type(devices) == int and devices > 1:
            self.sync_dist = True
        else:
            self.sync_dist = False



    def maybe_alter_configs(self,batch_size,batchsize_finder,compiled):
        if batch_size :
            self.configs["dataset_params"]["batch_size"] = batch_size
            # batch_size = self.configs["dataset_params"]["batch_size"]
        if batchsize_finder == True: # note even if you set a batchsize, that will be overridden by this.
            batch_size = self.heuristic_batch_size()
            self.configs["dataset_params"]["batch_size"] = batch_size
        if compiled:
            self.configs["model_params"]["compiled"] = compiled


    def qc_configs(self, configs,project):
        ratios = configs['dataset_params']["fgbg_ratio"]
        labels_fg = project.global_properties['labels_all']
        labels = [0]+labels_fg
        if isinstance(ratios,list):
            assert(a:=(len(ratios))==(b:=len(labels))), "Class ratios {0} do not match number of labels in dataset {1}".format(a,b)
        else:
            assert  isinstance(ratios, int), "If no list is provided, fgbg_ratio must be an integer"
        configs = self.select_plan(configs)

    def select_plan(self,configs):
        plan = configs['dataset_params']['plan']
        plan_keys = [key for key in configs.keys() if 'plan' in key]
        plan_selected = configs['plan'+str(plan)]
        configs['plan']=plan_selected
        for key in plan_keys:
            configs.pop(key)
        return configs


    def heuristic_batch_size(self):
        ram = psutil.virtual_memory()[3] / 1e9
        if ram < 15:
            return 6
        elif ram > 15 and ram < 32:
            return 8
        elif ram > 32 and ram < 48:
            return 20
        else:
            return 48

    def init_dm(self,cache_rate, ):
        DMClass = self.resolve_datamanager(self.configs["dataset_params"]["mode"])
        D = DMClass(
            self.project,
            dataset_params=self.configs["dataset_params"],
            plan=self.configs["plan"],
            transform_factors=self.configs["transform_factors"],
            affine3d=self.configs["affine3d"],
            batch_size=self.configs["dataset_params"]["batch_size"],
            cache_rate =cache_rate
        )

        return D

    def init_trainer(self,epochs):
        N = UNetTrainer(
            self.project,
            self.configs["dataset_params"],
            self.configs["model_params"],
            self.configs["loss_params"],
            lr=self.lr,
            max_epochs=epochs,
            sync_dist=self.sync_dist,
        )
        return  N


    def load_trainer(self,**kwargs):
        try:
            N = UNetTrainer.load_from_checkpoint(
                self.ckpt, project=self.project, dataset_params=self.configs['dataset_params'], lr=self.lr,**kwargs
            )
            print("Model loaded from checkpoint: ",self.ckpt)
        except:
            tr()
            ckpt_state = self.state_dict["state_dict"]
            ckpt_state_updated = fix_dict_keys(ckpt_state, "model", "model._orig_mod")
            print(ckpt_state_updated.keys())
            state_dict_neo = self.state_dict.copy()
            state_dict_neo["state_dict"] = ckpt_state_updated

            ckpt_old = self.ckpt.str_replace("_bkp", "")
            ckpt_old = self.ckpt.str_replace(".ckpt", ".ckpt_bkp")
            torch.save(state_dict_neo, self.ckpt)
            shutil.move(self.ckpt, ckpt_old)

            N = UNetTrainer.load_from_checkpoint(
                self.ckpt, project=self.project, dataset_params=self.configs['dataset_params'], lr=self.lr,**kwargs
            )
        return  N


    def load_dm(self):
        DMClass = self.resolve_datamanager(
            self.state_dict["datamodule_hyper_parameters"]["dataset_params"]["mode"]
        )
        D = DMClass.load_from_checkpoint(self.ckpt, project=self.project)
        return D

    def resolve_datamanager(self, mode: str):
        assert mode in [
            "patch",
            "whole",
            "lbd",
            "source",
        ], "mode must be 'patch', 'whole' or 'source'"
        if mode == "patch":
            DMClass = DataManagerPatch
        elif mode == "source":
            DMClass = DataManagerSource
        elif mode=="lbd":
            DMClass = DataManagerLBD
        else:
            raise NotImplementedError(
                "lowres whole image transforms not yet supported."
            )
        return DMClass

    def fit(self):
        self.trainer.fit(model=self.N, datamodule=self.D, ckpt_path=self.ckpt)

    def fix_state_dict_keys(self, bad_str="model", good_str="model._orig_mod"):
        state_dict = torch.load(self.ckpt)
        ckpt_state = state_dict["state_dict"]
        ckpt_state_updated = fix_dict_keys(ckpt_state, bad_str, good_str)
        state_dict_neo = state_dict.copy()
        state_dict_neo["state_dict"] = ckpt_state_updated

        ckpt_old = self.ckpt.str_replace(".ckpt", ".ckpt_bkp")
        shutil.move(self.ckpt, ckpt_old)

        torch.save(state_dict_neo, self.ckpt)
        return ckpt_old


# %%


if __name__ == "__main__":
    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")

    from fran.utils.common import *
    from torch.profiler import profile, record_function, ProfilerActivity
    project_title = "litsmc"
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    )
    configuration_filename = "/s/fran_storage/projects/litsmc/experiment_configs.xlsx"
    configuration_filename = None

    conf = ConfigMaker(
        proj, raytune=False
    ).config

    # conf['model_params']['lr']=1e-3

    pp(conf)
# %%
    device_id = 1
    bs = 6# if none, will get it from the conf file 
    run_name ='LITS-940'
    run_name = None
    compiled = False
    profiler=False

    batch_finder = False
    neptune =False 
    tags = []
    cache_rate=0.0
    description = f"Using DynUnet"
    Tm = TrainingManager(proj, conf, run_name)
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=600,
        batchsize_finder=batch_finder,
        profiler=profiler,
        neptune=neptune,
        tags=tags,
        description=description,
        cache_rate=cache_rate
    )
    # Tm.D.batch_size=8
    Tm.N.compiled = compiled
# %%
    m = Tm.N.model

    patch_size= Tm.N.dataset_params['patch_size']
    summ = summary(Tm.N.model, input_size=tuple([1,1]+patch_size),col_names=["input_size","output_size","kernel_size"],depth=4, verbose=0,device='cuda')
# %%
    Tm.fit()
        # model(inputs)
# %%

    Tm.D.setup()
    D = Tm.D
    ds = Tm.D.valid_ds
    ds = Tm.D.train_ds
# %%
    for i,id in enumerate(ds):
        print(i)
# %%
    dici = ds[7]
    dici = ds.data[7]
    dici = ds.transform(dici)

# %%
    L = LoadImaged(
            keys=["image", "lm" ],
            image_only=True,
            ensure_channel_first=False,
            simple_keys=True,
        )
    L.register(TorchReader())


    E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
    Rtr = RandCropByPosNegLabeld(
        keys=["image", "lm"],
        label_key="lm",
        image_key="image",
        fg_indices_key="lm_fg_indices",
        bg_indices_key="lm_bg_indices",
        image_threshold=-2600,
        spatial_size=D.src_dims,
        pos=3,
        neg=1,
        num_samples=D.dataset_params["samples_per_file"],
        lazy=True,
        allow_smaller=False,
    )
    Ld= LoadDict(keys= ["indices"],select_keys =  ["lm_fg_indices","lm_bg_indices"])

    Rva = RandCropByPosNegLabeld(
            keys=["image", "lm"],
            label_key="lm",
            image_key="image",
            image_threshold=-2600,
            fg_indices_key="lm_fg_indices",
            bg_indices_key="lm_bg_indices",
            spatial_size=D.dataset_params["patch_size"],
            pos=3,
            neg=1,
            num_samples=D.dataset_params["samples_per_file"],
            lazy=False,
            allow_smaller=True,
        )
    Re = ResizeWithPadOrCropd(
            keys=["image", "lm"],
            spatial_size=D.dataset_params["patch_size"],
            lazy=False,
        )

# %%
    D.prepare_data()
    D.setup(None)
# %%
    D.valid_ds[7]

# %%
    dici = D.valid_ds.data[7]
    dici = L(dici)
    dici =Ld(dici)
    dici = D.transforms_dict['E'](dici)
    dici = D.transforms_dict['Rva'](dici)
    dici = Re(dici[1])

# %%
    ImageMaskViewer([dici[0]['image'][0],dici[0]['lm'][0]])

# %%
    Ld= LoadDict(keys= ["indices"],select_keys =  ["lm_fg_indices","lm_bg_indices"])
    dici = Ld(dici)
# %%

# %%


    fn = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/images/lits_115.pt"
    fn2 = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/lms/lits_115.pt"
    tt = torch.load(fn)
    tt2 =  torch.load(fn2)
    ImageMaskViewer([tt,tt2])

# %%
    dl = Tm.D.train_dataloader()
    dl2 = Tm.D.val_dataloader()
    iteri = iter(dl)
    iteri2 = iter(dl2)
    batch = next(iteri2)
# %%



    Re = ResizeWithPadOrCropd(
                keys=["image", "lm"],
                spatial_size=D.dataset_params["patch_size"],
                lazy=False,
            )
# %%
    dici = Re(dici)
# %%
    dici = ds[1]
    dici = ds.data[0]
    keys_tr="L,E,Ind,Rtr,F1,F2,A,Re,N,I"
    keys_val="L,E,Ind,Rva,Re,N"
    keys_tr = keys_tr.split(",")
    keys_val = keys_val.split(",")


# %%
    dici = ds.data[5].copy()
    for k in keys_val[:3]:
        tfm = D.transforms_dict[k]
        dici =tfm(dici)
# %%

    ind =0
    dici = ds.data[ind]
    ImageMaskViewer([dici['image'][0],dici['lm'][0]])
    ImageMaskViewer([dici[ind]['image'][0],dici[ind]['lm'][0]])
# %%
    tfm2 = D.transforms_dict[keys_tr[5]]

# %%
    for didi in dici:
        dd = tfm2(didi)
# %%
    idx= 0
    ds.set_bboxes_labels(idx)
    if ds.enforce_ratios == True:
        ds.mandatory_label = ds.randomize_label()
        ds.maybe_randomize_idx()

    filename, bbox = ds.get_filename_bbox()
    img, lm= ds.load_tensors(filename)
    dici = {"image": img, "lm": lm, "bbox": bbox}
    dici = ds.transform(dici)

    E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
    dici = E(dici)
# %%
    # img = ds.create_metatensor(img_fn)
    # label = ds.create_metatensor(label_fn)
    dici = ds.data[3]
    dici =ds[3]
    dici[0]['image']
    dat =ds.data[5]
    dici = ds.transform(dat)
    type(dici)
    dici = ds[4]
    dici.keys()
    dat
    dici = {"image": img_fn, "lm": label_fn}
    im = torch.load(img_fn)

    im.shape

# %%

# %%
    b = next(iteri2)

    b['image'].shape
    m = Tm.N.model
    N = Tm.N

# %%
    for x  in range(len(ds)):
        casei= ds[x]
        for a in range(len(casei)):
            print(casei[a]['image'].shape)
# %%
    for i,b in enumerate(dl):
        print("\----------------------------")
        print(b['image'].shape )
        print(b['label'].shape )
# %%
    # b2 = next(iter(dl2))
    batch = b
    inputs, target, bbox = batch["image"], batch["lm"], batch["bbox"]

    [pp(a["filename"]) for a in bbox]
# %%
    preds = N.model(inputs.cuda())
    pred = preds[0]
    pred = pred.detach().cpu()
    pp(pred.shape)
# %%
    n = 1
    img = inputs[n, 0]
    mask = target[n, 0]
# %%
    ImageMaskViewer([img.permute(2, 1, 0), mask.permute(2, 1, 0)])
# %%
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacings/litsmall/spc_080_080_150/images/lits_4.pt"
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacings/litstp/spc_080_080_150/images/lits_4.pt"
    fn2 = "/home/ub/datasets/preprocessed/lits32/patches/spc_080_080_150/dim_192_192_128/masks/lits_4_1.pt"
    img = torch.load(fn)
    mask = torch.load(fn2)
    pp(img.shape)
# %%

    ImageMaskViewer([img, mask])
# %%
# %%

    Tm.trainer.callback_metrics
# %%
    ckpt = Path(
        "/s/fran_storage/checkpoints/litsmc/Untitled/LITS-709/checkpoints/epoch=81-step=1886.ckpt"
    )
    kk = torch.load(ckpt)
    kk.keys()
    kk["datamodule_hyper_parameters"]
# %%
