# %%
from lightning.pytorch import Trainer as TrainerL
from lightning.pytorch.callbacks import ModelCheckpoint
import shutil
from lightning.pytorch.profilers import AdvancedProfiler
from monai.transforms.io.dictionary import LoadImaged
from fran.transforms.misc_transforms import LoadTorchDict, MetaToDict
from fran.managers import UNetManager
from fran.managers.unet import maybe_ddp
from fran.transforms.imageio import TorchReader
import ipdb

from fran.utils.helpers import pp

tr = ipdb.set_trace

from label_analysis.overlap import get_ipython
import numpy as np
from typing import Union
from pathlib import Path
from fastcore.basics import store_attr
from monai.transforms.croppad.dictionary import (
    RandCropByPosNegLabeld,
    ResizeWithPadOrCropd,
)
from monai.transforms.utility.dictionary import EnsureChannelFirstd

import psutil
import torch._dynamo
from fran.callback.nep import NeptuneImageGridCallback

from fran.evaluation.losses import CombinedLoss, DeepSupervisionLoss
from fran.managers.data import DataManagerBaseline, DataManagerLBD, DataManagerPBD, DataManagerPatch, DataManagerSource, DataManagerWhole
from fran.utils.fileio import load_yaml
from fran.utils.imageviewers import ImageMaskViewer

torch._dynamo.config.suppress_errors = True
from fran.managers.nep import NeptuneManager
import itertools as il
import operator
import warnings
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    TQDMProgressBar,
    DeviceStatsMonitor,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fran.architectures.create_network import (
    create_model_from_conf,
    pool_op_kernels_nnunet,
)
import torch.nn.functional as F
from fran.utils import COMMON_PATHS
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


def checkpoint_from_model_id(model_id, sort_method="last"):
    fldr = Path(COMMON_PATHS["checkpoints_parent_folder"])
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
    if sort_method == "last":
        ckpt = max(list_of_files, key=lambda p: p.stat().st_mtime)
    elif sort_method == "best":
        tr()
    return ckpt


# from fran.managers.base import *


# class NeptuneCallback(Callback):
# def on_train_epoch_start(self, trainer, pl_module):
#     trainer.logger.experiment["training/epoch"] = trainer.current_epoch
class Trainer:
    def __init__(self, project, config, run_name=None):
        store_attr()
        self.ckpt = None if run_name is None else checkpoint_from_model_id(run_name)
        self.qc_config(config, project)

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
    ):
        self.maybe_alter_config(batch_size, batchsize_finder, compiled)
        self.set_lr(lr)
        self.set_strategy(devices)
        self.init_dm_unet(epochs)
        cbs, logger, profiler = self.init_cbs(neptune, profiler, tags, description)
        self.D.prepare_data()

        if self.config["model_params"]["compiled"] == True:
            self.N = torch.compile(self.N)

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

    def init_dm_unet(self, epochs):
        if self.ckpt:
            self.D = self.load_dm()
            self.config["dataset_params"] = self.D.dataset_params
            self.N = self.load_trainer()

        else:
            self.D = self.init_dm()
            self.N = self.init_trainer(epochs)

    def set_lr(self, lr):
        if lr and not self.ckpt:
            self.lr = lr
        elif lr and self.ckpt:
            self.lr = lr
            self.state_dict = torch.load(self.ckpt)
            self.state_dict["lr_schedulers"][0]["_last_lr"][0] = lr
            torch.save(self.state_dict, self.ckpt)

        elif lr is None and self.ckpt:
            self.state_dict = torch.load(self.ckpt)
            self.lr = self.state_dict["lr_schedulers"][0]["_last_lr"][0]
        else:
            self.lr = self.config["model_params"]["lr"]

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
                capture_stdout=True,
                capture_stderr=True,
                capture_traceback=True,
                capture_hardware_metrics=True,
            )
            N = NeptuneImageGridCallback(
                classes=self.config["model_params"]["out_channels"],
                patch_size=self.config["dataset_params"]["patch_size"],
            )

            cbs += [N]
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
        self.strategy = maybe_ddp(devices)
        if type(devices) == int and devices > 1:
            self.sync_dist = True
        else:
            self.sync_dist = False

    def maybe_alter_config(self, batch_size, batchsize_finder, compiled):
        if batch_size:
            self.config["dataset_params"]["batch_size"] = batch_size
            # batch_size = self.config["dataset_params"]["batch_size"]
        if (
            batchsize_finder == True
        ):  # note even if you set a batchsize, that will be overridden by this.
            batch_size = self.heuristic_batch_size()
            self.config["dataset_params"]["batch_size"] = batch_size
        if compiled:
            self.config["model_params"]["compiled"] = compiled

    def qc_config(self, config, project):
        ratios = config["dataset_params"]["fgbg_ratio"]
        labels_fg = project.global_properties["labels_all"]
        labels = [0] + labels_fg
        if isinstance(ratios, list):
            assert (
                a := (len(ratios)) == (b := len(labels))
            ), "Class ratios {0} do not match number of labels in dataset {1}".format(
                a, b
            )
        else:
            assert isinstance(
                ratios, int
            ), "If no list is provided, fgbg_ratio must be an integer"

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

    def init_dm(self):
        cache_rate = self.config["dataset_params"]["cache_rate"]
        ds_type = self.config["dataset_params"]["ds_type"]
        DMClass = self.resolve_datamanager(self.config["plan"]["mode"])
        D = DMClass(
            self.project,
            dataset_params=self.config["dataset_params"],
            config=self.config,
            transform_factors=self.config["transform_factors"],
            affine3d=self.config["affine3d"],
            batch_size=self.config["dataset_params"]["batch_size"],
            cache_rate=cache_rate,
            ds_type=ds_type,
        )

        return D

    def init_trainer(self, epochs):
        N = UNetManager(
            self.project,
            self.config["dataset_params"],
            self.config["model_params"],
            self.config["loss_params"],
            lr=self.lr,
            max_epochs=epochs,
            sync_dist=self.sync_dist,
        )
        return N

    def load_trainer(self, **kwargs):
        try:
            N = UNetManager.load_from_checkpoint(
                self.ckpt,
                project=self.project,
                dataset_params=self.config["dataset_params"],
                lr=self.lr,
                **kwargs,
            )
            print("Model loaded from checkpoint: ", self.ckpt)
        except:
            tr()
            ckpt_state = self.state_dict["state_dict"]
            ckpt_state_updated = fix_dict_keys(ckpt_state, "model", "model._orig_mod")
            # print(ckpt_state_updated.keys())
            state_dict_neo = self.state_dict.copy()
            state_dict_neo["state_dict"] = ckpt_state_updated
            ckpt_old = self.ckpt.str_replace("_bkp", "")
            ckpt_old = self.ckpt.str_replace(".ckpt", ".ckpt_bkp")
            torch.save(state_dict_neo, self.ckpt)
            shutil.move(self.ckpt, ckpt_old)

            N = UNetManager.load_from_checkpoint(
                self.ckpt,
                project=self.project,
                dataset_params=self.config["dataset_params"],
                lr=self.lr,
                **kwargs,
            )
        return N

    def load_dm(self):
        DMClass = self.resolve_datamanager(
            self.state_dict["datamodule_hyper_parameters"]["config"]["plan"]["mode"]
        )
        D = DMClass.load_from_checkpoint(self.ckpt, project=self.project)
        return D

    def resolve_datamanager(self, mode: str):
        if mode == "patch":
            DMClass = DataManagerPatch
        elif mode == "source":
            DMClass = DataManagerSource
        elif mode =="whole":
            DMClass = DataManagerWhole
        elif mode == "lbd":
            DMClass = DataManagerLBD
        elif mode == "pbd":
            DMClass = DataManagerPBD
        elif mode == "baseline":
            DMClass = DataManagerBaseline
        else:
            raise NotImplementedError(
                "Mode {} is not supported for datamanager".format(mode)
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
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR>


    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")

    from fran.utils.common import *
    from torch.profiler import profile, record_function, ProfilerActivity

    project_title = "nodes"
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_config_wholeimage.xlsx"
    )
    configuration_filename = "/s/fran_storage/projects/litsmc/experiment_config.xlsx"
    configuration_filename = None

    conf = ConfigMaker(proj, raytune=False).config

    # conf['model_params']['lr']=1e-3

    # conf['dataset_params']['plan']=5
# %%
    # run_name = "LITS-1007"
    # device_id = 1
    device_id = 1
    run_name ='LITS-1018'
    run_name = None
    bs = 10# 5 is good if LBD with 2 samples per case
    # run_name ='LITS-1003'
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    tags = []
    description = f""
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

# %%
# SECTION:-------------------- TROUBLESHOOTING-------------------------------------------------------------------------------------- <CR>

    Tm.D.setup()
    D = Tm.D
    ds = Tm.D.valid_ds
    ds = Tm.D.train_ds
# %%
    i = ds[12]

    for i, id in enumerate(ds):
        print(i)
# %%
    dici = ds[7]
    dici = ds.data[7]
    dici = ds.transform(dici)

# %%
    L = LoadImaged(
        keys=["image", "lm"],
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
        num_samples=D.plan["samples_per_file"],
        lazy=True,
        allow_smaller=False,
    )
    Ld = LoadDict(keys=["indices"], select_keys=["lm_fg_indices", "lm_bg_indices"])

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
        num_samples=D.plan["samples_per_file"],
        lazy=False,
        allow_smaller=True,
    )
    Re = ResizeWithPadOrCropd(
        keys=["image", "lm"],
        spatial_size=D.dataset_params["patch_size"],
        lazy=False,
    )

    Ind = MetaToDict(keys=["lm"], meta_keys=["lm_fg_indices", "lm_bg_indices"])
# %%
    D.prepare_data()
    D.setup(None)
# %%
    dici = D.data_train[0]
    D.valid_ds.data[0]

# %%
    dici = D.valid_ds.data[7]
    dici = L(dici)
    dici = Ind(dici)
    dici = Ld(dici)
    dici = D.transforms_dict["E"](dici)
    dici = D.transforms_dict["Rva"](dici)
    dici = Re(dici[1])

# %%
    ImageMaskViewer([dici[0]["image"][0], dici[0]["lm"][0]])

# %%
    Ld = LoadDict(keys=["indices"], select_keys=["lm_fg_indices", "lm_bg_indices"])
    dici = Ld(dici)
# %%

# %%

    fn = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/images/lits_115.pt"
    fn2 = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/lms/lits_115.pt"
    tt = torch.load(fn)
    tt2 = torch.load(fn2)
    ImageMaskViewer([tt, tt2])

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
    keys_tr = "L,E,Ind,Rtr,F1,F2,A,Re,N,I"
    keys_val = "L,E,Ind,Rva,Re,N"
    keys_tr = keys_tr.split(",")
    keys_val = keys_val.split(",")

# %%
    dici = ds.data[5].copy()
    for k in keys_val[:3]:
        tfm = D.transforms_dict[k]
        dici = tfm(dici)
# %%

    ind = 0
    dici = ds.data[ind]
    ImageMaskViewer([dici["image"][0], dici["lm"][0]])
    ImageMaskViewer([dici[ind]["image"][0], dici[ind]["lm"][0]])
# %%
    tfm2 = D.transforms_dict[keys_tr[5]]

# %%
    for didi in dici:
        dd = tfm2(didi)
# %%
    idx = 0
    ds.set_bboxes_labels(idx)
    if ds.enforce_ratios == True:
        ds.mandatory_label = ds.randomize_label()
        ds.maybe_randomize_idx()

    filename, bbox = ds.get_filename_bbox()
    img, lm = ds.load_tensors(filename)
    dici = {"image": img, "lm": lm, "bbox": bbox}
    dici = ds.transform(dici)

    E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
    dici = E(dici)
# %%
    # img = ds.create_metatensor(img_fn)
    # label = ds.create_metatensor(label_fn)
    dici = ds.data[3]
    dici = ds[3]
    dici[0]["image"]
    dat = ds.data[5]
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

    b["image"].shape
    m = Tm.N.model
    N = Tm.N

# %%
    for x in range(len(ds)):
        casei = ds[x]
        for a in range(len(casei)):
            print(casei[a]["image"].shape)
# %%
    for i, b in enumerate(dl):
        print("\----------------------------")
        print(b["image"].shape)
        print(b["label"].shape)
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
    kk = torch.load(self.ckpt)
    kk["datamodule_hyper_parameters"].keys()
    kk.keys()
    kk["datamodule_hyper_parameters"]
# %%
