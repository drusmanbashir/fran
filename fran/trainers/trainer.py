# %%
import shutil

from fastcore.all import in_ipython
import ipdb
from label_analysis.merge import pbar
from lightning.pytorch import Trainer as TrainerL
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import AdvancedProfiler
from monai.transforms.io.dictionary import LoadImaged
from utilz.helpers import pp
from utilz.string import headline

from fran.managers import Project, UNetManager
from fran.managers.data.training import DataManagerDual
from fran.managers.unet import maybe_ddp
from fran.trainers.base import checkpoint_from_model_id
from fran.utils.config_parsers import ConfigMaker

tr = ipdb.set_trace

from pathlib import Path

import psutil
import torch._dynamo
from monai.transforms.croppad.dictionary import (RandCropByPosNegLabeld,
                                                 ResizeWithPadOrCropd)
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from utilz.imageviewers import ImageMaskViewer

from fran.callback.nep import NeptuneImageGridCallback
from fran.managers.data import (DataManagerBaseline, DataManagerLBD,
                                DataManagerPatch, DataManagerSource,
                                DataManagerWhole, DataManagerWID)

torch._dynamo.config.suppress_errors = True
import warnings

from lightning.pytorch.callbacks import (DeviceStatsMonitor,
                                         LearningRateMonitor, TQDMProgressBar)

from fran.managers.nep import NeptuneManager
from fran.utils.common import COMMON_PATHS

try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass

import torch


def fix_dict_keys(input_dict, old_string, new_string):
    output_dict = {}
    for key in input_dict.keys():
        neo_key = key.replace(old_string, new_string)
        output_dict[neo_key] = input_dict[key]
    return output_dict


# class NeptuneCallback(Callback):
# def on_train_epoch_start(self, trainer, pl_module):
#     trainer.logger.experiment["training/epoch"] = trainer.current_epoch


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
    ):
        self.maybe_alter_configs(batch_size, batchsize_finder, compiled)
        self.set_lr(lr)
        self.set_strategy(devices)
        self.init_dm_unet(epochs, batch_size)
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

    def init_dm_unet(self, epochs, batch_size):
        if self.ckpt:
            self.D = self.load_dm(batch_size=batch_size)
            self.configs["dataset_params"] = self.D.configs["dataset_params"]
            self.N = self.load_trainer()

        else:
            self.D = self.init_dm()
            self.N = self.init_trainer(epochs)
        print("Data Manager initialized.\n {}".format(self.D))

    def set_lr(self, lr):
        if lr and not self.ckpt:
            self.lr = lr
        elif lr and self.ckpt:
            self.lr = lr
            self.state_dict = torch.load(
                self.ckpt, weights_only=False, map_location="cpu"
            )
            self.state_dict["lr_schedulers"][0]["_last_lr"][0] = lr
            torch.save(self.state_dict, self.ckpt)

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
                capture_stdout=True,
                capture_stderr=True,
                capture_traceback=True,
                capture_hardware_metrics=True,
            )
            N = NeptuneImageGridCallback(
                classes=self.configs["model_params"]["out_channels"],
                patch_size=self.configs["plan_train"]["patch_size"],
                epoch_freq=5,  # skip how many epochs.
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
        labels_fg = configs["model_params"]["out_channels"] - 1
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
        D = DataManagerDual(
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

    def load_trainer(self, **kwargs):
        try:
            N = UNetManager.load_from_checkpoint(
                self.ckpt,
                map_location="cpu",
                **kwargs,
            )
            print("Model loaded from checkpoint: ", self.ckpt)
        # except: #CODE: exception should be specific

        except RuntimeError as e:
            msg = str(e)
            headline(msg)
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
                plan=self.configs["plan"],
                lr=self.lr,
                **kwargs,
            )
        return N

    def load_dm(self, batch_size=None):
        D = DataManagerDual.load_from_checkpoint(
            self.ckpt,
            project_title=self.project.project_title,
            batch_size=batch_size,
            map_location = "cpu",
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

    def fix_state_dict_keys(self, bad_str="model", good_str="model._orig_mod"):
        state_dict = torch.load(self.ckpt, map_location="cpu", weights_only=False)
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
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR>

    # CODE: Project or configs should be the only arg not both
    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")

    from fran.utils.common import *

    proj_nodes = Project(project_title="nodes")
    proj_tsl = Project(project_title="totalseg")
    proj_litsmc = Project(project_title="litsmc")
    conf_litsmc = ConfigMaker(proj_litsmc, raytune=False).configs
    conf_nodes = ConfigMaker(proj_nodes, raytune=False).configs
    conf_tsl = ConfigMaker(proj_tsl, raytune=False).configs

    # conf['model_params']['lr']=1e-3
    conf_litsmc["dataset_params"]["cache_rate"]
    # run_name = "LITS-1007"
    device_id = 0
    run_none = None
    run_tsl = "LITS-1120"
    run_nodes = "LITS-1110"
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
# SECTION:-------------------- TOTALSEG TRAINING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
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
# SECTION:-------------------- LITSMC -------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>

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
# SECTION:-------------------- NODES-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
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

# SECTION:-------------------- TROUBLESHOOTING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR>

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
    D = DataManagerDual(
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
    pred = Tm.N.model(batch["image"])
# %%

    n = 1
    im = batch["image"][n][0].clone()
    pr = pred[0][n][3].clone()
    lab = batch["lm"][n][0].clone()
    lab_bin = (lab > 1).float()
# %%
    lab = lab.permute(2, 1, 0)
    im = im.permute(2, 1, 0)
    pr = pr.permute(2, 1, 0)
# %%
    ImageMaskViewer([im.detach().cpu(), pr.detach().cpu()])
    ImageMaskViewer([im.detach().cpu(), lab_bin.detach().cpu()])
# %%
    outfldr = Path("/s/fran_storage/misc")

# %%
    torch.save(im, outfldr / "im_no_tum.pt")
    torch.save(pr, outfldr / "pred_no_tum.pt")
    torch.save(lab, outfldr / "lab_no_tum.pt")
# %%
    while iteri:
        bb = next(iteri)
        lm = bb["lm"]
        labels = lm.unique()
        if (labels > 8).any():
            print("There are labels greater than 8.")
            print(bb["lm"].meta["filename_or_obj"])
            print(labels)
            tr()
        if (labels < 0).any():
            print(labels)
            tr()
            print("There are labels less than 0.")
            print(bb["image"].meta["filename_or_obj"])
# %%
    fns = [
        "/s/fran_storage/datasets/preprocessed/fixed_size/totalseg/sze_96_96_96/lms/totalseg_s1210.pt",
        "/s/fran_storage/datasets/preprocessed/fixed_size/totalseg/sze_96_96_96/lms/totalseg_s0851.pt",
        "/s/fran_storage/datasets/preprocessed/fixed_size/totalseg/sze_96_96_96/lms/totalseg_s1175.pt",
        "/s/fran_storage/datasets/preprocessed/fixed_size/totalseg/sze_96_96_96/lms/totalseg_s0726.pt",
        "/s/fran_storage/datasets/preprocessed/fixed_size/totalseg/sze_96_96_96/lms/totalseg_s0549.pt",
    ]
    for fn in fns:
        lm = torch.load(fn)
        print(lm.unique())
# %%
    pred = Tm.N(bb["image"])
# %%
    [x.shape for x in pred]
# %%
    i = 52
    dd = ds.data[i]

# %%
    im_fn = dd["image"]
    lm_fn = dd["lm"]
    im = torch.load(im_fn)
    lm = torch.load(lm_fn)
    im.shape
    lm.shape
# %%
    for i, id in enumerate(ds):
        print(i)
# %%
    dici = ds[7]
    dici = ds.data[7]
    dici = ds.transform(dici)

    #    D.train_ds[2] %%
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
# %%
    while iteri:
        batch = next(iteri)
        print(batch["image"].shape)

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
