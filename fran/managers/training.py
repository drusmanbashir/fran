# %%
import shutil
from lightning.pytorch.profilers import AdvancedProfiler

import psutil
import random
import torch._dynamo

from fran.managers.data import (DataManager, DataManagerPatch,
                                DataManagerShort, DataManagerSource)
from fran.utils.batch_size_scaling import (_reset_dataloaders,
                                           _scale_batch_size2)

torch._dynamo.config.suppress_errors = True
from fran.managers.neptune import NeptuneManager
import itertools as il
import operator
import warnings
from typing import Any

import neptune as nt
import torch
from lightning.pytorch import LightningModule, Trainer
# from fastcore.basics import GenttAttr
from lightning.pytorch.callbacks import (Callback, LearningRateMonitor,
                                         TQDMProgressBar, DeviceStatsMonitor)
from neptune.types import File
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose
from torchvision.utils import make_grid

from fran.architectures.create_network import (create_model_from_conf, nnUNet,
                                               pool_op_kernels_nnunet)
from fran.data.dataloader import img_mask_bbox_collated
from fran.data.dataset import (ImageMaskBBoxDatasetd, MaskLabelRemap2,
                               NormaliseClipd)
# from fastai.learner import *
from fran.evaluation.losses import *
from fran.transforms.spatialtransforms import one_hot
from fran.transforms.totensor import ToTensorT
from fran.utils.common import *
from fran.utils.fileio import *
from fran.utils.helpers import *
from fran.utils.helpers import folder_name_from_list
from fran.utils.imageviewers import *
from fran.utils.helpers import *

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
def normalize(tensr, intensity_percentiles=[0.0, 1.0]):
    tensr = (tensr - tensr.min()) / (tensr.max() - tensr.min())
    tensr = tensr.to("cpu", dtype=torch.float32)
    qtiles = torch.quantile(tensr, q=torch.tensor(intensity_percentiles))

    vmin = qtiles[0]
    vmax = qtiles[1]
    tensr[tensr < vmin] = vmin
    tensr[tensr > vmax] = vmax
    return tensr


# class NeptuneCallback(Callback):
# def on_train_epoch_start(self, trainer, pl_module):
#     trainer.logger.experiment["training/epoch"] = trainer.current_epoch


class NeptuneImageGridCallback(Callback):
    def __init__(
        self,
        classes,
        patch_size,
        grid_rows=6,
        imgs_per_batch=4,
        publish_deep_preds=False,
        apply_activation=True,
        epoch_freq=2,  # skip how many epochs.
    ):
        if not isinstance(patch_size, torch.Size):
            patch_size = torch.Size(patch_size)
        self.stride = int(patch_size[0] / imgs_per_batch)
        store_attr()

    #
    def on_train_start(self, trainer, pl_module):
        trainer.store_preds = False  # DO NOT SET THIS TO TRUE. IT WILL BUG
        len_dl = int(len(trainer.train_dataloader) / trainer.accumulate_grad_batches)
        self.freq = np.maximum(2, int(len_dl / self.grid_rows))

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.epoch_freq == 0:
            super().on_train_epoch_start(trainer, pl_module)
            self.grid_imgs = []
            self.grid_preds = []
            self.grid_labels = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_grid_created = False

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.current_epoch % self.epoch_freq == 0:
            if trainer.global_step % self.freq == 0:
                trainer.store_preds = True
            else:
                trainer.store_preds = False
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.store_preds == True:
            self.populate_grid(pl_module, batch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.epoch_freq == 0:
            if self.validation_grid_created == False:
                self.populate_grid(pl_module, batch)
                self.validation_grid_created = True

    #
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.epoch_freq == 0:
            grd_final = []
            for grd, category in zip(
                [self.grid_imgs, self.grid_preds, self.grid_labels],
                ["imgs", "preds", "labels"],
            ):
                grd = torch.cat(grd)
                if category == "imgs":
                    grd = normalize(grd)
                grd_final.append(grd)
            grd = torch.stack(grd_final)
            grd2 = (
                grd.permute(1, 0, 2, 3, 4)
                .contiguous()
                .view(-1, 3, grd.shape[-2], grd.shape[-1])
            )
            grd3 = make_grid(grd2, nrow=self.imgs_per_batch * 3)
            grd4 = grd3.permute(1, 2, 0)
            grd4 = np.array(grd4)
            trainer.logger.experiment["images"].append(File.as_image(grd4))

    def populate_grid(self, pl_module, batch):
        def _randomize():
            n_slices = img.shape[-1]
            batch_size = img.shape[0]
            self.slices = [
                random.randrange(0, n_slices) for i in range(self.imgs_per_batch)
            ]
            self.batches = [
                random.randrange(0, batch_size) for i in range(self.imgs_per_batch)
            ]

        img = batch["image"].cpu()

        label = batch["label"].cpu()
        label = label.squeeze(1)
        label = one_hot(label, self.classes, axis=1)
        img = img.cpu()
        # pred = torch.rand(img.shape,device='cuda')
        # pred.unsqueeze_(0) # temporary hack)
        # pred = pred.cpu()
        pred = pl_module.pred
        if isinstance(pred, Union[list, tuple]):
            pred = pred[0]
        elif pred.dim() == img.dim() + 1:  # deep supervision
            pred = pred[:, 0, :]  # Bx1xCXHXWxD

        # if self.apply_activation == True:
        pred = F.softmax(pred.to(torch.float32), dim=1)

        _randomize()
        img, label, pred = (
            self.img_to_grd(img),
            self.img_to_grd(label),
            self.img_to_grd(pred),
        )
        img, label, pred = (
            self.fix_channels(img),
            self.fix_channels(label),
            self.fix_channels(pred),
        )

        self.grid_imgs.append(img)
        self.grid_preds.append(pred)
        self.grid_labels.append(label)

    def img_to_grd(self, batch):
        imgs = batch[self.batches, :, :, :, self.slices].clone()
        return imgs

    def fix_channels(self, tnsr):
        if tnsr.shape[1] == 1:
            tnsr = tnsr.repeat(1, 3, 1, 1)
        elif tnsr.shape[1] == 2:
            tnsr = tnsr[:, 1:, :, :]
            tnsr = tnsr.repeat(1, 3, 1, 1)
        elif tnsr.shape[1] > 3:
            chs = tnsr.shape[1]
            tnsr = tnsr[:, chs - 3 : :, :]
        else:
            # tnsr already in 3d
            pass
        return tnsr

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
        inputs, target = batch["image"], batch["label"]
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

    def create_model(self):
        model = create_model_from_conf(self.model_params, self.dataset_params)
        # if self.checkpoints_folder:
        #     load_checkpoint(self.checkpoints_folder, model)
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
                # self.net_num_pool_op_kernel_sizes = [
                #     [2, 2, 2],
                # ] * num_pool
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
            # cbs += [DownsampleMaskForDS(self.deep_supervision_scales)]

        else:
            loss_func = CombinedLoss(
                **self.loss_params, fg_classes=self.model_params["out_channels"] - 1
            )
        return model, loss_func

    def populate_grid(self, img, label, pred):
        def _randomize():
            n_slices = img.shape[-1]
            batch_size = img.shape[0]
            self.slices = [
                random.randrange(0, n_slices) for i in range(self.imgs_per_batch)
            ]
            self.batches = [
                random.randrange(0, batch_size) for i in range(self.imgs_per_batch)
            ]

        img = img.cpu()
        label = label.cpu()
        label = label.squeeze(1)
        label = one_hot(label, self.classes, axis=1)
        pred = pred.cpu()
        if pred.dim() == img.dim() + 1:  # deep supervision
            pred = pred[0]

        # if self.apply_activation == True:
        pred = F.softmax(pred.to(torch.float32), dim=1)

        _randomize()
        img, label, pred = (
            self.img_to_grd(img),
            self.img_to_grd(label),
            self.img_to_grd(pred),
        )
        img, label, pred = (
            self.fix_channels(img),
            self.fix_channels(label),
            self.fix_channels(pred),
        )

        self.grid_imgs.append(img)
        self.grid_preds.append(pred)
        self.grid_labels.append(label)

    def img_to_grd(self, batch):
        imgs = batch[self.batches, :, :, :, self.slices].clone()
        return imgs

    def fix_channels(self, tnsr):
        if tnsr.shape[1] == 1:
            tnsr = tnsr.repeat(1, 3, 1, 1)
        elif tnsr.shape[1] == 2:
            tnsr = tnsr[:, 1:, :, :]
        elif tnsr.shape[1] > 3:
            chs = tnsr.shape[1]
            tnsr = tnsr[:, chs - 3 : :, :]
        else:
            # tnsr already in 3d
            pass
        return tnsr


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

    def setup(
        self,
        batch_size=8,
        cbs=[],
        logging_freq=25,
        lr=None,
        devices=1,
        compiled=False,
        neptune=True,
        profiler=False,
        tags=[],
        description="",
        epochs=1000,
        batchsize_finder=False,
    ):
        if batchsize_finder == True:
            batch_size = self.heuristic_batch_size()
        if lr:
            self.configs["model_params"]["lr"] = lr
        self.configs["model_params"]["compiled"] = compiled
        strategy = maybe_ddp(devices)
        if type(devices) == int and devices > 1:
            sync_dist = True
        else:
            sync_dist = False

        if self.ckpt:
            self.D, self.N = self.load_ckpts(lr=lr)
        else:
            self.D, self.N = self.init_D_N(batch_size, epochs, sync_dist)
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
                N,
                LearningRateMonitor(logging_interval="epoch"),
                TQDMProgressBar(refresh_rate=3),
            ]
        else:
            logger = None
        if profiler==True:
            # profiler = AdvancedProfiler(dirpath=self.project.log_folder, filename="profiler")
            profiler ='simple'
            cbs+=[DeviceStatsMonitor(cpu_stats=True)]
        else:
            profiler = None

        self.D.prepare_data()

        if self.configs["model_params"]["compiled"] == True:
            self.N = torch.compile(self.N)

        self.trainer = Trainer(
            callbacks=cbs,
            accelerator="gpu",
            devices=devices,
            precision="16-mixed",
            profiler = profiler,
            logger=logger,
            max_epochs=epochs,
            log_every_n_steps=logging_freq,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            default_root_dir=self.project.checkpoints_parent_folder,
            strategy=strategy
            # strategy='ddp_find_unused_parameters_true'
        )

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

    def init_D_N(self, batch_size, epochs, sync_dist, batch_finder=False):
        DMClass = self.resolve_datamanager(self.configs["dataset_params"]["mode"])
        D = DMClass(
            self.project,
            dataset_params=self.configs["dataset_params"],
            transform_factors=self.configs["transform_factors"],
            affine3d=self.configs["affine3d"],
            batch_size=batch_size,
        )
        N = UNetTrainer(
            self.project,
            self.configs["dataset_params"],
            self.configs["model_params"],
            self.configs["loss_params"],
            lr=self.configs["model_params"]["lr"],
            max_epochs=epochs,
            sync_dist=sync_dist,
        )
        return D, N

    def load_ckpts(self, lr=None):
        state_dict = torch.load(self.ckpt)
        DMClass = self.resolve_datamanager(
            state_dict["datamodule_hyper_parameters"]["dataset_params"]["mode"]
        )
        D = DMClass.load_from_checkpoint(self.ckpt, project=self.project)
        if lr is None:
            lr = state_dict["lr_schedulers"][0]["_last_lr"][0]
        else:
            state_dict["lr_schedulers"][0]["_last_lr"][0] = lr
            torch.save(state_dict, self.ckpt)
        try:
            N = UNetTrainer.load_from_checkpoint(
                self.ckpt, project=self.project, dataset_params=D.dataset_params, lr=lr
            )
        except:
            tr()
            ckpt_state = state_dict["state_dict"]
            ckpt_state_updated = fix_dict_keys(ckpt_state, "model", "model._orig_mod")
            print(ckpt_state_updated.keys())
            state_dict_neo = state_dict.copy()
            state_dict_neo["state_dict"] = ckpt_state_updated

            ckpt_old = self.ckpt.str_replace("_bkp", "")
            ckpt_old = self.ckpt.str_replace(".ckpt", ".ckpt_bkp")
            torch.save(state_dict_neo, self.ckpt)
            shutil.move(self.ckpt, ckpt_old)

            N = UNetTrainer.load_from_checkpoint(
                self.ckpt, project=self.project, dataset_params=D.dataset_params, lr=lr
            )
        return D, N

    def resolve_datamanager(self, mode: str):
        assert mode in [
            "patch",
            "whole",
            "source",
        ], "mode must be 'patch', 'whole' or 'source'"
        if mode == "patch":
            DMClass = DataManagerPatch
        elif mode == "source":
            DMClass = DataManagerSource
        else:
            raise NotImplementedError(
                "lowres whole image transforms not yet supported."
            )
        return DMClass

    def fit(self):
        # if self.configs['model_params']['compiled']==True:
        #     self.N = torch.compile(self.)
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
    project_title = "totalseg"
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    )
    configuration_filename = "/s/fran_storage/projects/litsmc/experiment_configs.xlsx"
    configuration_filename = None

    conf = ConfigMaker(
        proj, raytune=False, configuration_filename=configuration_filename
    ).config

    global_props = load_dict(proj.global_properties_filename)
# %%
    conf["model_params"]["arch"] = "nnUNet"
    # conf['model_params']['lr']=1e-3

# %%
    device_id = 1
# %%
    bs = 1
    run_name = None
    run_name ='LITS-827'
    compiled = False
    profiler=False

    batch_finder = False
    neptune = True
    tags = []
    description = ""
    Tm = TrainingManager(proj, conf, run_name)
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=500,
        batchsize_finder=batch_finder,
        profiler=profiler,
        neptune=neptune,
        tags=tags,
        description=description,
    )
    # Tm.D.batch_size=8
    Tm.N.compiled = compiled
# %%
    Tm.fit()
        # model(inputs)
# %%

    Tm.D.setup()
    ds = Tm.D.train_ds
    dl = Tm.D.train_dataloader()
    dl2 = Tm.D.val_dataloader()
    iteri = iter(dl)
# %%
    # img = ds.create_metatensor(img_fn)
    # label = ds.create_metatensor(label_fn)
    dici = {"image": img_fn, "label": label_fn}
    im = torch.load(img_fn)

    im.shape


# %%
    b = next(iteri)

    m = Tm.N.model
    N = Tm.N

# %%
    for x  in range(len(ds)):
        casei= ds[x]
        for a in range(len(casei)):
            print(casei[a]['image'].shape)
# %%
    for i,b in enumerate(dl):
        print(b['image'].shape )
# %%
    # b2 = next(iter(dl2))
    batch = b
    inputs, target, bbox = batch["image"], batch["label"], batch["bbox"]

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
