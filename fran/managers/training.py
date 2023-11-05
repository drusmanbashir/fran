# %%
import time
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
import torch.multiprocessing as mp
from monai.config.type_definitions import DtypeLike, NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms.intensity.array import RandGaussianNoise
from monai.transforms.spatial.array import RandFlip, Resize
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils.type_conversion import convert_to_tensor

import neptune
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropd
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import profile, record_function, ProfilerActivity
from neptune.types import File
from torchvision.utils import make_grid
from lightning.pytorch.profilers import AdvancedProfiler
import warnings
from typing import Any, Hashable, Mapping

# from fastcore.basics import GetAttr
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers.neptune import NeptuneLogger
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd
from torchvision.transforms import Compose
from monai.data import DataLoader
from monai.transforms import RandAffined
from lightning.pytorch import LightningDataModule, LightningModule, Trainer

import torch
import operator
from fran.data.dataset import ImageMaskBBoxDatasetd, MaskLabelRemap2, NormaliseClipd
from fran.transforms.spatialtransforms import one_hot
from fran.utils.helpers import folder_name_from_list
from fran.data.dataloader import img_mask_bbox_collated
import itertools as il
from fran.utils.helpers import *
from fran.utils.fileio import *
from fran.utils.imageviewers import *

from fran.utils.common import *

# from fastai.learner import *
from fran.evaluation.losses import *
from fran.architectures.create_network import (
    create_model_from_conf,
    nnUNet,
    pool_op_kernels_nnunet,
)

def compute_bs(project,config,bs=7,step=1):
        '''
        bs = starting bs
        
        '''
    
    
        print("Computing optimal batch-size for available vram")

        while True:

            Tm = TrainingManager(project,config)
            Tm.setup(batch_size = bs, epochs=1,neptune=False)

            try:
                print("Trial bs: {}".format(bs))
                Tm.fit()
            except RuntimeError:
                print("Final broken bs: {}\n-----------------".format(bs))
                bs  = bs-step*2
                print("\n----- Accepted bs: {}".format(bs))
                break
            bs+=step
            del Tm
            gc.collect()
            torch.cuda.empty_cache()
        return bs

def checkpoint_from_model_id(model_id):
    common_paths = load_yaml(common_vars_filename)
    fldr = Path(common_paths["checkpoints_parent_folder"])
    all_fldrs = [
        f for f in fldr.rglob("*{}/checkpoints".format(model_id)) if f.is_dir()
    ]
    if len(all_fldrs) == 1:
        fldr = all_fldrs[0]
    else:
        tr()

    list_of_files = list(fldr.glob("*"))
    ckpt = max(list_of_files, key=lambda p: p.stat().st_ctime)
    return ckpt

class PermuteImageMask(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys,
        prob: float = 1,
        do_transform: bool = True,
    ):
        MapTransform.__init__(self, keys, False)
        RandomizableTransform.__init__(self, prob)
        store_attr()

    def func(self,x):
        if np.random.rand() < self.p:
            img,mask=x
            sequence =(0,)+ tuple(np.random.choice([1,2],size=2,replace=False)   ) #if dim0 is different, this will make pblms
            img_permuted,mask_permuted = torch.permute(img,dims=sequence),torch.permute(mask,dims=sequence)
            return img_permuted,mask_permuted
        else: return x




class RandRandGaussianNoised(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys,
        std_limits,
        prob: float = 1,
        do_transform: bool = True,
        dtype: DtypeLike = np.float32,
    ):
        MapTransform.__init__(self, keys, False)
        RandomizableTransform.__init__(self, prob)
        store_attr("std_limits,dtype")

    def randomize(self):
        super().randomize(None)
        rand_std = self.R.uniform(low=self.std_limits[0], high=self.std_limits[1])
        self.rand_gaussian_noise = RandGaussianNoise(
            mean=0, std=rand_std, prob=1.0, dtype=self.dtype
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random noise
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.rand_gaussian_noise.randomize(d[first_key])
        for key in self.key_iterator(d):
            d[key] = self.rand_gaussian_noise(img=d[key], randomize=False)
        return d


# from fran.managers.base import *
from fran.utils.helpers import *


def get_neptune_checkpoint(project, run_id):
    nl = NeptuneManager(
        project=project,
        run_id=run_id,  # "LIT-46",
        nep_mode="read-only",
        log_model_checkpoints=True,  # Update to True to log model checkpoints
    )
    ckpt = nl.model_checkpoint
    nl.experiment.stop()
    return ckpt


def get_neptune_project(project, mode):
    """
    Returns project instance based on project title
    """

    project_name, api_token = get_neptune_config(project)
    return neptune.init_project(project=project_name, api_token=api_token, mode=mode)


def get_neptune_config(project):
    """
    Returns particular project workspace
    """
    project_title = project.project_title
    commons = load_yaml(common_vars_filename)
    project_name = "/".join([commons["neptune_workspace_name"], project_title])
    api_token = commons["neptune_api_token"]
    return project_name, api_token


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
        freq=20,
        grid_rows=6,
        imgs_per_batch=4,
        publish_deep_preds=False,
        apply_activation=True,
    ):
        if not isinstance(patch_size, torch.Size):
            patch_size = torch.Size(patch_size)
        self.stride = int(patch_size[0] / imgs_per_batch)
        store_attr()

    def on_train_start(self, trainer, pl_module):
        len_dl = int(len(trainer.train_dataloader) / trainer.accumulate_grad_batches)
        self.freq = int(len_dl / self.grid_rows)

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.grid_imgs = []
        self.grid_preds = []
        self.grid_labels = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_grid_created = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.freq == 0:
            self.populate_grid(pl_module, batch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.validation_grid_created == False:
            self.populate_grid(pl_module, batch)
            self.validation_grid_created = True

    #
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.freq == 0:
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

    def img_to_grd(self, batch):
        imgs = batch[0, :, :: self.stride, :, :].clone()
        imgs = imgs[:, : self.imgs_per_batch]
        imgs = imgs.permute(1, 0, 2, 3)  # BxCxHxW
        return imgs

    def fix_channels(self, tnsr):
        if tnsr.shape[1] == 2:
            tnsr = tnsr[:, 1:, :, :]
        if tnsr.shape[1] == 1:
            tnsr = tnsr.repeat(1, 3, 1, 1)
        return tnsr

    def populate_grid(self, pl_module, batch):
        img = batch["image"].cpu()

        label = batch["label"].cpu()
        label = label.squeeze(1)
        label = one_hot(label, self.classes, axis=1)
        pred = pl_module.pred
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = pred.cpu()

        if self.apply_activation == True:
            pred = F.softmax(pred.to(torch.float32), dim=1)

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


class NeptuneManager(NeptuneLogger, Callback):
    def __init__(
        self,
        *,
        project,
        nep_mode="async",
        run_id: Optional[str] = None,
        log_model_checkpoints: Optional[bool] = True,
        prefix: str = "training",
        **neptune_run_kwargs: Any
    ):
        store_attr("project")
        project_nep, api_token = get_neptune_config(project)
        os.environ["NEPTUNE_API_TOKEN"] = api_token
        os.environ["NEPTUNE_PROJECT"] = project_nep
        self.df = self.fetch_project_df()
        if run_id:
            nep_run = self.load_run(run_id, nep_mode)
            project_nep, api_token = None, None
        else:
            nep_run = None

        NeptuneLogger.__init__(
            self,
            api_key=api_token,
            project=project_nep,
            run=nep_run,
            log_model_checkpoints=log_model_checkpoints,
            prefix=prefix,
            **neptune_run_kwargs
        )

    @property
    def model_checkpoint(self):
        try:
            ckpt = self.experiment["training/model/best_model_path"].fetch()
            return ckpt
        except:
            print("No checkpoints in this run")

    def fetch_project_df(self, columns=None):
        print("Downloading runs history as dataframe")
        project_tmp = get_neptune_project(self.project, "read-only")
        df = project_tmp.fetch_runs_table(columns=columns).to_pandas()
        return df

    def load_run(
        self,
        run_name,
        nep_mode="async",
    ):
        """

        :param run_name:
            If a legit name is passed it will be loaded.
            If an illegal run-name is passed, throws an exception
            If most_recent is passed, most recent run  is loaded.

        :param update_nep_run_from_config: This is a dictionary which can be uploaded on Neptune to alter the parameters of the existing model and track new parameters
        """
        run_id, msg = self.get_run_id(run_name)
        print("{}. Loading".format(msg))
        nep_run = neptune.init_run(with_id=run_id, mode=nep_mode)
        return nep_run

    def get_run_id(self, run_id):
        if run_id == "most_recent":
            run_id = self.id_most_recent()
            msg = "Most recent run"
        elif run_id is any(["", None]):
            raise Exception(
                "Illegal run name: {}. No ids exist with this name".format(run_id)
            )

        else:
            self.id_exists(run_id)
            msg = "Run id matching {}".format(run_id)
        return run_id, msg

    def id_exists(self, run_id):
        row = self.df.loc[self.df["sys/id"] == run_id]
        try:
            print("Existing Run found. Run id {}".format(row["sys/id"].item()))
            return row["sys/id"].item()
        except Exception as e:
            print("No run with that name exists .. {}".format(e))

    def id_most_recent(self):
        self.df = self.df.sort_values(by="sys/creation_time", ascending=False)
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            if self._has_checkpoints(row):
                print("Loading most recent run. Run id {}".format(row["sys/id"]))
                return row["sys/id"], row["metadata/run_name"]

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_batch_end(self, trn, plm, outputs, batch, batch_idx):
        ld = plm.loss_fnc.loss_dict
        plm.logger.log_metrics(ld)

    @property
    def run_id(self):
        return self.experiment["sys/id"].fetch()

    #
    @property
    def save_dir(self) -> Optional[str]:
        sd = self.project.checkpoints_parent_folder
        return str(sd)


#
#
#
class NepImages(Callback):
    def __init__(self, freq):
        store_attr()

    def on_train_start(self, trainer, pl_module):
        pass

    def populate_grid(self):
        for batch, category, grd in zip(
            [self.learn.x, self.learn.pred, self.learn.y],
            ["imgs", "preds", "masks"],
            [self.grid_imgs, self.grid_preds, self.grid_masks],
        ):
            if isinstance(batch, (list, tuple)) and self.publish_deep_preds == False:
                batch = [x for x in batch if x.size()[2:] == self.patch_size][
                    0
                ]  # gets that pred which has same shape as imgs
            elif isinstance(batch, (list, tuple)) and self.publish_deep_preds == True:
                batch_tmp = [
                    F.interpolate(b, size=batch[-1].shape[2:], mode="trilinear")
                    for b in batch[:-1]
                ]
                batch = batch_tmp + batch[-1]
            batch = batch.cpu()

            if self.apply_activation == True and category == "preds":
                batch = F.softmax(batch, dim=1)

            imgs = self.img_to_grd(batch)
            if category == "masks":
                imgs = imgs.squeeze(1)
                imgs = one_hot(imgs, self.classes, axis=1)
            if category != "imgs" and imgs.shape[1] != 3:
                imgs = imgs[:, 1:, :, :]
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)

            grd.append(imgs)


class DataManager(LightningDataModule):
    def __init__(
        self,
        project,
        dataset_params: dict,

        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
    ):
        """ """
        super().__init__()
        self.save_hyperparameters()
        store_attr(but="transform_factors")
        global_properties = load_dict(project.global_properties_filename)
        self.dataset_params["intensity_clip_range"] = global_properties[
            "intensity_clip_range"
        ]
        self.dataset_params["mean_fg"] = global_properties["mean_fg"]
        self.dataset_params["std_fg"] = global_properties["std_fg"]
        self.batch_size = batch_size
        self.assimilate_tfm_factors(transform_factors)

    #
    # def state_dict(self):
    #     state={'batch_size':'j'}
    #     return state
    #     # return self.dataset_params
    #
    # def load_state_dict(self,state_dict):
    #     self.batch_size= state_dict['batch_size']
    # #
    def assimilate_tfm_factors(self, transform_factors):
        for key, value in transform_factors.items():
            dici = {"value": value[0], "prob": value[1]}
            setattr(self, key, dici)

    def prepare_data(self):
        # getting the right folders
        prefixes, value_lists = ["spc", "dim"], [
            self.dataset_params["spacings"],
            self.dataset_params["src_dims"],
        ]
        if bool(self.dataset_params["patch_based"]) == True:
            parent_folder = self.project.patches_folder
        else:
            parent_folder = self.project.whole_images_folder
            for listi in prefixes, value_lists:
                del listi[0]

        for prefix, value_list in zip(prefixes, value_lists):
            parent_folder = folder_name_from_list(prefix, parent_folder, value_list)
        self.dataset_folder = parent_folder
        assert self.dataset_folder.exists(), "Dataset folder {} does not exists".format(
            self.dataset_folder
        )

    def create_transforms(self):
        all_after_item = [
            MaskLabelRemap2(
                keys=["label"], src_dest_labels=self.dataset_params["src_dest_labels"]
            ),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            NormaliseClipd(
                keys=["image"],
                clip_range=self.dataset_params["intensity_clip_range"],
                mean=self.dataset_params["mean_fg"],
                std=self.dataset_params["std_fg"],
            ),
        ]

        t2 = [
            # EnsureTyped(keys=["image", "label"], device="cuda", track_meta=False),
            RandFlipd(keys=["image", "label"], prob=self.flip["prob"], spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=self.flip["prob"], spatial_axis=1),
            RandScaleIntensityd(
                keys="image", factors=self.scale["value"], prob=self.scale["prob"]
            ),
            RandRandGaussianNoised(
                keys=["image"], std_limits=self.noise["value"], prob=self.noise["prob"]
            ),
            # RandGaussianNoised(
            #     keys=["image"], std=self.noise["value"], prob=self.noise["prob"]
            # ),
            RandShiftIntensityd(
                keys="image", offsets=self.shift["value"], prob=self.shift["prob"]
            ),
            RandAdjustContrastd(
                ["image"], gamma=self.contrast["value"], prob=self.contrast["prob"]
            ),
            self.create_affine_tfm(),
        ]
        t3 = [
            ResizeWithPadOrCropd(
                keys=["image", "label"],
                source_key="image",
                spatial_size=self.dataset_params["patch_size"],
            )
        ]
        self.tfms_train = Compose(all_after_item + t2 + t3)
        self.tfms_valid = Compose(all_after_item + t3)

    def create_affine_tfm(self):
        affine = RandAffined(
            keys=["image", "label"],
            mode=["bilinear", "nearest"],
            prob=self.affine3d["p"],
            # spatial_size=self.dataset_params['src_dims'],
            rotate_range=self.affine3d["rotate_range"],
            scale_range=self.affine3d["scale_range"],
        )
        return affine

    def setup(self, stage: str = None):
        self.train_list, self.valid_list = self.project.get_train_val_files(
            self.dataset_params["fold"]
        )
        self.create_transforms()
        bboxes_fname = self.dataset_folder / "bboxes_info"
        self.train_ds = ImageMaskBBoxDatasetd(
            self.train_list,
            bboxes_fname,
            self.dataset_params["class_ratios"],
            transform=self.tfms_train,
        )
        self.valid_ds = ImageMaskBBoxDatasetd(
            self.valid_list, bboxes_fname, transform=self.tfms_valid
        )

    def train_dataloader(self, num_workers=24, **kwargs):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=num_workers,
            collate_fn=img_mask_bbox_collated,
            persistent_workers=True,
            pin_memory=True,
        )
        return train_dl

    def val_dataloader(self, num_workers=24, **kwargs):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=num_workers,
            collate_fn=img_mask_bbox_collated,
            persistent_workers=True,
            pin_memory=True,
        )
        return valid_dl

    def forward(self, inputs, target):
        return self.model(inputs)


class nnUNetTrainer(LightningModule):
    def __init__(
        self,
        project,
        dataset_params,
        model_params,
        loss_params,
        lr=None,
        compiled=False,
    ):
        super().__init__()
        self.lr = lr if lr else model_params['lr']
        store_attr()
        self.save_hyperparameters("model_params", "loss_params")
        self.model, self.loss_fnc = self.create_model()

    def _calc_loss(self, batch):
        inputs, target, bbox = batch["image"], batch["label"], batch["bbox"]
        self.pred = self.forward(inputs)
        target_listed = []
        for s in self.deep_supervision_scales:
            if all([i == 1 for i in s]):
                target_listed.append(target)
            else:
                size = [int(np.round(ss * aa)) for ss, aa in zip(s, target.shape[2:])]
                target_downsampled = F.interpolate(target, size=size, mode="nearest")
                target_listed.append(target_downsampled)
        loss = self.loss_fnc(self.pred, target_listed)
        loss_dict = self.loss_fnc.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self._calc_loss(batch)
        self.log_metrics(loss_dict, prefix="train")
        return loss

    def log_metrics(self, loss_dict, prefix):
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
        self.logger.log_metrics(logger_dict)
        self.log(prefix + "_" + "loss_dice", loss_dict["loss_dice"], logger=True)

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self._calc_loss(batch)
        self.log_metrics(loss_dict, prefix="val")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img = batch["image"]
        outputs = self.forward(img)
        outputs2 = outputs[0]
        batch["pred"] = outputs2
        # output=outputs[0]
        # outputs = {'pred':output,'org_size':batch['org_size']}
        # outputs_backsampled=self.post_process(outputs)
        return batch

    def post_process(self, pred_output):
        preds_bs = []
        pred = pred_output["pred"]
        sizes = pred_output["org_size"]
        for p, s in zip(pred, sizes):
            R = Resize(s)
            pred_bs = R(p)
            # maxed = torch.argmax(pred_bs, 1, keepdim=False)
            preds_bs.append(pred_bs)
        return preds_bs

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
        # self.device = device
        # creates learner from configs. Loads checkpoint if any exists in self.checkpoints_folder

        model = create_model_from_conf(self.model_params, self.dataset_params)
        # if self.checkpoints_folder:
        #     load_checkpoint(self.checkpoints_folder, model)
        if (
            self.model_params["arch"] == "DynUNet"
            or self.model_params["arch"] == "nnUNet"
        ):
            if self.model_params["arch"] == "DynUNet":
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
                fg_classes=self.model_params["out_channels"] - 1,
            )
            # cbs += [DownsampleMaskForDS(self.deep_supervision_scales)]

        else:
            loss_func = CombinedLoss(
                **self.loss_params,
                fg_classes=self.model_params["out_channels"] - 1
            )
        # if distributed==True:
        #     cbs+=[DistributedTrainer]
        if self.compiled == True:
            model = torch.compile(model)
        return model, loss_func


def update_nep_run_from_config(nep_run, config):
    for key, value in config.items():
        nep_run[key] = value
    return nep_run

def maybe_ddp(devices):
    if devices == 1:
        return 'auto'
    if 'get_ipython' in globals():
        return 'ddp_notebook'
    else:
        return 'ddp'



class TrainingManager:
    def __init__(self, project, configs):
        store_attr()

    def setup(self,batch_size, run_name=None, cbs=None, devices=1, neptune=True,epochs=500):
        self.ckp = None if run_name is None else checkpoint_from_model_id(run_name)
        strategy= maybe_ddp(devices)

        if self.ckp:
            self.D = DataManager.load_from_checkpoint(self.ckp)
            self.N = nnUNetTrainer.load_from_checkpoint(
                self.ckp, project=self.project, dataset_params=self.D.dataset_params
            )
        else:
            self.N = nnUNetTrainer(
                self.project,
                self.configs["dataset_params"],
                self.configs["model_params"],
                self.configs["loss_params"],
                lr=self.configs["model_params"]["lr"],
            )
            self.D = DataManager(
                self.project,
                dataset_params=self.configs["dataset_params"],
                transform_factors=self.configs["transform_factors"],
                affine3d=self.configs["affine3d"],
                batch_size=batch_size
            )

        self.D.prepare_data()
        if neptune == True:
            logger = NeptuneManager(
                project=self.project,
                run_id=run_name,
                log_model_checkpoints=True,  # Update to True to log model checkpoints
            )

        else:
            logger = None

        self.trainer = Trainer(
            callbacks=cbs,
            accelerator="gpu",
            devices=devices,
            precision="16-mixed",
            logger=logger,
            max_epochs=epochs,
            log_every_n_steps=10,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            default_root_dir=self.project.checkpoints_parent_folder,
            strategy=strategy
            # strategy='ddp_find_unused_parameters_true'
        )

    def fit(self):
        self.trainer.fit(model=self.N, datamodule=self.D, ckpt_path=self.ckp)


# %%


if __name__ == "__main__":
    # warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *

    project_title = "lits32"
    proj = Project(project_title=project_title)

    conf = ConfigMaker(
        proj,
        raytune=False,
        configuration_filename="/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx",
    ).config
    # configs = ConfigMaker(project, raytune=False).config

    global_props = load_dict(proj.global_properties_filename)
# %%
    Tm = TrainingManager(proj,conf)
    Tm.setup(epochs=1)
    Tm.fit()
# %%

    # %%

    ds = D.train_ds
    a = ds[0]
    im = a["image"]
    im2 = RandFlip(prob=1, spatial_axis=0)(im)
    im3 = RandFlip(prob=1, spatial_axis=1)(im)
    im4 = RandFlip(prob=1, spatial_axis=0)(im3)
    ImageMaskViewer([im[0], im2[0]])
    # %%
    trainer.fit(model=N, datamodule=D, ckpt_path=ckp)
#     # trainer.fit(model = N,datamodule=D,ckpt_path=cpk)
#     # trainer.fit(model = N,train_dataloaders=D.train_dataloader(),val_dataloaders=D.val_dataloader(),ckpt_path='/home/ub/code/fran/fran/.neptune/Untitled/LITS-567/checkpoints/epoch=53-step=2484.ckpt')
#     # trainer.fit(model = N,train_dataloaders=D.train_dataloader(),val_dataloaders=D.val_dataloader())
# %%
    bs  =compute_bs(proj,conf,1,bs=6,step=1)
# %%

