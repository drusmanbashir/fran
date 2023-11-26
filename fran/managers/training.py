# %%
from neptune.exceptions import FileNotFound
import shutil
from fran.utils.batch_size_scaling import _scale_batch_size2, _reset_dataloaders
from paramiko import SSHClient
from copy import deepcopy
import time
from lightning.pytorch.utilities.exceptions import MisconfigurationException, _TunerExitException
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
import torch.multiprocessing as mp
from monai.config.type_definitions import DtypeLike, NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms.intensity.array import RandGaussianNoise
from monai.transforms.spatial.array import RandFlip, Resize
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils.type_conversion import convert_to_tensor

import neptune as nt
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
from typing import Any, Dict, Hashable, Mapping

# from fastcore.basics import GenttAttr
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
from fran.transforms.totensor import ToTensorT
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



try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass

import torch
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

def fix_dict_keys(input_dict, old_string,new_string):
            output_dict = {}
            for key in input_dict.keys():
                neo_key = key.replace(old_string,new_string)
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
        print("no local files. Model may be on remote path. use download_neptune_checkpoint() ")
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
        log_model_checkpoints=False,  # Update to True to log model checkpoints
    )
    ckpt = nl.model_checkpoint
    nl.experiment.stop()
    return ckpt


def download_neptune_checkpoint(project, run_id):
    nl = NeptuneManager(
        project=project,
        run_id=run_id,  # "LIT-46",
        log_model_checkpoints=False,  # Update to True to log model checkpoints
    )
    nl.download_checkpoints()
    ckpt = nl.model_checkpoint
    nl.experiment.stop()
    return ckpt


def get_neptune_project(project, mode):
    """
    Returns project instance based on project title
    """

    project_name, api_token = get_neptune_config()
    return nt.init_project(project=project_name, api_token=api_token, mode=mode)


def get_neptune_config():
    """
    Returns particular project workspace
    """
    commons = load_yaml(common_vars_filename)
    project_name = commons['neptune_project']
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
        grid_rows=6,
        imgs_per_batch=4,
        publish_deep_preds=False,
        apply_activation=True,
        epoch_freq=2 # skip how many epochs.
    ):
        if not isinstance(patch_size, torch.Size):
            patch_size = torch.Size(patch_size)
        self.stride = int(patch_size[0] / imgs_per_batch)
        store_attr()
    #
    def on_train_start(self, trainer, pl_module):
        len_dl = int(len(trainer.train_dataloader) / trainer.accumulate_grad_batches)
        self.freq = int(len_dl / self.grid_rows)

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.epoch_freq == 0:
            super().on_train_epoch_start(trainer, pl_module)
            self.grid_imgs = []
            self.grid_preds = []
            self.grid_labels = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_grid_created = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.epoch_freq == 0:
            if trainer.global_step % self.freq == 0:
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

    def img_to_grd(self, batch):
        # imgs = batch[0, :, :: self.stride, :, :].clone()
        # imgs = imgs[:, : self.imgs_per_batch]
        imgs = batch[self.batches, :, :,:, self.slices].clone()
        # imgs = imgs.permute(1, 0, 2, 3)  # BxCxHxW
        # tr()
        return imgs

    def fix_channels(self, tnsr):
        if tnsr.shape[1] == 2:
            tnsr = tnsr[:, 1:, :, :]
        if tnsr.shape[1] == 1:
            tnsr = tnsr.repeat(1, 3, 1, 1)
        return tnsr

    def populate_grid(self, pl_module, batch):
        def _randomize():
            n_slices= img.shape[-1]
            batch_size=img.shape[0]
            self.slices = [random.randrange(0,n_slices) for i in range(self.imgs_per_batch)]
            self.batches=[random.randrange(0,batch_size) for i in range(self.imgs_per_batch)]



        img = batch["image"].cpu()

        label = batch["label"].cpu()
        label = label.squeeze(1)
        label = one_hot(label, self.classes, axis=1)
        pred = pl_module.pred
        if isinstance(pred, Union[list,tuple]):
            pred = pred[0]
        pred = pred.cpu()

        if self.apply_activation == True:
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


class NeptuneManager(NeptuneLogger, Callback):
    def __init__(
        self,
        *,
        project,
        nep_mode="async",
        run_id: Optional[str] = None,
        log_model_checkpoints: Optional[bool] = False,
        prefix: str = "training",
        **neptune_run_kwargs: Any
    ):
        store_attr("project")
        project_nep, api_token = get_neptune_config()
        os.environ["NEPTUNE_API_TOKEN"] = api_token
        os.environ["NEPTUNE_PROJECT"] = project_nep
        self.df = self.fetch_project_df()
        if run_id:
            nep_run = self.load_run(run_id, nep_mode)
            project_nep, api_token = None, None
            neptune_run_kwargs={}
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
    def nep_run(self): return self.experiment

    @property
    def model_checkpoint(self):
        try:
            ckpt = self.experiment["training/model/best_model_path"].fetch()
            return ckpt
        except:
            print("No checkpoints in this run")

    @model_checkpoint.setter
    def model_checkpoint(self,value):
        self.experiment["training/model/best_model_path"]= value
        self.experiment.wait()

    def fetch_project_df(self, columns=None):
        print("Downloading runs history as dataframe")
        project_tmp = get_neptune_project(self.project, "read-only")
        df = project_tmp.fetch_runs_table(columns=columns).to_pandas()
        return df

    def on_fit_start(self):
        self.experiment["sys/name"]=self.project.project_title
        self.experiment.wait()

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
        nep_run = nt.init_run(with_id=run_id, mode=nep_mode)
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
        tr()
        ld = plm.loss_fnc.loss_dict
        plm.logger.log_metrics(ld)


    def download_checkpoints(self):
        remote_dir =str(Path(self.model_checkpoint).parent)
        latest_ckpt = self.shadow_remote_ckpts(remote_dir)
        if latest_ckpt:
            self.nep_run['training']['model']['best_model_path'] = latest_ckpt
            self.nep_run.wait() 

    def shadow_remote_ckpts(self, remote_dir):
        hpc_settings = load_yaml(hpc_settings_fn)
        local_dir = self.project.checkpoints_parent_folder /("Untitled")/ self.run_id/("checkpoints")
        print("\nSSH to remote folder {}".format(remote_dir))
        client = SSHClient()
        client.load_system_host_keys()
        client.connect(
            hpc_settings["host"],
            username=hpc_settings["username"],
            password=hpc_settings["password"],
        )
        ftp_client = client.open_sftp()
        try:
            fnames = []
            for f in sorted(ftp_client.listdir_attr(remote_dir), key=lambda k: k.st_mtime, reverse=True):
                fnames.append(f.filename)
        except FileNotFoundError as e:
            print("\n------------------------------------------------------------------")
            print("Error:Could not find {}.\nIs this a remote folder and exists?\n".format(remote_dir))
            return
        remote_fnames = [os.path.join(remote_dir, f) for f in fnames]
        local_fnames = [os.path.join(local_dir, f) for f in fnames]
        maybe_makedirs(local_dir)
        for rem, loc in zip(remote_fnames, local_fnames):
            if Path(loc).exists():
                print("Local file {} exists already.".format(loc))
            else:
                print("Copying file {0} to local folder {1}".format(rem, local_dir))
                ftp_client.get(rem, loc)
        latest_ckpt = local_fnames[0]
        return latest_ckpt

    def stop(self): self.experiment.stop()




    @property
    def run_id(self):
        return self.experiment["sys/id"].fetch()

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
        self.train_list, self.valid_list = self.project.get_train_val_files(
            self.dataset_params["fold"]
        )
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

class DataManagerShort(DataManager):
    def prepare_data(self):
        super().prepare_data()
        self.train_list= self.train_list[:32]

    def train_dataloader(self, num_workers=4, **kwargs): return super().train_dataloader(num_workers,**kwargs)

class UNetTrainer(LightningModule):
    def __init__(
        self,
        project,
        dataset_params,
        model_params,

        loss_params,
        max_epochs=1000,
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

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self._calc_loss(batch)
        self.log_metrics(loss_dict, prefix="val")
        return loss

    def on_fit_start(self):
        self.logger.experiment["sys/name"]=self.project.project_title
        self.logger.experiment.wait()

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


    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     img = batch["image"]
    #     outputs = self.forward(img)
    #     tr()
    #     outputs2 = outputs[0]
    #     batch["pred"] = outputs2
        # output=outputs[0]
        # outputs = {'pred':output,'org_size':batch['org_size']}
        # outputs_backsampled=self.post_process(outputs)
        # return batch

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
        # optimizer = torch.optim.SGD(
        #         self.model.parameters(),
        #         lr=self.lr,
        #         momentum=0.99,
        #         weight_decay=3e-5,
        #         nesterov=True,
        #     )
        #
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / self.max_epochs) ** 0.9)
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
        if self.compiled == True:
            model = torch.compile(model)
        return model, loss_func


def update_nep_run_from_config(nep_run, config):
    for key, value in config.items():
        nep_run[key] = value
    return nep_run

def maybe_ddp(devices):
    if devices == 1 or isinstance(devices,Union[list,str,tuple]):
        return 'auto'
    ip = get_ipython()
    if ip:
        print ("Using interactive-shell ddp strategy")
        return 'ddp_notebook'
    else:
        print ("Using non-interactive shell ddp strategy")
        return 'ddp'

class TrainingManager():
    def __init__(self, project, configs):
        super().__init__()
        store_attr()

    def setup(self,batch_size, run_name=None, cbs=[], devices=1,compiled=False, neptune=True,tags=[],description="",epochs=1000,batch_finder=False):
        self.ckpt = None if run_name is None else checkpoint_from_model_id(run_name)
        strategy= maybe_ddp(devices)

        if self.ckpt: self.load_ckpts()
        else:
            if batch_finder==True:
                DMclass = DataManagerShort
            else: DMclass = DataManager
            self.D = DMclass(
                self.project,
                dataset_params=self.configs["dataset_params"],
                transform_factors=self.configs["transform_factors"],
                affine3d=self.configs["affine3d"],
                batch_size=batch_size
            )
            self.N = UNetTrainer(
                self.project,
                self.configs["dataset_params"],
                self.configs["model_params"],
                self.configs["loss_params"],
                lr=self.configs["model_params"]["lr"],
                compiled=compiled,
                max_epochs=epochs
            )
        if neptune == True:
            logger = NeptuneManager(
                project=self.project,
                run_id=run_name,
                log_model_checkpoints=False,  # Update to True to log model checkpoints
                tags=tags,
                description=description
            )
            N= NeptuneImageGridCallback(
                    classes=self.configs['model_params']['out_channels'], patch_size=self.configs["dataset_params"]["patch_size"])

            cbs+=[N,
            TQDMProgressBar(refresh_rate=3)
              ]

        else:
            logger = None

        self.D.prepare_data()
        self.trainer = Trainer(
            callbacks=cbs,
            accelerator="gpu",
            devices=devices,
            precision="16-mixed",
            logger=logger,
            max_epochs=epochs,
            log_every_n_steps=25,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            default_root_dir=self.project.checkpoints_parent_folder,
            strategy=strategy
            # strategy='ddp_find_unused_parameters_true'
        )


    def load_ckpts(self):
            self.D = DataManager.load_from_checkpoint(self.ckpt,project=self.project)
            state_dict=torch.load(self.ckpt)
            lr=state_dict['lr_schedulers'][0]['_last_lr'][0]

            try:
                self.N = UNetTrainer.load_from_checkpoint(
                    self.ckpt, project=self.project, dataset_params=self.D.dataset_params,lr=lr
                )
            except:
                ckpt_state = state_dict['state_dict']
                ckpt_state_updated = fix_dict_keys(ckpt_state,'model','model._orig_mod')
                print(ckpt_state_updated.keys())
                state_dict_neo = state_dict.copy()
                state_dict_neo['state_dict']=ckpt_state_updated

                ckpt_old = self.ckpt.str_replace('_bkp','')
                ckpt_old = self.ckpt.str_replace('.ckpt','.ckpt_bkp')
                torch.save(state_dict_neo,self.ckpt)

                shutil.move(self.ckpt,ckpt_old)

                self.N = UNetTrainer.load_from_checkpoint(
                    self.ckpt, project=self.project, dataset_params=self.D.dataset_params,lr=lr
                )


    def fit(self):
        self.trainer.fit(model=self.N, datamodule=self.D, ckpt_path=self.ckpt)


    def fix_state_dict_keys(self,bad_str="model",good_str="model._orig_mod"):
        state_dict = torch.load(self.ckpt)
        ckpt_state = state_dict['state_dict']
        ckpt_state_updated = fix_dict_keys(ckpt_state,bad_str,good_str)
        state_dict_neo = state_dict.copy()
        state_dict_neo['state_dict']=ckpt_state_updated

        ckpt_old = self.ckpt.str_replace('.ckpt','.ckpt_bkp')
        shutil.move(self.ckpt,ckpt_old)

        torch.save(state_dict_neo,self.ckpt)
        return ckpt_old



# %%


if __name__ == "__main__":
    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *

    project_title = "lits32"
    proj = Project(project_title=project_title)

    configuration_filename="/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    configuration_filename=None

    conf = ConfigMaker(
        proj,
        raytune=False,
        configuration_filename=configuration_filename
    ).config

    global_props = load_dict(proj.global_properties_filename)
# %%
    conf['model_params']['arch']='nnUNet'
    conf['model_params']['lr']=1e-3

    Tm = TrainingManager(proj,conf)
# %%
    bs = 8
    run_name ='LIT-153'
    run_name =None
    compiled=False
    batch_finder=False
    neptune=True
    tags=[]
    description="Baseline all transforms as in previous full data runs"
    Tm.setup(run_name=run_name,compiled=False,batch_size=bs,devices = [1], epochs=400,batch_finder=batch_finder,neptune=neptune,tags=tags,description=description)
    # Tm.D.batch_size=8
    # Tm.N.compiled=compiled
# %%
    Tm.fit()
# %%
# %%

    Tm.D.setup()
    dl=Tm.D.train_dataloader()
    dl2=Tm.D.val_dataloader()
    iteri=iter(dl)
# %%
    m = Tm.N.model
    N=  Tm.N

    
# %%
    b=next(iteri)
    b2=next(iter(dl2))
    batch= b2
    inputs, target, bbox = batch["image"], batch["label"], batch["bbox"]
    
    [pp(a['filename']) for a in bbox]
# %%
    preds = N.model(inputs.cuda())
    pred = preds[0]
    pred=pred.detach().cpu()
    pp(pred.shape)
# %%
    n=4
    img = inputs[n,0]
    mask = target[n,0]
    pre = pred[n,2]
# %%
    ImageMaskViewer([img.permute(2,1,0),mask.permute(2,1,0)])
# %%
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacings/litsmall/spc_080_080_150/images/lits_4.pt"
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacings/litstp/spc_080_080_150/images/lits_4.pt"
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacings/lits32/spc_080_080_150/images/lits_4.pt"
    fn="/home/ub/datasets/preprocessed/lits32/patches/spc_080_080_150/dim_192_192_128/images/lits_4_1.pt"
    fn2="/home/ub/datasets/preprocessed/lits32/patches/spc_080_080_150/dim_192_192_128/masks/lits_4_1.pt"
    img=torch.load(fn)
    mask=torch.load(fn2)
    pp(img.shape)
# %%

    ImageMaskViewer([img,mask])
# %%

# %%
