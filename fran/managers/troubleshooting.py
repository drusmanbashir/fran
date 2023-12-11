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

class BoringModel(nn.Module):
    def __init__(self):
         super().__init__()
         self.model= torch.nn.Linear(1572864,1)

    def forward(self,input):
        output = self.model(input)
        return [output]

class RandomDataset(Dataset):
    def __init__(self, size ):
        self.data = torch.randn( size)
        a = torch.empty(size).uniform_(0, 1)
        self.masks = torch.bernoulli(a)
        self.len=size[0]

    def __getitem__(self, index):
        img  = self.data[index]
        mask = self.masks[index]
        dici={'image':img,'label':mask ,'bbox':[0]}
        return dici

    def __len__(self):
        return self.len


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
        log_model_checkpoints=True,  # Update to True to log model checkpoints
    )
    ckpt = nl.model_checkpoint
    nl.experiment.stop()
    return ckpt


def download_neptune_checkpoint(project, run_id):
    nl = NeptuneManager(
        project=project,
        run_id=run_id,  # "LIT-46",
        log_model_checkpoints=True,  # Update to True to log model checkpoints
    )
    nl.download_checkpoints()
    ckpt = nl.model_checkpoint
    nl.experiment.stop()
    return ckpt


def get_neptune_project(project, mode):
    """
    Returns project instance based on project title
    """

    project_name, api_token = get_neptune_config(project)
    return nt.init_project(project=project_name, api_token=api_token, mode=mode)


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
        ld = plm.loss_fnc.loss_dict
        plm.logger.log_metrics(ld)


    def download_checkpoints(self):
        remote_dir =str(Path(self.model_checkpoint).parent)
        latest_ckpt = self.shadow_remote_ckpts(remote_dir)
        if latest_ckpt:
            tr()
            self.nep_run['training']['model']['best_model_path'] = latest_ckpt
            self.nep_run.wait() 

    def fix_remote_ckpt(self,ckpt):
        st_dict=torch.load(ckpt)

        dm_dict=st_dict['datamodule_hyper_parameters']
        remote_proj=dm_dict['project']
        remote_proj.predictions_folder=self.project.predictions_folder

        dm_dict['project']=remote_proj
        st_dict['datamodule_hyper_parameters']=dm_dict
        print("Fixed project predictions folder")
        torch.save(st_dict,ckpt)

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
        batch_size=8,
    ):
        super().__init__()
        self.batch_size=batch_size
        self.save_hyperparameters()
    def setup(self, stage: str = None):


        self.train_ds = RandomDataset((40,1, 128,128,96))
        self.valid_ds= RandomDataset((40,1, 128,128,96))

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

class UNetTrainer(LightningModule):
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

        self.model ,self.loss_fnc=  self.create_model()
        # self.model= torch.nn.Linear(1572864,1)

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
        loss = self.loss_fnc(self.pred, target_listed[0])
        loss = self.loss_fnc(self.pred, tt)
        target_listed[0].shape
        tt = target.squeeze(1).long()
        tt.require
        target.shape
        loss_dict = {'loss':loss.item()}
        return loss, loss_dict


    def training_step(self, batch, batch_idx):
        inputs, target, bbox = batch["image"], batch["label"], batch["bbox"]
        self.pred = self.forward(inputs)
        pred2= self.pred[0]
        loss = pred2.mean()

        self.log("train_loss", loss)
        return loss


    def log_loss(self, loss_dict, prefix):
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
        # self.logger.log_metrics(logger_dict)
        self.log(prefix + "_" + "loss_dice", loss_dict["loss_dice"], logger=True)

    def validation_step(self, batch, batch_idx):
            inputs, target, bbox = batch["image"], batch["label"], batch["bbox"]
            self.pred = self.forward(inputs)
            pred2= self.pred[0]
            loss = pred2.mean()
            self.log("val_loss", loss)
            return loss
    #
    # def validation_step(self, batch, batch_idx):
    #     loss =  batch[0].sum()
    #     self.log("valid_loss", loss)
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
        return torch.optim.SGD(self.model.parameters(), lr=0.01)
    #
    # def configure_optimizers(self):
    #     # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    #     scheduler = ReduceLROnPlateau(optimizer, "min", patience=30)
    #     output = {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "monitor": "train_loss_dice",
    #             "frequency": 2
    #             # If "monitor" references validation metrics, then "frequency" should be set to a
    #             # multiple of "trainer.check_val_every_n_epoch".
    #         },
    #     }
    #     return output

    def forward(self, inputs):
        bs = inputs.shape[0]
        inputs = inputs.view(bs,1,-1)
        outputs= self.model(inputs)
        return outputs

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
        model = BoringModel()
        loss_func= nn.CrossEntropyLoss()
        return model, loss_func


def update_nep_run_from_config(nep_run, config):
    for key, value in config.items():
        nep_run[key] = value
    return nep_run

def maybe_ddp(devices):
    if devices == 1:
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

    def setup(self,epochs=2):
        cbs = [BatchSizeFinder(mode='binsearch',init_val=8)]
        self.D =DataManager(batch_size=8)
        self.N = UNetTrainer(
            self.project,
            self.configs["dataset_params"],
            self.configs["model_params"],
            self.configs["loss_params"],
            lr=self.configs["model_params"]["lr"],
            compiled=False
        )
        logger = None

        self.D.prepare_data()
        self.trainer = Trainer(
            callbacks=cbs,
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            logger=logger,
            max_epochs=epochs,
            log_every_n_steps=2,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            # default_root_dir=self.project.checkpoints_parent_folder,
        )




    def fit(self):
        self.trainer.fit(model=self.N, datamodule=self.D)

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
    ckpt=download_neptune_checkpoint(proj,'LIT-161')

    Tm = TrainingManager(proj,conf)
# %%
    Tm.setup(epochs=5)
    Tm.D.batch_size=8
# %%
    Tm.fit()
# %%
    dl = Tm.D.val_dataloader()
    bt = next(iter(dl))
    img= bt['image']
    img.shape
    i2 = img.view(8,1,-1)
    i2.shape
# %%
    128*128*96
    mod = torch.nn.Linear(1572864,1)
    pred = mod(i2)
    loss = pred.sum()
# %%

