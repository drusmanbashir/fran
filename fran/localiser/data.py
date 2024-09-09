# %%
import lightning as L
from monai.transforms.io.array import SaveImage
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.transforms import Compose
from fran.data.collate import as_is_collated
from fran.localizer.helpers import draw_image_bbox, draw_image_lm_bbox
from fran.transforms.imageio import LoadTorchd, TorchWriter
import SimpleITK as sitk
import itertools as il
from monai.apps.detection.transforms.dictionary import (
    ClipBoxToImaged,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    RandCropBoxByPosNegLabeld,
    RandFlipBoxd,
    RandRotateBox90d,
    RandZoomBoxd,
    StandardizeEmptyBoxd,
)
from monai.transforms.croppad.dictionary import BoundingRectd, RandCropByPosNegLabeld
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.utility.dictionary import (
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    RepeatChanneld,
)
from monai.transforms.utils import generate_spatial_bounding_box
import matplotlib.patches as patches
import torch.nn.functional as F
from pathlib import Path
from monai.transforms.intensity.array import NormalizeIntensity, ScaleIntensity
from monai.transforms.intensity.dictionary import (
    NormalizeIntensityD,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.transforms.spatial.dictionary import RandFlipd, RandRotated, RandZoomd, Resized, Resize
from torchvision.datasets.folder import is_image_file

from fran.transforms.imageio import LoadSITKd
from fran.transforms.intensitytransforms import MakeBinary, RandRandGaussianNoised
from fran.transforms.misc_transforms import BoundingBoxYOLOd, DictToMeta, MetaToDict
from fran.transforms.spatialtransforms import Project2D
from fran.utils.helpers import match_filename_with_case_id, pbar
import shutil, os
import h5py
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from fran.utils.fileio import is_sitk_file, load_dict, maybe_makedirs
from fran.utils.helpers import find_matching_fn
import ipdb

tr = ipdb.set_trace

from fran.utils.imageviewers import ImageMaskViewer, view_sitk
from fran.utils.string import info_from_filename
from monai.visualize import *
import matplotlib.pyplot as plt
from monai.data import dataset_summary, register_writer
import pandas as pd
import numpy as np


from torch.utils.data import DataLoader, random_split
from monai.data import Dataset

from fran.utils.fileio import is_sitk_file
from fran.utils.helpers import find_matching_fn


class DetectDS(Dataset):
    def __init__(self, fldr):
        fldr = Path(fldr)
        self.fldr_imgs = fldr / "images"
        self.fldr_lms = fldr / "lms"
        imgs = list(self.fldr_imgs.glob("*"))
        lms = list(self.fldr_lms.glob("*"))
        self.data_dicts = []
        for img in imgs:
            lm = [fn for fn in lms if fn.name == img.name]
            if len(lm) == 1:
                lm = lm[0]
            else:
                raise FileNotFoundError(f"Cannot find lm for {img}")
            dici = {"image": img, "lm": lm}
            self.data_dicts.append(dici)

    def __len__(self) -> int:
        return len(self.data_dicts)




class DetectDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.fldr_imgs = self.data_dir / "images"
        self.fldr_lms = self.data_dir / "lms"

    def prepare_data(self):
        imgs = list(self.fldr_imgs.glob("*"))
        tot = len(imgs)
        train_len = int(0.8 * tot)
        val_len = tot - train_len
        imgs_train, imgs_val = random_split(
            imgs, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

        self.train_dicts = self.create_data_dicts(imgs_train)
        self.val_dicts = self.create_data_dicts(imgs_val)

    def create_data_dicts(self, imgs):
        lms = list(self.fldr_lms.glob("*"))
        data_dicts = []
        for img in imgs:
            lm = [fn for fn in lms if fn.name == img.name]
            if len(lm) == 1:
                lm = lm[0]
            else:
                raise FileNotFoundError(f"Cannot find lm for {img}")
            dici = {"image": img, "lm": lm}
            data_dicts.append(dici)
        return data_dicts

    def create_transforms(self, probs=.3, probs_intensity=.3):
        image_key = "image"
        label_key = "lm"
        box_key = "lm_bbox"
        probs= probs
        outputsize = [512,512]
    
        probs_int =probs_intensity

        L = LoadTorchd([image_key, label_key])
        E = EnsureChannelFirstd(keys=[image_key])

        MkB = MakeBinary([label_key])
        Et = EnsureTyped(keys=[image_key ], dtype=torch.float32)
        Et2 = EnsureTyped(keys=[label_key], dtype=torch.long)
        E2 = EnsureTyped(keys=[image_key], dtype=torch.float16)
        ExtractBbox = BoundingRectd(keys=[label_key])
        CB = ConvertBoxToStandardModed(mode="xxyy", box_keys=[box_key])
        YoloBbox = BoundingBoxYOLOd(
            [box_key], 2, key_template_tensor=label_key, output_keys=["bbox_yolo"]
        )

        Rotate = RandRotated(
            keys=[image_key, label_key],
            prob=probs,
            keep_size=True,
            mode=["bilinear", "nearest"],
            range_x=[0.4, 0.4],
            lazy=True
        )
        Zoom = RandZoomd(
            keys=[image_key, label_key],
            mode = ['bilinear','nearest'],
            prob=probs,
            min_zoom=0.7,
            max_zoom=1.4,
            padding_mode="constant",
            keep_size=True,
            lazy=True
        )
        Flip1 = RandFlipd(
            keys=[image_key,label_key],
            prob=probs,
            spatial_axis=0,
            lazy=True
        )

        Flip2 = RandFlipd(
            keys=[image_key,label_key],
            prob=probs,
            spatial_axis=1,
            lazy=True
        )
        Resize = Resized(
            keys=[image_key,label_key],
            spatial_size = outputsize,
            mode= ['bilinear','nearest'],
            lazy=True
            
        )

        N = NormalizeIntensityd(keys=[image_key])
        int_augs = [
            RandGaussianSmoothd(
                keys=[image_key],
                prob=probs_int,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys="image", factors=0.25, prob=probs_int),
            RandGaussianNoised(keys=["image"], mean=0, prob=probs_int, std=0.1),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=probs_int),
            RandAdjustContrastd(["image"], gamma=(0.7, 1.5)),
        ]
        Rp = RepeatChanneld(keys = [image_key],repeats=3)
        DelI = DeleteItemsd(keys = [label_key])
        self.transforms_dict = {
            "N": N,
            "L": L,
            "E": E,
            "Et": Et,
            "MkB": MkB,
            "Et2": Et2,
            "E2": E2,
            "ExtractBbox": ExtractBbox,
            "YoloBbox": YoloBbox,
            "DelI": DelI,
            "CB": CB,
            "Rotate": Rotate,
            "Zoom": Zoom,
            "IntensityTfms": int_augs,
            "Flip1": Flip1,
            "Flip2": Flip2,
            "Resize":Resize,
            "Rp":Rp,
            # "RotBbox": RotBbox,
            # "ClipoBbox": ClipoBbox,
        }

    def set_transforms(self, keys_tr: str, keys_val: str):
        self.tfms_train = self.tfms_from_dict(keys_tr)
        self.tfms_valid = self.tfms_from_dict(keys_val)

    def tfms_from_dict(self, keys: str):
        keys = keys.replace(" ", "").split(",")
        tfms = []
        for key in keys:
            tfm = self.transforms_dict[key]
            if key == "IntensityTfms":
                tfms.extend(tfm)
            else:
                tfms.append(tfm)
        tfms = Compose(tfms)
        return tfms

        tfms_train = Compose([L, MkB, Rotate, ExtractBbox, CB, Et, Et2, Z, Flip1, Flip2, RotBbox, ClipoBbox, YoloBbox])
        tfms_val = Compose([L, MkB, ExtractBbox, CB, Et, Et2, ClipoBbox, YoloBbox])

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.create_transforms()
        self.set_transforms(
            # keys_tr="L,MkB,  Et, Et2, N,I, Flip1, Flip2, Rotate,Z, Resize, ExtractBbox, CB,YoloBbox, DelI,Rp ",
            keys_tr="L,MkB,  Et, Et2, N,IntensityTfms, Flip1, Flip2, Zoom, Resize, ExtractBbox, CB,YoloBbox, Rp ", # NO ROTATION, No Del
            keys_val="L,MkB,Et, Et2, N, Resize,ExtractBbox, CB,YoloBbox,DelI, Rp",
        )
        if stage == "fit":
            self.ds_train = Dataset(data=self.train_dicts, transform=self.tfms_train)
            self.ds_val = Dataset(data=self.val_dicts, transform=self.tfms_valid)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size,num_workers=16,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size,num_workers=16,pin_memory=True)


# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------

if __name__ == "__main__":
    dm = DetectDataModule(data_dir = "/s/xnat_shadow/lidc2d/")
    dm.prepare_data()
    dm.setup(stage="fit")
    dl = dm.train_dataloader()


# %%




# %%
    dici = dm.train_dicts[0]
    dici  = dm.transforms_dict['N'](dici)
    dici = dm.transforms_dict['ExtractBbox'](dici)
# %%

    dici2 = dm.ds_train[153]
    dici2.keys()

    im = dici2['image'][0]
    lm = dici2['lm'][0]
    bb = dici2['lm_bbox'].copy()
    bb = bb[0].tolist()
    imv= torch.permute(im, (1, 0))
    lmv= torch.permute(lm, (1, 0))
    draw_image_lm_bbox(imv,lmv,*bb)
# %%

    start_x,start_y,stop_x,stop_y = bb
    # img= torch.load(fn_img)
    size_x = stop_x-start_x
    size_y = stop_y-start_y
    fig,(ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(imv)
    rect = patches.Rectangle((start_x,start_y),size_x,size_y, linewidth=1,edgecolor='r',facecolor='none')
    ax1.add_patch(rect)

    ax2.imshow(lmv)
    rect = patches.Rectangle((start_x,start_y),size_x,size_y, linewidth=1,edgecolor='r',facecolor='none')
    ax2.add_patch(rect)


# %%
    L = LoadTorchd(["image", "lm"])
    ExtractBbox = BoundingRectd(keys=["lm"])
    MkB = MakeBinary(["lm"])
# %%
# %%
    plt.ion()
    plt.imshow(dici["lm"][0])
# %%
    image_key = "image"
    box_key = "lm_bbox"
    label_key = "lm"
    patch_size = 92
    samples = 3
# %%
# %%
#SECTION:-------------------- TROUBLESHOOT--------------------------------------------------------------------------------------


    image_key = "image"
    label_key = "lm"
    box_key = "lm_bbox"
    probs= 1.0
    outputsize = [512,512]
    probs_int =1.0
    L = LoadTorchd([image_key, label_key])
    E = EnsureChannelFirstd(keys=[image_key])

    MkB = MakeBinary([label_key])
    Et = EnsureTyped(keys=[image_key ], dtype=torch.float32)
    Et2 = EnsureTyped(keys=[label_key], dtype=torch.long)
    E2 = EnsureTyped(keys=[image_key], dtype=torch.float16)
    ExtractBbox = BoundingRectd(keys=[label_key])
    CB = ConvertBoxToStandardModed(mode="xxyy", box_keys=[box_key])
    YoloBbox = BoundingBoxYOLOd(
        [box_key], 2, key_template_tensor=label_key, output_keys=["bbox_yolo"]
    )

    Rotate = RandRotated(
        keys=[image_key, label_key],
        prob=probs,
        keep_size=True,
        mode=["bilinear", "nearest"],
        range_x=[0.4, 0.4],
        lazy=True
    )
    Zoom = RandZoomd(
        keys=[image_key, label_key],
        mode = ['bilinear','nearest'],
        prob=probs,
        min_zoom=0.7,
        max_zoom=1.4,
        padding_mode="constant",
        keep_size=True,
        lazy=True
    )
    Flip1 = RandFlipd(
        keys=[image_key,label_key],
        prob=probs,
        spatial_axis=0,
        lazy=True
    )

    Flip2 = RandFlipd(
        keys=[image_key,label_key],
        prob=probs,
        spatial_axis=1,
        lazy=True
    )
    Resize = Resized(
        keys=[image_key,label_key],
        spatial_size = outputsize,
        mode= ['bilinear','nearest'],
        lazy=True
        
    )

    N = NormalizeIntensityd(keys=[image_key])
    IntensityTfms= Compose([
        RandGaussianSmoothd(
            keys=[image_key],
            prob=probs_int,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),
        RandScaleIntensityd(keys="image", factors=0.25, prob=probs_int),
        RandGaussianNoised(keys=["image"], mean=0, prob=probs_int, std=0.1),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=probs_int),
        RandAdjustContrastd(["image"], gamma=(0.7, 1.5)),
    ])
    Rp = RepeatChanneld(keys = [image_key],repeats=3)
    DelI = DeleteItemsd(keys = [label_key])

# %%
    tfms_train = Compose([L,MkB,  Et, Et2, N,IntensityTfms, Flip1, Flip2, Zoom, Resize, ExtractBbox, CB,YoloBbox, DelI,Rp ])
    # tfms_val = Compose([L, MkB, ExtractBbox, CB, Et, Et2, ClipoBbox, YoloBbox])
# %%
# %%

    dm = DetectDataModule(data_dir = "/s/xnat_shadow/lidc2d/")
    dm.prepare_data()
    dm.setup(stage="fit")
    dl = dm.train_dataloader()
    dici = ds[0]
    dici = dm.ds_train.data[10]
    dici  =tfms_train(dici)
# %%
    dici = L(dici)
    dici = MkB(dici)
    dici = Et(dici)
    dici = Et2(dici)
    dici = N(dici)
    dici = IntensityTfms(dici)
    dici = Rotate(dici)
    dici = ExtractBbox(dici)
    dici = CB(dici)
    dici = Z(dici)
    dici = Flip1(dici)
    dici = Flip2(dici)
    dici = RotBbox(dici)
    dici = ClipoBbox(dici)
    dici = YoloBbox(dici)

# %%

    lm = dici["image"][0]
    lmv = torch.permute(lm, (1, 0))
    bb = dici["lm_bbox"]
    bbl = bb.tolist()[0]

    print(bbl)
    draw_image_bbox(lmv, *bbl)
# %%
# %%
    plt.imshow(dici["image"][0])
    # dici = BB(dici)
# %%
# DeleteItemsd(keys=["box_mask"])
# RandGaussianNoised(keys=[image_key], prob=0.1, mean=0, std=0.1)
# RandGaussianSmoothd(
#     keys=[image_key],
#     prob=0.1,
#     sigma_x=(0.5, 1.0),
#     sigma_y=(0.5, 1.0),
#     sigma_z=(0.5, 1.0),
# ),
# RandScaleIntensityd(keys=[image_key], prob=0.15, factors=0.25),
# RandShiftIntensityd(keys=[image_key], prob=0.15, offsets=0.1),
# RandAdjustContrastd(keys=[image_key], prob=0.3, gamma=(0.7, 1.5)),
# EnsureTyped(keys=[image_key, box_key], dtype=compute_dtype),
# EnsureTyped(keys=[label_key], dtype=torch.long),
#
# %%
