# %%
import math
from functools import reduce
from operator import add
from pathlib import Path
from typing import Sequence, Union

import ipdb
import itk
import numpy as np
import SimpleITK as sitk
import torch
from fastcore.basics import listify, store_attr, warnings
from lightning.pytorch import LightningDataModule
from monai.config.type_definitions import KeysCollection, PathLike
from monai.data import DataLoader, Dataset
from monai.data.dataset import CacheDataset, PersistentDataset
from monai.data.image_reader import (ImageReader, ITKReader,
                                     _copy_compatible_dict, _stack_images)
from monai.data.itk_torch_bridge import has_itk
from monai.data.utils import is_supported_format, orientation_ras_lps
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (RandCropByPosNegLabeld,
                                                 ResizeWithPadOrCropd)
from monai.transforms.intensity.dictionary import (RandAdjustContrastd,
                                                   RandScaleIntensityd,
                                                   RandShiftIntensityd)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd
from monai.transforms.utility.dictionary import (EnsureChannelFirstd,
                                                 FgBgToIndicesd, ToDeviced)
from monai.utils.enums import MetaKeys, SpaceKeys
from monai.utils.misc import ensure_tuple
from monai.utils.module import require_pkg
from SimpleITK import Not
from torchvision.utils import Any

from fran.data.dataloader import img_lm_bbox_collated
from fran.data.dataset import (ImageMaskBBoxDatasetd, MaskLabelRemap2,
                               NormaliseClipd, SimpleDataset)
from fran.transforms.imageio import LoadTorchd, TorchReader
from fran.transforms.intensitytransforms import RandRandGaussianNoised
from fran.transforms.misc_transforms import MetaToDict
from fran.transforms.spatialtransforms import PadDeficitd
from fran.utils.fileio import load_dict
from fran.utils.helpers import find_matching_fn, folder_name_from_list
from fran.utils.imageviewers import ImageMaskViewer
from fran.utils.string import strip_extension

tr = ipdb.set_trace


def int_to_ratios(n_fg_labels, fgbg_ratio=3):
    ratios = [1] + [fgbg_ratio / n_fg_labels] * n_fg_labels
    return ratios


def list_to_fgbg(class_ratios):
    bg = class_ratios[0]
    fg = class_ratios[1:]
    fg = reduce(add, fg)
    return fg,bg


def simple_collated(batch):
    imgs = []
    labels = []
    for i, item in enumerate(batch):
        for ita in item:
            imgs.append(ita["image"])
            labels.append(ita["lm"])
    output = {"image": torch.stack(imgs, 0), "lm": torch.stack(labels, 0)}
    return output


class DataManager(LightningDataModule):
    def __init__(
        self,
        project,
        dataset_params: dict,
        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
        cache_rate=0.0,
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
        self.cache_rate = cache_rate
        self.assimilate_tfm_factors(transform_factors)

    def assimilate_tfm_factors(self, transform_factors):
        for key, value in transform_factors.items():
            dici = {"value": value[0], "prob": value[1]}
            setattr(self, key, dici)

    def prepare_data(self):
        # getting the right folders
        dataset_mode = self.dataset_params["mode"]
        assert dataset_mode in [
            "whole",
            "patch",
            "source",
            "lbd",
        ], "Set a value for mode in 'whole', 'patch' or 'source' "
        self.train_nii, self.valid_nii = self.project.get_train_val_files(
            self.dataset_params["fold"]
        )
        self.dataset_folder = self.derive_dataset_folder()
        # self.data_train = self.create_data_dicts(self.train_nii)
        # self.data_valid = self.create_data_dicts(self.valid_nii)

    def create_data_dicts(self, fnames):
        fnames = [strip_extension(fn) for fn in fnames]
        fnames = [fn + ".pt" for fn in fnames]
        fnames = fnames
        images_fldr = self.dataset_folder / ("images")
        lms_fldr = self.dataset_folder / ("lms")
        images = list(images_fldr.glob("*.pt"))
        data = []
        for fn in fnames:
            fn = Path(fn)
            img_fn = find_matching_fn(fn.name, images)
            mask_fn = lms_fldr / img_fn.name
            assert img_fn.exists(), "Missing image {}".format(img_fn)
            assert mask_fn.exists(), "Missing labelmap fn {}".format(mask_fn)
            dici = {"image": img_fn, "lm": mask_fn}
            data.append(dici)
        return data

    def derive_dataset_folder(self):
        raise NotImplementedError

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.batch_size * 2,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )
        return train_dl

    def val_dataloader(self):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.batch_size * 2,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )
        return valid_dl

    def create_affine_tfm(self):
        affine = RandAffined(
            keys=["image", "lm"],
            mode=["bilinear", "nearest"],
            prob=self.affine3d["p"],
            # spatial_size=self.dataset_params['src_dims'],
            rotate_range=self.affine3d["rotate_range"],
            scale_range=self.affine3d["scale_range"],
        )
        return affine

    def forward(self, inputs, target):
        return self.model(inputs)

    def create_transforms(self):
        raise NotImplementedError

    def setup(self, stage: str) -> None:
        raise NotImplementedError

    @property
    def src_dims(self):
        if self.dataset_params["zoom"] == True:
            src_dims = self.dataset_params["src_dims"]
        else:
            src_dims = self.dataset_params["patch_size"]
        return src_dims


class DataManagerSource(DataManager):
    def __init__(
        self,
        project,
        dataset_params: dict,
        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
        **kwargs
    ):
        super().__init__(
            project, dataset_params, transform_factors, affine3d, batch_size, **kwargs
        )
        self.collate_fn = simple_collated

    def derive_dataset_folder(self):
        prefix = "spc"
        spacing = self.dataset_params["spacing"]
        parent_folder = self.project.fixed_spacing_folder
        dataset_folder = folder_name_from_list(prefix, parent_folder, spacing)
        return dataset_folder

    def prepare_data(self):
        super().prepare_data()
        self.data_train = self.create_data_dicts(self.train_nii)
        self.data_valid = self.create_data_dicts(self.valid_nii)

    def create_transforms(self):
        fgbg_ratio = self.dataset_params["fgbg_ratio"]
        if isinstance(fgbg_ratio, list):
            fg,bg= list_to_fgbg(fgbg_ratio)
        else:
            fg = fgbg_ratio
            bg=1

        L = LoadImaged(
            keys=["image", "lm"],
            image_only=True,
            ensure_channel_first=False,
            simple_keys=True,
        )
        L.register(TorchReader())

        E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
        # G= ToDeviced(device = 'cuda',keys = ['image','lm'])
        # P = PadDeficitd(
        #     keys=["image", "lm"],
        #     source_key="image",
        #     spatial_size=self.src_dims,
        #     lazy=True,
        # )
        #
        Ind = MetaToDict(keys=["lm"], meta_keys=["lm_fg_indices", "lm_bg_indices"])
        Rtr = RandCropByPosNegLabeld(
            keys=["image", "lm"],
            label_key="lm",
            image_key="image",
            fg_indices_key="lm_fg_indices",
            bg_indices_key="lm_bg_indices",
            image_threshold=-2600,
            spatial_size=self.src_dims,
            pos=fg,
            neg=bg,
            num_samples=self.dataset_params["samples_per_file"],
            lazy=True,
            allow_smaller=True,
        )
        Rva = RandCropByPosNegLabeld(
            keys=["image", "lm"],
            label_key="lm",
            image_key="image",
            image_threshold=-2600,
            fg_indices_key="lm_fg_indices",
            bg_indices_key="lm_bg_indices",
            spatial_size=self.dataset_params["patch_size"],
            pos=1,
            neg=1,
            num_samples=self.dataset_params["samples_per_file"],
            lazy=True,
            allow_smaller=True,
        )

        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )

        # EnsureTyped(keys=["image", "lm"], device="cuda", track_meta=False),
        F1 = RandFlipd(
            keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=0, lazy=True
        )
        F2 = RandFlipd(
            keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=1, lazy=True
        )
        A = self.create_affine_tfm()
        int_augs = [
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
        ]

        Re = ResizeWithPadOrCropd(
            keys=["image", "lm"],
            source_key="image",
            spatial_size=self.dataset_params["patch_size"],
            lazy=True,
        )

        self.tfms_train = Compose([L, E, Ind, Rtr, F1, F2, A, Re, N, *int_augs])
        self.tfms_valid = Compose([L, E, Ind, Rva, Re, N])

        # self.tfms_train = Compose([L, E, P, Rtr, F1, F2, A, Re, N, *int_augs])
        # self.tfms_valid = Compose([L, E, P, Rva, N])

    def setup(self, stage: str = None):
        self.create_transforms()
        if self.cache_rate == 0.0:
            # self.train_ds = PersistentDataset(data=self.data_train, transform=self.tfms_train,cache_dir = self.project.cache_folder)
            self.train_ds = Dataset(data=self.data_train, transform=self.tfms_train)
        else:
            self.train_ds = CacheDataset(
                data=self.data_train,
                transform=self.tfms_train,
                cache_rate=self.cache_rate,
            )
        self.valid_ds = PersistentDataset(
            data=self.data_valid,
            transform=self.tfms_valid,
            cache_dir=self.project.cache_folder,
        )


class DataManagerLBD(DataManagerSource):
    def derive_dataset_folder(self, dataset_mode=None):
        spacing = self.dataset_params["spacing"]
        parent_folder = self.project.lbd_folder
        dataset_folder = folder_name_from_list(
            prefix="spc", parent_folder=parent_folder, values_list=spacing
        )
        assert dataset_folder.exists(), "Dataset folder {} does not exists".format(
            dataset_folder
        )
        return dataset_folder

    def prepare_data(self):
        super().prepare_data()
        self.data_train = self.create_data_dicts(self.train_nii)
        self.data_valid = self.create_data_dicts(self.valid_nii)


class DataManagerShort(DataManager):
    def prepare_data(self):
        super().prepare_data()
        self.train_list = self.train_list[:32]

    def train_dataloader(self, num_workers=4, **kwargs):
        return super().train_dataloader(num_workers, **kwargs)


class DataManagerPatch(DataManager):
    def __init__(
        self,
        project,
        dataset_params: dict,
        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
        **kwargs
    ):
        super().__init__(
            project, dataset_params, transform_factors, affine3d, batch_size, **kwargs
        )
        self.collate_fn = img_lm_bbox_collated

    def derive_dataset_folder(self):
        parent_folder = self.project.patches_folder
        spacing = self.dataset_params["spacing"]
        subfldr1 = folder_name_from_list("spc", parent_folder, spacing)
        src_dims = self.src_dims
        subfldr2 = folder_name_from_list("dim", subfldr1, src_dims)
        return subfldr2

    def create_transforms(self):
        P = MaskLabelRemap2(
            keys=["lm"], src_dest_labels=self.dataset_params["src_dest_labels"]
        )
        E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )
        if not math.isnan(self.dataset_params["src_dest_labels"]):
            all_after_item = [P, E, N]
        else:
            all_after_item = [E, N]

        t2 = [
            # EnsureTyped(keys=["image", "lm"], device="cuda", track_meta=False),
            RandFlipd(keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=0),
            RandFlipd(keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=1),
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
                keys=["image", "lm"],
                source_key="image",
                spatial_size=self.dataset_params["patch_size"],
            )
        ]
        self.tfms_train = Compose(all_after_item + t2 + t3)
        self.tfms_valid = Compose(all_after_item + t3)

    def setup(self, stage: str = None):
        self.create_transforms()
        fgbg_ratio = self.dataset_params["fgbg_ratio"]
        if isinstance(fgbg_ratio, int):
            n_fg_labels = len(self.project.global_properties["labels_all"])
            class_ratios = int_to_ratios(n_fg_labels=n_fg_labels, fgbg_ratio=fgbg_ratio)
        else:
            class_ratios = fgbg_ratio

        bboxes_fname = self.dataset_folder / "bboxes_info"
        self.train_ds = ImageMaskBBoxDatasetd(
            self.train_nii,
            bboxes_fname,
            class_ratios,
            transform=self.tfms_train,
        )
        self.valid_ds = ImageMaskBBoxDatasetd(
            self.valid_nii, bboxes_fname, transform=self.tfms_valid
        )


class DataManagerShort(DataManagerPatch):
    def prepare_data(self):
        super().prepare_data()
        self.train_list = self.train_list[:32]

    def train_dataloader(self, num_workers=4, **kwargs):
        return super().train_dataloader(num_workers, **kwargs)


# %%
if __name__ == "__main__":

    import torch

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *

    project_title = "lidc2"
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    )
    configuration_filename = None

    configs = ConfigMaker(
        proj, raytune=False, configuration_filename=configuration_filename
    ).config

    global_props = load_dict(proj.global_properties_filename)
    # %%
    batch_size = 2
    D = DataManagerSource(
        proj,
        dataset_params=configs["dataset_params"],
        transform_factors=configs["transform_factors"],
        affine3d=configs["affine3d"],
        batch_size=batch_size,
    )
    # %%
    D.prepare_data()
    D.setup(None)
    # %%

    dici = D.train_ds.data[0]
    L = LoadTorchd(keys=["image", "lm"])
    dici = L(dici)

    # %%

    Ind = MetaToDict(keys=["lm"], meta_keys=["lm_fg_indices", "lm_bg_indices"])
    Rtr = RandCropByPosNegLabeld(
        keys=["image", "lm"],
        label_key="lm",
        image_key="image",
        fg_indices_key="lm_fg_indices",
        bg_indices_key="lm_bg_indices",
        image_threshold=-2600,
        spatial_size=D.src_dims,
        pos=1,
        neg=1,
        num_samples=D.dataset_params["samples_per_file"],
        lazy=True,
        allow_smaller=False,
    )
    dici = Ind(dici)
    # %%
    dl = D.train_dataloader()
    iteri = iter(dl)
    aa = D.Train_ds[0]
    b = next(iteri)
    print(b["image"].shape)
    # %%
    im_fn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/nodes/spc_080_080_300/images/nodes_20_20190611_Neck1p0I30f3_thick.pt"
    label_fn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/nodes/spc_080_080_300/masks/nodes_20_20190611_Neck1p0I30f3_thick.pt"
    dici = {"image": im_fn, "lm": label_fn}
    D.setup()
    # %%
    ind = 1
    img = b["image"][ind][0]
    lab = b["lm"][ind][0]
    ImageMaskViewer([img, lab])
# %
