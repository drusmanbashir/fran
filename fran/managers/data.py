# %%
from fran.managers.project import Project
from lightning import LightningDataModule
from fran.utils.helpers import pbar, pp
from monai.transforms.transform import RandomizableTransform
from fran.preprocessing.helpers import bbox_bg_only
import ast
import math
from functools import reduce
from operator import add
from pathlib import Path

import ipdb
import numpy as np
import torch
from fastcore.basics import listify, operator, store_attr, warnings
from monai.data import DataLoader, Dataset
from monai.data.dataset import CacheDataset, LMDBDataset, PersistentDataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (
    RandCropByPosNegLabeld,
    ResizeWithPadOrCropd,
)
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, Resized
from monai.transforms.utility.dictionary import (
    EnsureChannelFirstd,
)

from fran.data.collate import img_lm_bbox_collated, source_collated, whole_collated
from fran.data.dataset import (
    ImageMaskBBoxDatasetd,
    NormaliseClipd,
    fg_in_bboxes,
)
from fran.transforms.imageio import LoadTorchd, TorchReader
from fran.transforms.intensitytransforms import RandRandGaussianNoised
from fran.transforms.misc_transforms import LoadTorchDict, MetaToDict
from fran.utils.config_parsers import ConfigMaker, is_excel_None
from fran.utils.fileio import load_dict, load_yaml
from fran.utils.helpers import find_matching_fn, folder_name_from_list
from fran.utils.imageviewers import ImageMaskViewer
from fran.utils.string import ast_literal_eval, strip_extension
import re
import os


common_vars_filename = os.environ["FRAN_COMMON_PATHS"]
COMMON_PATHS = load_yaml(common_vars_filename)

tr = ipdb.set_trace


def int_to_ratios(n_fg_labels, fgbg_ratio=3):
    ratios = [1] + [fgbg_ratio / n_fg_labels] * n_fg_labels
    return ratios


def list_to_fgbg(class_ratios):
    bg = class_ratios[0]
    fg = class_ratios[1:]
    fg = reduce(add, fg)
    return fg, bg


class RandomPatch(RandomizableTransform):
    """
    to be used by DataManagerPatch
    """

    def randomize(self, data=None):
        n_patches = data["n_patches"]
        self.indx = self.R.randint(0, n_patches)
        self.indx = str(self.indx)

    def __call__(self, data: list):
        self.randomize(data)
        image_key = "image_" + self.indx
        lm_key = "lm_" + self.indx
        indices_key = "indices_" + self.indx
        dici = {
            "image": data[image_key],
            "lm": data[lm_key],
            "indices": data[indices_key],
        }
        return dici


class DataManager(LightningDataModule):
    def __init__(
        self,
        project,
        dataset_params: dict,
        config: dict,
        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
        cache_rate=0.0,
        ds_type=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.plan = config["plan"]
        store_attr(but="transform_factors")
        global_properties = load_dict(project.global_properties_filename)
        self.dataset_params["intensity_clip_range"] = global_properties[
            "intensity_clip_range"
        ]
        self.dataset_params["mean_fg"] = global_properties["mean_fg"]
        self.dataset_params["std_fg"] = global_properties["std_fg"]
        self.batch_size = batch_size
        self.cache_rate = cache_rate
        self.ds_type = ds_type
        self.set_effective_batch_size()
        self.assimilate_tfm_factors(transform_factors)
        self.set_tfm_keys()
        self.collate_fn = None # needs to be set in each inheriting class

    def __str__(self):
        return 'DataManager instance with parameters: ' + ', '.join([f'{k}={v}' for k, v in vars(self).items()])

    def __repr__(self):
        return f'DataManager(' + ', '.join([f'{k}={v}' for k, v in vars(self).items()]) + ')'

    def set_tfm_keys(self):
        raise NotImplementedError


    def set_effective_batch_size(self):
        if "samples_per_file" in self.plan:
            self.effective_batch_size = int(np.maximum(1, 
                self.batch_size / self.plan["samples_per_file"]
            ))
            print(
                "Given {0} Samples per file and {1} batch_size on the GPU, effective batch size (number of file tensors loaded then sampled for for training is:\n {2} ".format(
                    self.plan["samples_per_file"],
                    self.batch_size,
                    self.effective_batch_size,
                )
            )

        else:
            self.effective_batch_size = self.batch_size

    def assimilate_tfm_factors(self, transform_factors):
        for key, value in transform_factors.items():
            dici = {"value": value[0], "prob": value[1]}
            setattr(self, key, dici)
    def create_transforms(self, but=None):
        """
        Creates transformations used for data preprocessing. 
        Transforms specified in 'but' are omitted from creation and inclusion.

        Parameters:
        but (str): A comma-separated string of transform keys to be excluded.
        """
        # Parse the 'but' string into a list of keys to exclude
        if but:
            exclude_keys = {key.strip() for key in but.split(",")}
        else:
            exclude_keys = set()

        # Initialize an empty dictionary to store the transforms
        self.transforms_dict = {}

        # Conditionally create transforms based on exclusion list
        if "E" not in exclude_keys:
            E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
            self.transforms_dict["E"] = E

        if "N" not in exclude_keys:
            N = NormaliseClipd(
                keys=["image"],
                clip_range=self.dataset_params["intensity_clip_range"],
                mean=self.dataset_params["mean_fg"],
                std=self.dataset_params["std_fg"],
            )
            self.transforms_dict["N"] = N
        
        if "RP" not in exclude_keys:
            RP = RandomPatch()
            self.transforms_dict["RP"] = RP
        
        if "F1" not in exclude_keys:
            F1 = RandFlipd(
                keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=0, lazy=True
            )
            self.transforms_dict["F1"] = F1
        
        if "F2" not in exclude_keys:
            F2 = RandFlipd(
                keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=1, lazy=True
            )
            self.transforms_dict["F2"] = F2

        if "IntensityTfms" not in exclude_keys:
            IntensityTfms = [
                RandScaleIntensityd(
                    keys="image", factors=self.scale["value"], prob=self.scale["prob"]
                ),
                RandRandGaussianNoised(
                    keys=["image"], std_limits=self.noise["value"], prob=self.noise["prob"]
                ),
                RandShiftIntensityd(
                    keys="image", offsets=self.shift["value"], prob=self.shift["prob"]
                ),
                RandAdjustContrastd(
                    ["image"], gamma=self.contrast["value"], prob=self.contrast["prob"]
                ),
            ]
            self.transforms_dict["IntensityTfms"] = IntensityTfms

        if "Affine" not in exclude_keys:
            Affine = RandAffined(
                keys=["image", "lm"],
                mode=["bilinear", "nearest"],
                prob=self.affine3d["p"],
                rotate_range=self.affine3d["rotate_range"],
                scale_range=self.affine3d["scale_range"],
            )
            self.transforms_dict["Affine"] = Affine

        if "Re" not in exclude_keys:
            Re = ResizeWithPadOrCropd(
                keys=["image", "lm"],
                spatial_size=self.plan["patch_size"],
                lazy=True,
            )
            self.transforms_dict["Re"] = Re

        # Continue similarly for the remaining transforms like L, Ld, Ind, Rtr, Rva...

        # Example for some more exclusions:
        if "L" not in exclude_keys:
            L = LoadImaged(
                keys=["image", "lm"],
                image_only=True,
                ensure_channel_first=False,
                simple_keys=True,
            )
            L.register(TorchReader())
            self.transforms_dict["L"] = L

        if "Ld" not in exclude_keys:
            Ld = LoadTorchDict(
                keys=["indices"], select_keys=["lm_fg_indices", "lm_bg_indices"]
            )
            self.transforms_dict["Ld"] = Ld
        
        if "Ind" not in exclude_keys:
            Ind = MetaToDict(keys=["lm"], meta_keys=["lm_fg_indices", "lm_bg_indices"])
            self.transforms_dict["Ind"] = Ind

        if "Rtr" not in exclude_keys:
            Rtr = RandCropByPosNegLabeld(
                keys=["image", "lm"],
                label_key="lm",
                image_key="image",
                fg_indices_key="lm_fg_indices",
                bg_indices_key="lm_bg_indices",
                image_threshold=-2600,
                spatial_size=self.src_dims,
                pos=self.dataset_params["fgbg_ratio"],
                neg=1,
                num_samples=self.plan["samples_per_file"],
                lazy=True,
                allow_smaller=True,
            )
            self.transforms_dict["Rtr"] = Rtr

        if "Rva" not in exclude_keys:
            Rva = RandCropByPosNegLabeld(
                keys=["image", "lm"],
                label_key="lm",
                image_key="image",
                fg_indices_key="lm_fg_indices",
                bg_indices_key="lm_bg_indices",
                image_threshold=-2600,
                spatial_size=self.plan["patch_size"],
                pos=1,
                neg=1,
                num_samples=self.plan["samples_per_file"],
                lazy=True,
                allow_smaller=True,
            )
            self.transforms_dict["Rva"] = Rva

    def create_transforms(self, keys="all"):
        """
        Creates transformations used for data preprocessing. 
        Only the transforms specified in 'include' are created and included.
        If 'include' is 'all', all available transforms are created.

        Parameters:
        include (str): A comma-separated string of transform keys to be included, 
                       or 'all' to include all transforms.
        """
        # Parse the 'include' string into a list of keys to include
        if keys != "all":
            include_keys = {key.strip() for key in keys.split(",")}
        else:
            include_keys = 'all'# Use this to indicate all transforms should be included

        # Initialize an empty dictionary to store the transforms
        self.transforms_dict = {}

        # Conditionally create transforms based on inclusion list
        if include_keys =='all' or "E" in include_keys:
            E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
            self.transforms_dict["E"] = E

        if include_keys =='all' or "N" in include_keys:
            N = NormaliseClipd(
                keys=["image"],
                clip_range=self.dataset_params["intensity_clip_range"],
                mean=self.dataset_params["mean_fg"],
                std=self.dataset_params["std_fg"],
            )
            self.transforms_dict["N"] = N

        if include_keys =='all' or "RP" in include_keys:
            RP = RandomPatch()
            self.transforms_dict["RP"] = RP

        if include_keys =='all' or "F1" in include_keys:
            F1 = RandFlipd(
                keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=0, lazy=True
            )
            self.transforms_dict["F1"] = F1

        if include_keys =='all' or "F2" in include_keys:
            F2 = RandFlipd(
                keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=1, lazy=True
            )
            self.transforms_dict["F2"] = F2

        if include_keys =='all' or "IntensityTfms" in include_keys:
            IntensityTfms = [
                RandScaleIntensityd(
                    keys="image", factors=self.scale["value"], prob=self.scale["prob"]
                ),
                RandRandGaussianNoised(
                    keys=["image"], std_limits=self.noise["value"], prob=self.noise["prob"]
                ),
                RandShiftIntensityd(
                    keys="image", offsets=self.shift["value"], prob=self.shift["prob"]
                ),
                RandAdjustContrastd(
                    ["image"], gamma=self.contrast["value"], prob=self.contrast["prob"]
                ),
            ]
            self.transforms_dict["IntensityTfms"] = IntensityTfms

        if include_keys =='all' or "Affine" in include_keys:
            Affine = RandAffined(
                keys=["image", "lm"],
                mode=["bilinear", "nearest"],
                prob=self.affine3d["p"],
                rotate_range=self.affine3d["rotate_range"],
                scale_range=self.affine3d["scale_range"],
            )
            self.transforms_dict["Affine"] = Affine

        if include_keys =='all' or "Re" in include_keys:
            Re = ResizeWithPadOrCropd(
                keys=["image", "lm"],
                spatial_size=self.plan["patch_size"],
                lazy=True,
            )
            self.transforms_dict["Re"] = Re

        # Continue similarly for the remaining transforms like L, Ld, Ind, Rtr, Rva...

        if include_keys =='all' or "L" in include_keys:
            L = LoadImaged(
                keys=["image", "lm"],
                image_only=True,
                ensure_channel_first=False,
                simple_keys=True,
            )
            L.register(TorchReader())
            self.transforms_dict["L"] = L

        if include_keys =='all' or "Ld" in include_keys:
            Ld = LoadTorchDict(
                keys=["indices"], select_keys=["lm_fg_indices", "lm_bg_indices"]
            )
            self.transforms_dict["Ld"] = Ld

        if include_keys =='all' or "Ind" in include_keys:
            Ind = MetaToDict(keys=["lm"], meta_keys=["lm_fg_indices", "lm_bg_indices"])
            self.transforms_dict["Ind"] = Ind

        if include_keys =='all' or "Rtr" in include_keys:
            Rtr = RandCropByPosNegLabeld(
                keys=["image", "lm"],
                label_key="lm",
                image_key="image",
                fg_indices_key="lm_fg_indices",
                bg_indices_key="lm_bg_indices",
                image_threshold=-2600,
                spatial_size=self.src_dims,
                pos=self.dataset_params["fgbg_ratio"],
                neg=1,
                num_samples=self.plan["samples_per_file"],
                lazy=True,
                allow_smaller=True,
            )
            self.transforms_dict["Rtr"] = Rtr

        if include_keys =='all' or "Rva" in include_keys:
            Rva = RandCropByPosNegLabeld(
                keys=["image", "lm"],
                label_key="lm",
                image_key="image",
                fg_indices_key="lm_fg_indices",
                bg_indices_key="lm_bg_indices",
                image_threshold=-2600,
                spatial_size=self.plan["patch_size"],
                pos=1,
                neg=1,
                num_samples=self.plan["samples_per_file"],
                lazy=True,
                allow_smaller=True,
            )
            self.transforms_dict["Rva"] = Rva


    def set_transforms(self, keys_tr: str, keys_val: str):
        self.tfms_train = self.tfms_from_dict(keys_tr)
        self.tfms_valid = self.tfms_from_dict(keys_val)

    def tfms_from_dict(self, keys: str):
        keys = keys.split(",")
        tfms = []
        for key in keys:
            tfm = self.transforms_dict[key]
            if key == "IntensityTfms":
                tfms.extend(tfm)
            else:
                tfms.append(tfm)
        tfms = Compose(tfms)
        return tfms

    def prepare_data(self):
        # getting the right folders
        dataset_mode = self.plan["mode"]
        assert dataset_mode in [
            "whole",
            "baseline",
            "patch",
            "source",
            "pbd",
            "lbd",
        ], "Set a value for mode in 'whole', 'patch' or 'source' "
        self.train_cases, self.valid_cases = self.project.get_train_val_files(
            self.dataset_params["fold"],self.plan['datasources']
        )
        self.data_folder = self.derive_data_folder()

    def create_data_dicts(self, fnames):
        fnames = [strip_extension(fn) for fn in fnames]
        fnames = [fn + ".pt" for fn in fnames]
        fnames = fnames
        images_fldr = self.data_folder / ("images")
        lms_fldr = self.data_folder / ("lms")
        inds_fldr = self.infer_inds_fldr(self.plan)
        images = list(images_fldr.glob("*.pt"))
        data = []

        for fn in pbar(fnames):
            fn = Path(fn)
            img_fn = find_matching_fn(fn.name, images, 'all')
            lm_fn = find_matching_fn(fn.name, lms_fldr, 'all')
            indices_fn = inds_fldr / img_fn.name
            assert img_fn.exists(), "Missing image {}".format(img_fn)
            assert lm_fn.exists(), "Missing labelmap fn {}".format(lm_fn)
            dici = {"image": img_fn, "lm": lm_fn, "indices": indices_fn}
            data.append(dici)
        return data

    def infer_inds_fldr(self, plan):
        fg_indices_exclude = plan["fg_indices_exclude"]
        if is_excel_None(fg_indices_exclude):
            fg_indices_exclude = None
            indices_subfolder = "indices"
        else:
            if isinstance(fg_indices_exclude, str):
                fg_indices_exclude = ast.literal_eval(fg_indices_exclude)
            fg_indices_exclude = listify(fg_indices_exclude)
            indices_subfolder = "indices_fg_exclude_{}".format(
                "".join([str(x) for x in fg_indices_exclude])
            )
        return self.data_folder / (indices_subfolder)

    def derive_data_folder(self):
        raise NotImplementedError

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.effective_batch_size,
            num_workers=self.effective_batch_size * 2,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )
        return train_dl
    def val_dataloader(self):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.effective_batch_size,
            num_workers=self.effective_batch_size * 2,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )
        return valid_dl

    def forward(self, inputs, target):
        return self.model(inputs)

    def setup(self, stage: str = None) -> None:
        raise NotImplementedError

    @property
    def src_dims(self):
        if self.dataset_params["zoom"] == True:
            src_dims = self.dataset_params["src_dims"]
        else:
            src_dims = self.plan["patch_size"]
        return src_dims

    @property
    def cache_folder(self):
        parent_folder = Path(COMMON_PATHS['cache_folder'])/(self.project.project_title)
        return parent_folder/(self.data_folder.name)

class DataManagerSource(DataManager):
    def __init__(
        self,
        project,
        dataset_params: dict,
        config: dict,
        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
        **kwargs
    ):
        super().__init__(
            project,
            dataset_params,
            config,
            transform_factors,
            affine3d,
            batch_size,
            **kwargs
        )
        self.collate_fn = source_collated

    def __str__(self):
        return 'DataManagerSource instance with parameters: ' + ', '.join([f'{k}={v}' for k, v in vars(self).items()])

    def __repr__(self):
        return f'DataManagerSource(' + ', '.join([f'{k}={v}' for k, v in vars(self).items()]) + ')'
    def set_tfm_keys(self):
        self.keys_val = "L,Ld,E,Rva,Re,N"
        self.keys_tr = "L,Ld,E,Rtr,F1,F2,Affine,Re,N,IntensityTfms"

    def derive_data_folder(self):
        prefix = "spc"
        spacing = self.plan["spacing"]
        parent_folder = self.project.fixed_spacing_folder
        data_folder = folder_name_from_list(prefix, parent_folder, spacing)
        return data_folder

    def prepare_data(self):
        super().prepare_data()
        self.data_train = self.create_data_dicts(self.train_cases)
        self.data_valid = self.create_data_dicts(self.valid_cases)

    def create_transforms(self,keys='all'):
        super().create_transforms(keys)

    def setup(self, stage: str = None):
        self.create_transforms()
        self.set_transforms(keys_tr=self.keys_tr, keys_val=self.keys_val)
        print("Setting up datasets. Training ds type is: ", self.ds_type)
        if is_excel_None(self.ds_type):
            self.train_ds = Dataset(data=self.data_train, transform=self.tfms_train)
            print("Vanilla Pytorch Dataset set up.")
        elif self.ds_type == "cache":
            self.train_ds = CacheDataset(
                data=self.data_train,
                transform=self.tfms_train,
                cache_rate=self.cache_rate,
            )
        elif self.ds_type == "lmdb":
            self.train_ds = LMDBDataset(
                data=self.data_train,
                transform=self.tfms_train,
                cache_dir=self.cache_folder,
                db_name="training_cache",
            )
        else:
            raise NotImplementedError
        self.valid_ds = PersistentDataset(
            data=self.data_valid,
            transform=self.tfms_valid,
            cache_dir=self.cache_folder,
        )



class DataManagerWhole(DataManagerSource):

    def __init__(
        self,
        project,
        dataset_params: dict,
        config: dict,
        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
        **kwargs
    ):
        super().__init__(
            project,
            dataset_params,
            config,
            transform_factors,
            affine3d,
            batch_size,
            **kwargs
        )
        self.keys_tr = "L,E,F1,F2,Affine,Resize,N,IntensityTfms"
        self.keys_val = "L,E,Resize,N"
        self.collate_fn = whole_collated

    def __str__(self):
        return 'DataManagerWhole instance with parameters: ' + ', '.join([f'{k}={v}' for k, v in vars(self).items()])

    def __repr__(self):
        return f'DataManagerWhole(' + ', '.join([f'{k}={v}' for k, v in vars(self).items()]) + ')'

    def derive_data_folder(self):
        prefix = "sze"
        spatial_size = self.plan["patch_size"]
        parent_folder = self.project.fixed_size_folder
        data_folder = folder_name_from_list(prefix, parent_folder, spatial_size)
        return data_folder

    def create_transforms(self):
        super().create_transforms(keys= self.keys_tr+","+self.keys_val)
        Resize = Resized(
            keys=["image", "lm"],
            spatial_size=self.plan["patch_size"],
            mode=["linear", "nearest"],
            lazy=True,
        )
        self.transforms_dict.update({"Resize": Resize})


    def create_data_dicts(self, fnames):
        fnames = [strip_extension(fn) for fn in fnames]
        fnames = [fn + ".pt" for fn in fnames]
        fnames = fnames
        images_fldr = self.data_folder / ("images")
        lms_fldr = self.data_folder / ("lms")
        images = list(images_fldr.glob("*.pt"))
        data = []
        # for fn in fnames[400:432]:
        for fn in pbar(fnames):
            fn = Path(fn)
            img_fn = find_matching_fn(fn.name, images, 'all')
            lm_fn = find_matching_fn(fn.name, lms_fldr, 'all')
            assert img_fn.exists(), "Missing image {}".format(img_fn)
            assert lm_fn.exists(), "Missing labelmap fn {}".format(lm_fn)
            dici = {"image": img_fn, "lm": lm_fn}
            data.append(dici)
        return data

class DataManagerLBD(DataManagerSource):
    def derive_data_folder(self, dataset_mode=None):
        spacing = ast_literal_eval(self.plan["spacing"])
        parent_folder = self.project.lbd_folder
        folder_suffix = "plan" + str(self.dataset_params["plan"])
        data_folder = folder_name_from_list(
            prefix="spc",
            parent_folder=parent_folder,
            values_list=spacing,
            suffix=folder_suffix,
        )
        assert data_folder.exists(), "Dataset folder {} does not exists".format(
            data_folder
        )
        return data_folder
    def __repr__(self):
        return (f"DataManagerLBD(plan={self.plan}, "
                f"dataset_params={self.dataset_params}, "
                f"lbd_folder={self.project.lbd_folder})")

    def __str__(self):
        return ("LBD Data Manager with plan {} and dataset parameters: {} "
                "(using LBD folder: {})".format(
                    self.plan,
                    self.dataset_params,
                    self.project.lbd_folder
                ))


    # def prepare_data(self):
    #     super().prepare_data()
    #     self.data_train = self.create_data_dicts(self.train_cases[:32])
    #     self.data_valid = self.create_data_dicts(self.valid_cases[:16])


class DataManagerWID(DataManagerLBD):
    def derive_data_folder(self, dataset_mode=None):
        spacing = self.plan["spacing"]
        parent_folder = self.project.pbd_folder
        folder_suffix = "plan" + str(self.dataset_params["plan"])
        data_folder = folder_name_from_list(
            prefix="spc",
            parent_folder=parent_folder,
            values_list=spacing,
            suffix=folder_suffix,
        )
        assert data_folder.exists(), "Dataset folder {} does not exists".format(
            data_folder
        )
        return data_folder

    def __repr__(self):
        return (f"DataManagerPBD(plan={self.plan}, "
                f"dataset_params={self.dataset_params}, "
                f"pbd_folder={self.project.pbd_folder})")

    def __str__(self):
        return ("Patient-bound  Data Manager (PBD) with plan {} and dataset parameters: {} "
                "(using PBD folder: {})".format(
                    self.plan,
                    self.dataset_params,
                    self.project.pbd_folder
                ))



class DataManagerShort(DataManager):
    def prepare_data(self):
        super().prepare_data()
        self.train_list = self.train_list[:32]

    def train_dataloader(self, num_workers=4, **kwargs):
        return super().train_dataloader(num_workers, **kwargs)


# CODE: in the below class move Rtr after Affine and get rid of Re to see if it affects training speed / model accuracy
class DataManagerPatchLegacy(DataManager):
    """
    Uses bboxes to randonly select fg bg labels. New version(below) uses monai fgbgindices instead
    """

    def __init__(
        self,
        project,
        dataset_params: dict,
        config: dict,
        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
        **kwargs
    ):
        super().__init__(
            project,
            dataset_params,
            config,
            transform_factors,
            affine3d,
            batch_size,
            **kwargs
        )
        self.collate_fn = img_lm_bbox_collated

    def derive_data_folder(self):
        parent_folder = self.project.patches_folder
        plan_name = "plan" + str(self.dataset_params["plan"])
        source_plan_name = self.plan["source_plan"]
        source_plan = self.config[source_plan_name]
        spacing = ast.literal_eval(source_plan["spacing"])
        # spacing = self.dataset_params["spacing"]
        subfldr1 = folder_name_from_list("spc", parent_folder, spacing)
        src_dims = self.src_dims
        subfldr2 = folder_name_from_list("dim", subfldr1, src_dims, plan_name)
        return subfldr2

    def setup(self, stage: str = None):
        self.create_transforms()
        if not math.isnan(self.dataset_params["src_dest_labels"]):
            keys_tr = "P,E,F1,F2,Affine,Re,N,I"
        else:
            keys_tr = "E,F1,F2,Affine,Re,N,I"
        keys_val = "E,Re,N"
        self.set_transforms(keys_tr=keys_tr, keys_val=keys_val)
        fgbg_ratio = self.dataset_params["fgbg_ratio"]
        if isinstance(fgbg_ratio, int):
            n_fg_labels = len(self.project.global_properties["labels_all"])
            class_ratios = int_to_ratios(n_fg_labels=n_fg_labels, fgbg_ratio=fgbg_ratio)
        else:
            class_ratios = fgbg_ratio

        bboxes_fname = self.data_folder / "bboxes_info"
        self.train_ds = ImageMaskBBoxDatasetd(
            self.train_cases,
            bboxes_fname,
            class_ratios,
            transform=self.tfms_train,
        )
        self.valid_ds = ImageMaskBBoxDatasetd(
            self.valid_cases, bboxes_fname, transform=self.tfms_valid
        )

    @property
    def src_dims(self):
        return self.plan["patch_size"]


class DataManagerPatch(DataManager):
    def __init__(
        self,
        project,
        dataset_params: dict,
        config: dict,
        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
        **kwargs
    ):
        self.collate_fn = source_collated
        super().__init__(
            project,
            dataset_params,
            config,
            transform_factors,
            affine3d,
            batch_size,
            **kwargs
        )
        self.derive_data_folder()
        self.load_bboxes()
        self.fg_bg_prior = fg_in_bboxes(self.bboxes)

    def get_patch_files(self, bboxes, case_id: str):
        cids = np.array([bb["case_id"] for bb in bboxes])
        cid_inds = np.where(cids == case_id)[0]
        assert len(cid_inds) > 0, "No bboxes for case {0}".format(case_id)
        n_patches = len(cid_inds)
        bboxes_out = {"n_patches": n_patches}
        for ind in cid_inds:
            bb = bboxes[ind]
            pat = re.compile(r"_(\d+)\.pt")

            fn = bb["filename"]
            matched = pat.search(fn.name)
            indx = matched.groups()[0]
            fn_name = strip_extension(fn.name) + "_" + str(indx) + ".pt"
            lm_fn = Path(fn)
            img_fn = lm_fn.str_replace("lms", "images")
            indices_fn = lm_fn.str_replace("lms", "indices")
            # assert(bb['case_id'] == case_id),"Strange error: {} not in bb".format(case_id)
            assert all(
                [fn.exists() for fn in [lm_fn, img_fn, indices_fn]]
            ), "Image of LM file does not exists {0}, {1}, {2}".format(
                lm_fn, img_fn, indices_fn
            )
            bb_out = {
                "lm_" + indx: lm_fn,
                "image_" + indx: img_fn,
                "indices_" + indx: indices_fn,
                "bbox_stats_" + indx: bb["bbox_stats"],
            }
            # bb.pop("filename")
            bboxes_out.update(bb_out)
        return bboxes_out

    def load_bboxes(self):
        bbox_fn = self.data_folder / "bboxes_info"
        self.bboxes = load_dict(bbox_fn)

    def prepare_data(self):
        self.train_cids, self.valid_cids = self.project.get_train_val_cids(
            self.dataset_params["fold"]
        )
        print("Creating train data dicts")
        self.data_train = self.create_data_dicts(self.train_cids)
        print("Creating val data dicts")
        self.data_valid = self.create_data_dicts(self.valid_cids)

    def get_label_info(self, case_patches):
        indices = []
        labels_per_file = []
        for indx, bb in enumerate(case_patches):
            bbox_stats = bb["bbox_stats"]
            labels = [(a["label"]) for a in bbox_stats if not a["label"] == "all_fg"]
            if bbox_bg_only(bbox_stats) == True:
                labels = [0]
            else:
                labels = [0] + labels
            indices.append(indx)
            labels_per_file.append(labels)
        labels_this_case = list(set(reduce(operator.add, labels_per_file)))
        return {
            "file_indices": indices,
            "labels_per_file": labels_per_file,
            "labels_this_case": labels_this_case,
        }

    def create_data_dicts(self, cids):
        patches = []
        for cid in pbar(cids):
            dici = {"case_id": cid}
            patch_fns = self.get_patch_files(self.bboxes, cid)
            dici.update(patch_fns)
            patches.append(dici)
        return patches

    def create_transforms(self, keys='all'):
        super().create_transforms(keys)

    def derive_data_folder(self):
        parent_folder = self.project.patches_folder
        plan_name = "plan" + str(self.dataset_params["plan"])
        source_plan_name = self.plan["source_plan"]
        source_plan = self.config[source_plan_name]
        spacing = ast.literal_eval(source_plan["spacing"])
        # spacing = self.dataset_params["spacing"]
        subfldr1 = folder_name_from_list("spc", parent_folder, spacing)
        patch_size = ast.literal_eval(
            self.plan["patch_size"]
        )  # self.plan['patch_size']
        subfldr2 = folder_name_from_list("dim", subfldr1, patch_size, plan_name)
        self.data_folder = subfldr2

    # CODE: use same validation dataloader in all flavours of training to make it comparable
    def setup(self, stage: str = None):
        self.create_transforms()
        fgbg_ratio = self.dataset_params["fgbg_ratio"]
        fgbg_ratio_adjusted = fgbg_ratio / self.fg_bg_prior
        self.dataset_params["fgbg_ratio"] = fgbg_ratio_adjusted
        if not math.isnan(self.dataset_params["src_dest_labels"]):
            keys_tr = "RP,L,Ld,P,E,Rtr,F1,F2,Affine,Re,N,I"
        else:
            keys_tr = "RP,L,Ld,E,Rva,F1,F2,Affine,Re,N,I"
        keys_val = "RP,L,Ld,E,Rva,Re,N"
        self.set_transforms(keys_tr=keys_tr, keys_val=keys_val)
        self.train_ds = LMDBDataset(
            data=self.data_train,
            transform=self.tfms_train,
            cache_dir=self.cache_folder,
            db_name="training_cache",
        )
        self.valid_ds = LMDBDataset(
            data=self.data_valid,
            transform=self.tfms_valid,
            cache_dir=self.cache_folder,
            db_name="valid_cache",
        )

    @property
    def src_dims(self):
        return self.plan["patch_size"]


class DataManagerBaseline(DataManagerLBD):
    '''
    Small dataset of size =batchsize comprising a single batch. No augmentations. Used to get a baseline
    It has no training augmentations. Whether the flag is True or False doesnt matter.
    Note: It inherits from LBD dataset.
    '''
    def __init__(self, project, dataset_params: dict, config: dict, transform_factors: dict, affine3d: dict, batch_size=8, **kwargs):
        super().__init__(project, dataset_params, config, transform_factors, affine3d, batch_size,**kwargs)
        self.collate_fn = whole_collated


    def __str__(self):
        return 'DataManagerBaseline instance with parameters: ' + ', '.join([f'{k}={v}' for k, v in vars(self).items()])

    def __repr__(self):
        return f'DataManagerBaseline(' + ', '.join([f'{k}={v}' for k, v in vars(self).items()]) + ')'

    def set_effective_batch_size(self):
        self.effective_batch_size = self.batch_size

    def set_tfm_keys(self):
        self.keys_val = "L,Ld,E,Re,N"
        self.keys_tr=self.keys_val


    def derive_data_folder(self, dataset_mode=None):
        # return data_folder
        source_plan_name = self.plan["source_plan"]
        source_plan = self.config[source_plan_name]

        source_ds_type =  source_plan['mode']
        if source_ds_type == 'lbd':
            parent_folder = self.project.lbd_folder
        else:
            raise NotImplemented
        spacing = ast_literal_eval(source_plan['spacing'])

        data_folder= folder_name_from_list("spc", parent_folder, spacing, source_plan_name)
        assert data_folder.exists(), "Dataset folder {} does not exists".format(
            data_folder
        )
        return data_folder

    def setup(self, stage: str = None):
        self.create_transforms()
        self.set_transforms(keys_tr=self.keys_tr, keys_val=self.keys_val)
        print("Setting up datasets. Training ds type is: ", self.ds_type)
        self.train_ds = Dataset(data=self.data_train, transform=self.tfms_train)
        self.valid_ds = Dataset(
            data=self.data_valid,
            transform=self.tfms_valid,
        )
    
    def prepare_data(self):
        super().prepare_data()
        self.data_train= self.data_train[:self.batch_size]
        self.data_valid= self.data_valid[:self.batch_size]

    @property
    def cache_folder(self):
        parent_folder = Path(COMMON_PATHS['cache_folder'])/(self.project.project_title)
        return parent_folder/(self.data_folder.name+"_baseline")

# %%
if __name__ == "__main__":
# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>

    import torch

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *

    project_title = "totalseg"
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    )
    configuration_filename = None

    config = ConfigMaker(
        proj, raytune=False, configuration_filename=configuration_filename
    ).config

    global_props = load_dict(proj.global_properties_filename)

    pp(config['plan'])

# SECTION:-------------------- DataManagerWhole-------------------------------------------------------------------------------------- <CR>
# %%
    D = DataManagerWhole(
        project=proj,
        dataset_params = config["dataset_params"],
        affine3d=config["affine3d"],
        batch_size=4,
        transform_factors=config["transform_factors"],
        config=config,
    )

# %%
    D.prepare_data()
    D.setup()
    D.data_folder
    dl = D.train_dataloader()
    bb = D.train_ds[0]
# %%
    iteri = iter(dl)
    b = next(iteri)
    b = D.train_ds[0]
    im = b['image']
    lm = b['lm']

    ImageMaskViewer([im[0], lm[0]])


# %%
# %%


#SECTION:-------------------- DataManagerPlain--------------------------------------------------------------------------------------
# %%
    batch_size = 2
    D = DataManagerBaseline(
        proj,
        config=config,
        dataset_params=config["dataset_params"],
        transform_factors=config["transform_factors"],
        affine3d=config["affine3d"],
        batch_size=batch_size,
    )
    # D.effective_batch_size = int(D.batch_size / D.plan["samples_per_file"])
# %%
    D.prepare_data()
    D.setup()
    b = D.train_ds[0]
    b['image'].shape




# # %%
#    b = D.valid_ds[1]
#    b['image'].shape
# # %%
#
#    dl = D.train_dataloader()
# # %%
#     iteri = iter(dl)
#     b = next(iteri)
#     im = b['image']
    lm = b['lm']
# %
# SECTION:-------------------- DataManagerSource ------------------------------------------------------------------------------------------------------ <CR> <CR> <CR> <CR> <CR>

# %%
    batch_size = 2
    D = DataManagerSource(
        proj,
        config=config,
        dataset_params=config["dataset_params"],
        transform_factors=config["transform_factors"],
        affine3d=config["affine3d"],
        batch_size=batch_size,
    )
    D.effective_batch_size = int(D.batch_size / D.plan["samples_per_file"])
# %%
# %%
    D.prepare_data()

    D.setup()
    D.data_folder
    b = D.train_ds[0]

    dl = D.train_dataloader()
    iteri = iter(dl)
    b = next(iteri)
    im = b['image']
    lm = b['lm']

# %%
# %%
#SECTION:-------------------- LBD--------------------------------------------------------------------------------------

# %%
    batch_size = 2
    D = DataManagerLBD(
        proj,
        config=config,
        dataset_params=config["dataset_params"],
        transform_factors=config["transform_factors"],
        affine3d=config["affine3d"],
        batch_size=batch_size,
    )
    D.effective_batch_size = int(D.batch_size / D.plan["samples_per_file"])
# %%
    D.prepare_data()
    D.setup()
    D.data_folder
    b = D.train_ds[0]

    dl = D.train_dataloader()
    iteri = iter(dl)
    b = next(iteri)
    im = b['image']
    lm = b['lm']

# %%



# %%
# SECTION:-------------------- Patch-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR>

    batch_size = 2
    D = DataManagerPatch(
        proj,
        config=config,
        dataset_params=config["dataset_params"],
        transform_factors=config["transform_factors"],
        affine3d=config["affine3d"],
        batch_size=batch_size,
    )

    D.prepare_data()
    D.setup()
# %%
    dl = D.train_dataloader()
    iteri = iter(dl)
    b = next(iteri)
    im = b['image']
    lm = b['lm']
# %%
    for i, dd in enumerate(D.train_ds):
        print(i)

# %%
    cids = D.train_cids
    patches_per_id = []

# %%
    dici = D.data_train[0]
    D.transforms_dict.keys()
    D.transforms_dict[""](dici)
    dici = RD(dici)
    D.tfms_train
    dici = DP(dici)
# %%
    RD = RandomPatch()
    dici2 = RD(dici)
    D.setup()
# %%

    cid = D.train_cids[0]

    bboxes = D.get_patch_files(D.bboxes, cid)
    bboxes.append(D.get_label_info(bboxes))
    D.bboxes_per_id.append(bboxes)
# %%
# %%
# SECTION:-------------------- ROUGH-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR>

# %%

    E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
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
        num_samples=D.plan["samples_per_file"],
        lazy=True,
        allow_smaller=False,
    )
    Ld = LoadTorchDict(keys=["indices"], select_keys=["lm_fg_indices", "lm_bg_indices"])

    Rva = RandCropByPosNegLabeld(
        keys=["image", "lm"],
        label_key="lm",
        image_key="image",
        image_threshold=-2600,
        fg_indices_key="lm_fg_indices",
        bg_indices_key="lm_bg_indices",
        spatial_size=D.dataset_params["patch_size"],
        pos=1,
        neg=1,
        num_samples=D.plan["samples_per_file"],
        lazy=True,
        allow_smaller=True,
    )
    Re = ResizeWithPadOrCropd(
        keys=["image", "lm"],
        spatial_size=D.dataset_params["patch_size"],
        lazy=True,
    )

    L = LoadTorchd(keys=["image", "lm"])
# %%
    D.prepare_data()
    D.setup(None)
# %%
    D.valid_ds[7]

    keys_val = "L,Ld,E,Rva,Re,N"
    dici = D.valid_ds.data[7]
    dici = L(dici)
    dici = Ld(dici)
    dici = E(dici)
    dici = Rva(dici)
    dici = Re(dici)

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
