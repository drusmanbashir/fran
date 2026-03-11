# %%
from __future__ import annotations

import ast
import os
import re
from functools import reduce
from operator import add
from pathlib import Path
from typing import Optional, Tuple

import ipdb
import numpy as np
import pandas as pd
import torch
from fastcore.basics import listify, operator, warnings
from lightning import LightningDataModule
from lightning.pytorch import LightningDataModule
from monai.config.type_definitions import KeysCollection
from monai.data import DataLoader, Dataset, GridPatchDataset, PatchIterd
from monai.data.dataset import CacheDataset, LMDBDataset, PersistentDataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (RandCropByPosNegLabeld,
                                                 RandSpatialCropSamplesD,
                                                 ResizeWithPadOrCropd,
                                                 SpatialPadd)
from monai.transforms.intensity.dictionary import (RandAdjustContrastd,
                                                   RandScaleIntensityd,
                                                   RandShiftIntensityd)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd, RandAffined, RandFlipd, Resized, Spacingd
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.utility.dictionary import (EnsureChannelFirstd,
                                                 MapLabelValued, ToDeviceD)
from torchvision.datasets.folder import is_image_file
from tqdm.auto import tqdm as pbar
from utilz.cprint import cprint
from utilz.fileio import is_sitk_file, load_dict, load_yaml
from utilz.helpers import (find_matching_fn, multiprocess_multiarg,
                           project_title_from_folder, resolve_device)
from utilz.imageviewers import ImageMaskViewer
from utilz.stringz import (ast_literal_eval, headline, info_from_filename,
                           strip_extension)

from fran.configs.parser import ConfigMaker, is_excel_None
from fran.data.collate import (as_is_collated, grid_collated, patch_collated,
                               source_collated, whole_collated)
from fran.data.dataset import NormaliseClipd, fg_in_bboxes
from fran.managers.project import Project
from fran.preprocessing.helpers import bbox_bg_only, compute_fgbg_ratio
from fran.transforms.imageio import LoadSITKd, LoadTorchd, TorchReader
from fran.transforms.intensitytransforms import RandRandGaussianNoised
from fran.transforms.misc_transforms import (DummyTransform, LoadTorchDict,
                                             MetaToDict)
from fran.managers.data.training import DataManager
from fran.utils.common import PAD_VALUE
from fran.utils.folder_names import folder_names_from_plan
from fran.utils.misc import convert_remapping

common_vars_filename = os.environ["FRAN_CONF"] + "/config.yaml"
COMMON_PATHS = load_yaml(common_vars_filename)


class DataManagerTestFF(DataManager):
    def __init__(
        self, project, configs, batch_size, device, data_folder, keys, collate_fn, remapping_gt=None
    ):
        self.project=project

        self.configs = configs
        self.batch_size = batch_size
        self.device = device
        self.keys = keys

        self.set_data_folder(data_folder)
        self.set_collate_fn(collate_fn)
        self.set_preprocessing_params()
        self.plan = self.configs["plan_train"]
        if remapping_gt is not None:
            self.plan["remapping_train"] = remapping_gt
        else:
            self.plan["remapping_train"] =None



    def set_data_folder(self,data_folder):
        self.data_folder = Path(data_folder)

    def prepare_data(self):
        images_folder = self.data_folder/"images"
        lms_folder = self.data_folder/"lms"
        images = [fn for fn in images_folder.glob("*") if is_image_file(fn.name)]

        self.data = []
        for img_fn  in images:
            lm_fn = find_matching_fn(img_fn.name, lms_folder, ["case_id"],allow_multiple_matches=False)[0]

            dici = {"image":img_fn, "lm":lm_fn}
            self.data.append(dici)
        sample_file  = self.cases[0]
        self.file_type = "nifti" if is_sitk_file(sample_file) else "pt"

    def create_transforms(self):
        if self.file_type == "nifti":
            self.O= Orientationd(keys=["image", "lm"], axcodes="RPS")
            self.L= LoadSITKd(
                keys=["image", "lm"],
                image_only=True,
                ensure_channel_first=False,
                simple_keys=True,)
            self.transforms_dict["L"]  = self.L
            self.transforms_dict["O"]  = self.O
            self.keys= "L,E,S,N,Remap,ResizeP"  # experimental
        else:
            self.keys= "L,O, E,N,Remap"  # Remap is a dummy transform unless self.plan_train specifies it



    def _create_nifti_transform(self):
        spacing = self.plan["spacing"]
        self.transforms_dict = {
            "L": LoadSITKd(
                keys=["image", "lm"],
                image_only=True,
                ensure_channel_first=False,
                simple_keys=True,
            ),
            "E": EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel"),
            "S": Spacingd(keys=["image", "lm"], pixdim=spacing),
            "N": NormaliseClipd(
                keys=["image"],
                clip_range=self.dataset_params["intensity_clip_range"],
                mean=self.dataset_params["mean_fg"],
                std=self.dataset_params["std_fg"],
            ),
            "O": Orientationd(keys=["image", "lm"], axcodes="RPS"),  # nOTE RPS
        }


    def setup(self):
        headline(f"Setting up test/valid dataset")
        print("Src Dims: ", self.configs["dataset_params"]["src_dims"])
        print("Patch Size: ", self.plan["patch_size"])
        keys_test="L,E,N,Remap,ResizeP"
        self.create_transforms()
        self.set_transforms(self.keys)




