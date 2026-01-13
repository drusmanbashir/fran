# %%
import os
import re
from functools import reduce
from operator import add
from pathlib import Path
from typing import Optional, Union

import ipdb
import numpy as np
import torch
from fastcore.basics import listify, operator, store_attr, warnings
from lightning import LightningDataModule
from monai.data import DataLoader, Dataset, GridPatchDataset, PatchIterd
from monai.data.dataset import CacheDataset, LMDBDataset, PersistentDataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (RandCropByPosNegLabeld,
                                                 RandSpatialCropSamplesD,
                                                 ResizeWithPadOrCropd)
from monai.transforms.intensity.dictionary import (RandAdjustContrastd,
                                                   RandScaleIntensityd,
                                                   RandShiftIntensityd)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, Resized
from monai.transforms.transform import RandomizableTransform
from monai.transforms.utility.dictionary import (EnsureChannelFirstd,
                                                 MapLabelValued, ToDeviceD)
from tqdm.auto import tqdm as pbar
from utilz.fileio import load_dict, load_yaml
from utilz.helpers import (find_matching_fn, folder_name_from_list,
                           resolve_device)
from utilz.imageviewers import ImageMaskViewer
from utilz.string import ast_literal_eval, info_from_filename, strip_extension

from fran.configs.parser import ConfigMaker, is_excel_None
from fran.data.collate import grid_collated, source_collated, whole_collated
from fran.data.dataset import NormaliseClipd, fg_in_bboxes
from fran.managers.db import find_matching_plan
from fran.managers.project import Project
from fran.preprocessing.helpers import bbox_bg_only
from fran.transforms.imageio import LoadTorchd, TorchReader
from fran.transforms.intensitytransforms import RandRandGaussianNoised
from fran.transforms.misc_transforms import (DummyTransform, LoadTorchDict,
                                             MetaToDict)
from fran.utils.folder_names import folder_names_from_plan

common_vars_filename = os.environ["FRAN_COMMON_PATHS"] + "/config.yaml"
COMMON_PATHS = load_yaml(common_vars_filename)

tr = ipdb.set_trace

from fran.managers.data.training import (DataManager, DataManagerMulti,
                                         DataManagerLBD, DataManagerSource)


class DataManagerDual2( DataManagerMulti):

    def __init__(
        self,
        project_title,
        configs: dict,
        batch_size: int,
        cache_rate=0.0,
        device="cuda",
        ds_type=None,
        save_hyperparameters=True,
        keys_tr="L,Remap,Ld,E,N,Rtr,F1,F2,Affine,ResizePC,IntensityTfms",
        keys_val="L,N,Remap,Ld,E,ResizePC",
        keys_val2="L,N,Remap,Ld,E,ResizePC",
        data_folder: Optional[str | Path] = None,
    ):

        super().__init__(
                    project_title=project_title,
                    configs=configs,
                    batch_size=batch_size,
                    cache_rate=cache_rate,
                    device=device,
                    ds_type=ds_type,
                    save_hyperparameters=save_hyperparameters,
                    keys_tr=keys_tr,
                    keys_val=keys_val,
                    data_folder=data_folder)
        self.keys_val2 = keys_val2


    def prepare_data (self):
        super().prepare_data()
        _, manager_class_valid = self.infer_manager_classes(self.configs)
        self.valid_manager2 = manager_class_valid(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            device=self.device,
            ds_type=None,
            split='valid',
            keys=self.keys_val2,
            data_folder=self.data_folder
        )
        self.valid_manager2.prepare_data()

    def setup(self, stage: str):
        super().setup(stage)
        self.valid_manager2.setup(stage)

    def val_dataloader(self):
        return [self.valid_manager.dl, self.valid_manager2.dl]
        


class DataManager2(DataManager):
    # this has 2 validation dataloaders, one is regular, the other is grid-based and allows comparison across varying plans

    def _create_valid_ds(self) -> Dataset:
        """
        valid-ds is a GridPatchDataset to make training runs comparable
        """
        ds1 = PersistentDataset(
            data=self.data,
            transform=self.transforms,
            cache_dir=self.cache_folder,
        )

        # patch_iter = PatchIterd(keys =['image','lm'], patch_size=  self.plan['patch_size'] )
        # ds = GridPatchDataset(data=ds1 ,patch_iter=patch_iter)
        return ds1

    def create_dataloader(self):

        dl1 = DataLoader(
            self.ds,
            batch_size=self.effective_batch_size,
            num_workers=self.effective_batch_size * 2,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )
        if self.split == "train":
            self.dl = dl1
        else:
            patch_iter = PatchIterd(
                keys=["image", "lm"], patch_size=self.plan["patch_size"]
            )
            ds2 = GridPatchDataset(data=self.ds, patch_iter=patch_iter)
            dl2 = DataLoader(
                ds2,
                batch_size=self.effective_batch_size,
                num_workers=self.effective_batch_size * 2,
                collate_fn=self.collate_fn,
                persistent_workers=True,
                pin_memory=True,
            )
            self.dl = [dl1, dl2]
