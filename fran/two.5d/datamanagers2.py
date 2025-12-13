# %%
"""Slice-based DataManagers (3-slice input)

This module is the slice-dataset counterpart to training.py's volume DataManagers.

Key property: each sample is built by selecting a center z and returning 3 contiguous
slices (z-1, z, z+1) for both image and label via ExtractContiguousSlicesd.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from operator import add
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from fastcore.basics import listify, operator, store_attr
from lightning import LightningDataModule
from monai.data import DataLoader, Dataset
from monai.data.dataset import CacheDataset, LMDBDataset, PersistentDataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (
    RandSpatialCropSamplesD,
    ResizeWithPadOrCropd,
)
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, Resized
from monai.transforms.transform import RandomizableTransform
from monai.transforms.utility.dictionary import (
    EnsureChannelFirstd,
    MapLabelValued,
    ToDeviceD,
)

from fran.configs.parser import is_excel_None
from fran.managers.project import Project
from fran.transforms.imageio import TorchReader
from fran.transforms.intensitytransforms import RandRandGaussianNoised
from fran.transforms.spatialtransforms import ExtractContiguousSlicesd
from fran.transforms.misc_transforms import DummyTransform
from utilz.fileio import load_dict, load_yaml
from tqdm.auto import tqdm as pbar, resolve_device
from utilz.string import ast_literal_eval, strip_extension

import os


# --------------------------------------------------------------------------------------
# Collation for slice-based datasets
# --------------------------------------------------------------------------------------

def _process_items(items: Iterable[Dict[str, Any]]):
    data: Dict[str, List[Any]] = defaultdict(list)
    filenames: Dict[str, List[Any]] = defaultdict(list)

    for item in items:
        for key, value in item.items():
            data[key].append(value)
            try:
                filenames[key].append(value.meta.get("filename_or_obj", None))
            except Exception:
                filenames[key].append(None)

    return dict(data), dict(filenames)


def source_collated(batch: List[Any]) -> Dict[str, Any]:
    """Collate for datasets where __getitem__ returns a *list* of dicts.

    ExtractContiguousSlicesd / RandSpatialCropSamplesD may produce lists of samples per case.
    This collate flattens those lists and stacks tensors.
    """
    all_data: Dict[str, List[Any]] = defaultdict(list)
    all_filenames: Dict[str, List[Any]] = defaultdict(list)

    for item in batch:
        data, filenames = _process_items(item)
        for k, v in data.items():
            all_data[k].extend(v)
        for k, v in filenames.items():
            all_filenames[k].extend(v)

    output: Dict[str, Any] = {}
    for key, values in all_data.items():
        if all(isinstance(x, torch.Tensor) for x in values):
            stacked = torch.stack(values, 0)
            # attach filenames if available
            if hasattr(stacked, "meta"):
                stacked.meta["filename_or_obj"] = (
                    all_filenames[key][0] if len(batch) == 1 else all_filenames[key]
                )
            output[key] = stacked
        else:
            output[key] = values
    return output


# --------------------------------------------------------------------------------------
# Common paths (matches training.py behaviour)
# --------------------------------------------------------------------------------------

_common_cfg = os.environ.get("FRAN_COMMON_PATHS")
if _common_cfg is None:
    raise RuntimeError("Environment variable FRAN_COMMON_PATHS is not set")
COMMON_PATHS = load_yaml(str(Path(_common_cfg) / "config.yaml"))


def int_to_ratios(n_fg_labels: int, fgbg_ratio: float = 3.0) -> List[float]:
    ratios = [1.0] + [fgbg_ratio / n_fg_labels] * n_fg_labels
    return ratios


def list_to_fgbg(class_ratios: List[float]) -> Tuple[float, float]:
    bg = class_ratios[0]
    fg = reduce(add, class_ratios[1:], 0.0)
    return fg, bg


# --------------------------------------------------------------------------------------
# DataManagers
# --------------------------------------------------------------------------------------

class DataManagerDual(LightningDataModule):
    """A higher-level DataManager that manages separate training and validation DataManagers."""

    def __init__(
        self,
        project_title: str,
        configs: dict,
        batch_size: int,
        cache_rate: float = 0.0,
        device: str = "cuda",
        ds_type: Optional[str] = None,
        save_hyperparameters: bool = True,
        keys_tr: str = "Ex,L,E,Remap,Affine,F1,F2,ResizePC,N,IntensityTfms",
        keys_val: str = "Ex,L,E,Remap,ResizePC,N",
        data_folder: Optional[str | Path] = None,
    ):
        super().__init__()
        self.project = Project(project_title)
        self.configs = configs
        self.batch_size = batch_size
        self.cache_rate = cache_rate
        self.device = device
        self.ds_type = ds_type
        self.keys_tr = keys_tr
        self.keys_val = keys_val
        self.data_folder = Path(data_folder) if data_folder is not None else None

        if save_hyperparameters:
            self.save_hyperparameters("project_title", "configs", logger=False)

    def prepare_data(self):
        manager_class_train, manager_class_valid = self.infer_manager_classes(self.configs)

        self.train_manager = manager_class_train(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="train",
            device=self.device,
            ds_type=self.ds_type,
            keys=self.keys_tr,
            data_folder=self.data_folder,
        )

        self.valid_manager = manager_class_valid(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="valid",
            device=self.device,
            ds_type=None,
            keys=self.keys_val,
            data_folder=self.data_folder,
        )

        self.train_manager.prepare_data()
        self.valid_manager.prepare_data()

    def setup(self, stage: Optional[str] = None):
        self.train_manager.setup(stage)
        self.valid_manager.setup(stage)

    def train_dataloader(self):
        return self.train_manager.dl

    def val_dataloader(self):
        return self.valid_manager.dl

    @property
    def train_ds(self):
        return self.train_manager.ds

    @property
    def valid_ds(self):
        return self.valid_manager.ds

    def infer_manager_classes(self, configs: dict):
        train_mode = configs["plan_train"]["mode"]
        valid_mode = configs["plan_valid"]["mode"]
        assert train_mode == valid_mode, f"Train mode '{train_mode}' and valid mode '{valid_mode}' must match"

        mode_to_class = {
            "source": DataManagerSource,
            "whole": DataManagerWhole,
            "lbd": DataManagerLBD,
            "pbd": DataManagerWID,
        }
        if train_mode not in mode_to_class:
            raise ValueError(f"Unrecognized mode: {train_mode}. Must be one of {list(mode_to_class.keys())}")
        return mode_to_class[train_mode], mode_to_class[valid_mode]


class DataManager(LightningDataModule):
    """Base class for slice-based datasets.

    Expected folder structure (data_folder):
      images/<case_id>/*.pt      (per-slice tensors)
      lms/<case_id>/*.pt         (per-slice label tensors, same naming/order convention)

    Each case returns 3 contiguous slices via ExtractContiguousSlicesd.
    """

    def __init__(
        self,
        project: Project,
        configs: dict,
        batch_size: int = 8,
        cache_rate: float = 0.0,
        device: str = "cuda:0",
        ds_type: Optional[str] = None,
        split: str = "train",
        save_hyperparameters: bool = False,
        keys: Optional[str] = None,
        data_folder: Optional[str | Path] = None,
    ):
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters("project", "configs", "split", logger=False)

        device = resolve_device(device)
        store_attr()

        self.plan = configs[f"plan_{split}"]
        self.dataset_params = configs["dataset_params"]
        self.transform_factors = configs.get("transform_factors", {})
        self.affine3d = configs.get("affine3d", None)

        self.device = device
        self.keys = keys or ("Ex,L,E,Remap,Affine,F1,F2,ResizePC,N,IntensityTfms" if split == "train" else "Ex,L,E,Remap,ResizePC,N")

        self.set_effective_batch_size()

        if data_folder is None:
            self.data_folder = self.derive_data_folder(mode=self.plan["mode"])
        else:
            self.data_folder = Path(data_folder)
            assert self.data_folder.is_dir(), f"Dataset folder {self.data_folder} does not exist or is not a directory"

        self.assimilate_tfm_factors(self.transform_factors)
        self.set_collate_fn()

    # ----- required overrides -----

    def set_collate_fn(self):
        raise NotImplementedError

    # ----- transforms -----

    def assimilate_tfm_factors(self, transform_factors: dict):
        for key, value in transform_factors.items():
            # expected [value, prob]
            setattr(self, key, {"value": value[0], "prob": value[1]})

    def create_transforms(self):
        include = [k.strip() for k in self.keys.split(",") if k.strip()]

        # Patch size for 2D ops
        patch_xy = self.plan.get("patch_size", None)
        if patch_xy is None:
            raise KeyError("plan['patch_size'] is required")
        patch_xy = list(patch_xy)[:2]

        tfms: List[Any] = []

        # 1) Extract 3-slice sample from case folders
        if "Ex" in include:
            tfms.append(ExtractContiguousSlicesd(keys=["image_fns", "lm_fldr", "n_slices"]))

        # 2) Load selected slices (TorchReader supports .pt)
        if "L" in include:
            L = LoadImaged(keys=["image", "lm"], image_only=True, ensure_channel_first=False, simple_keys=True)
            L.register(TorchReader())
            tfms.append(L)

        # 3) Optional remapping on label
        if "Remap" in include:
            if not is_excel_None(self.plan.get("remapping_train", None)) and self.split == "train":
                remap = self.plan["remapping_train"]
            elif not is_excel_None(self.plan.get("remapping_valid", None)) and self.split != "train":
                remap = self.plan["remapping_valid"]
            else:
                remap = None

            if remap is not None:
                assert isinstance(remap, (tuple, list)) and len(remap) == 2, "remapping must be a (orig_labels, target_labels) pair"
                tfms.append(MapLabelValued(keys=["lm"], orig_labels=remap[0], target_labels=remap[1]))
            else:
                tfms.append(DummyTransform(keys=["lm"]))

        # 4) Channel first
        if "E" in include:
            tfms.append(EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel"))

        # 5) Spatial augmentation
        if "F1" in include:
            prob = getattr(self, "flip", {"prob": 0.0})["prob"]
            tfms.append(RandFlipd(keys=["image", "lm"], prob=prob, spatial_axis=0, lazy=True))
        if "F2" in include:
            prob = getattr(self, "flip", {"prob": 0.0})["prob"]
            tfms.append(RandFlipd(keys=["image", "lm"], prob=prob, spatial_axis=1, lazy=True))

        if "Affine" in include:
            if self.affine3d is None:
                raise KeyError("configs['affine3d'] is required when using 'Affine'")
            tfms.append(
                RandAffined(
                    keys=["image", "lm"],
                    mode=["bilinear", "nearest"],
                    prob=self.affine3d["p"],
                    rotate_range=self.affine3d["rotate_range"],
                    scale_range=self.affine3d["scale_range"],
                )
            )

        # 6) Crop / pad or resize to patch size
        if "ResizePC" in include:
            tfms.append(ResizeWithPadOrCropd(keys=["image", "lm"], spatial_size=patch_xy, lazy=True))
        if "ResizeW" in include:
            tfms.append(Resized(keys=["image", "lm"], spatial_size=patch_xy, mode=["linear", "nearest"], lazy=True))
        if "Crop" in include:
            # training-time random crop samples; returns list of dicts (num_samples)
            num_samples = int(self.plan.get("samples_per_file", 1))
            tfms.append(RandSpatialCropSamplesD(keys=["image", "lm"], roi_size=patch_xy, num_samples=num_samples, lazy=True))

        # 7) Normalisation / intensity
        if "N" in include:
            # require global properties from Project
            gp = load_dict(self.project.global_properties_filename)
            clip_range = gp["intensity_clip_range"]
            mean_fg = gp["mean_fg"]
            std_fg = gp["std_fg"]
            from fran.data.dataset import NormaliseClipd  # local import avoids circulars
            tfms.append(NormaliseClipd(keys=["image"], clip_range=clip_range, mean=mean_fg, std=std_fg))

        if "IntensityTfms" in include:
            scale = getattr(self, "scale", {"value": 0.0, "prob": 0.0})
            noise = getattr(self, "noise", {"value": 0.0, "prob": 0.0})
            shift = getattr(self, "shift", {"value": 0.0, "prob": 0.0})
            contrast = getattr(self, "contrast", {"value": 1.0, "prob": 0.0})
            tfms.extend(
                [
                    RandScaleIntensityd(keys="image", factors=scale["value"], prob=scale["prob"]),
                    RandRandGaussianNoised(keys=["image"], std_limits=noise["value"], prob=noise["prob"]),
                    RandShiftIntensityd(keys="image", offsets=shift["value"], prob=shift["prob"]),
                    RandAdjustContrastd(["image"], gamma=contrast["value"], prob=contrast["prob"]),
                ]
            )

        # 8) Move to device (after tensor creation); keep optional to avoid pin_memory issues
        if "Dev" in include:
            tfms.append(ToDeviceD(keys=["image", "lm"], device=self.device))

        self.transforms = Compose(tfms)

    # ----- batching / datasets -----

    def set_effective_batch_size(self):
        if self.split != "train" or "samples_per_file" not in self.plan:
            self.plan["samples_per_file"] = 1
        self.effective_batch_size = int(max(1, self.batch_size / self.plan["samples_per_file"]))

    def prepare_data(self):
        if not hasattr(self, "cases"):
            self.cases_from_project_split()
        self.data = self.create_data_dicts(self.cases)

    def cases_from_project_split(self):
        train_cases, valid_cases = self.project.get_train_val_files(self.dataset_params["fold"], self.plan["datasources"])
        self.cases = train_cases if self.split == "train" else valid_cases
        assert len(self.cases) > 0, "There are no cases, aborting!"

    def create_data_dicts(self, cases: List[str]) -> List[Dict[str, Any]]:
        cases = [strip_extension(fn) for fn in cases]
        image_case_folders = list((self.data_folder / "images").glob("*"))
        lm_case_folders = list((self.data_folder / "lms").glob("*"))

        data: List[Dict[str, Any]] = []
        for case_id in pbar(cases):
            img_matches = [fn for fn in image_case_folders if case_id == fn.name]
            lm_matches = [fn for fn in lm_case_folders if case_id == fn.name]
            assert len(img_matches) == 1 and len(lm_matches) == 1, f"Expected exactly one image and lm folder for case {case_id}"
            img_fldr = img_matches[0]
            lm_fldr = lm_matches[0]
            img_fns = sorted(list(img_fldr.glob("*")))
            n_slices = len(img_fns)
            data.append({"image_fns": img_fns, "lm_fldr": lm_fldr, "n_slices": n_slices})
        return data

    def setup(self, stage: Optional[str] = None):
        self.create_transforms()
        self.create_dataset()
        self.create_dataloader()

    def create_dataset(self):
        if not hasattr(self, "data") or len(self.data) == 0:
            raise RuntimeError("No data prepared; call prepare_data() before setup().")

        if self.split == "train":
            self.ds = self._create_train_ds()
        else:
            self.ds = self._create_valid_ds()

    def _create_train_ds(self):
        if is_excel_None(self.ds_type):
            return Dataset(data=self.data, transform=self.transforms)
        if self.ds_type == "cache":
            return CacheDataset(data=self.data, transform=self.transforms, cache_rate=self.cache_rate)
        if self.ds_type == "lmdb":
            return LMDBDataset(data=self.data, transform=self.transforms, cache_dir=self.cache_folder, db_name=f"{self.split}_cache")
        raise NotImplementedError(f"Unknown ds_type: {self.ds_type}")

    def _create_valid_ds(self):
        # PersistentDataset provides deterministic caching without loading everything into RAM
        return PersistentDataset(data=self.data, transform=self.transforms, cache_dir=self.cache_folder)

    def create_dataloader(self):
        self.dl = DataLoader(
            self.ds,
            batch_size=self.effective_batch_size,
            num_workers=max(1, self.effective_batch_size * 2),
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )

    # ----- filesystem -----

    def derive_data_folder(self, mode: str) -> Path:
        raise NotImplementedError

    @property
    def cache_folder(self) -> Path:
        parent_folder = Path(COMMON_PATHS["cache_folder"]) / self.project.project_title
        return parent_folder / (self.data_folder.name + "_slices")


class DataManagerSource(DataManager):
    def set_collate_fn(self):
        self.collate_fn = source_collated

    def derive_data_folder(self, mode: str) -> Path:
        assert mode == "source", f"Dataset mode must be 'source' for DataManagerSource, got '{mode}'"
        prefix = "spc"
        spacing = self.plan["spacing"]
        parent_folder = self.project.fixed_spacing_folder
        from utilz.helpers import folder_name_from_list
        return folder_name_from_list(prefix, parent_folder, spacing)


class DataManagerWhole(DataManagerSource):
    # same dataset structure; typically different plan/data_folder selection
    pass


class DataManagerLBD(DataManagerSource):
    def derive_data_folder(self, mode: str) -> Path:
        assert mode == "lbd", f"Dataset mode must be 'lbd' for DataManagerLBD, got '{mode}'"
        spacing = self.plan.get("spacing")
        if isinstance(spacing, str):
            spacing = ast_literal_eval(spacing)
        parent_folder = self.project.lbd_folder
        folder_suffix = "plan" + str(self.dataset_params.get(f"plan_{self.split}", ""))
        from utilz.helpers import folder_name_from_list
        data_folder = folder_name_from_list(prefix="spc", parent_folder=parent_folder, values_list=spacing, suffix=folder_suffix)
        assert data_folder.exists(), f"Dataset folder {data_folder} does not exist"
        return data_folder


class DataManagerWID(DataManagerSource):
    def derive_data_folder(self, mode: str) -> Path:
        assert mode in ["pbd", "wid"], f"Dataset mode must be 'pbd' for DataManagerWID, got '{mode}'"
        spacing = self.plan.get("spacing")
        if isinstance(spacing, str):
            spacing = ast_literal_eval(spacing)
        parent_folder = self.project.pbd_folder
        folder_suffix = "plan" + str(self.dataset_params.get(f"plan_{self.split}", ""))
        from utilz.helpers import folder_name_from_list
        data_folder = folder_name_from_list(prefix="spc", parent_folder=parent_folder, values_list=spacing, suffix=folder_suffix)
        assert data_folder.exists(), f"Dataset folder {data_folder} does not exist"
        return data_folder
