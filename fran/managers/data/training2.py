# %%
from __future__ import annotations

import ast
import json
import os
import shutil
from functools import reduce
from operator import add
from pathlib import Path
from typing import Optional, Tuple

import ipdb
import numpy as np
import pandas as pd
import torch
from fastcore.basics import listify, operator
from fran.configs.parser import is_excel_None
from fran.data.collate import patch_collated, source_collated, whole_collated
from fran.data.dataset import NormaliseClipd
from fran.managers.project import Project
from fran.preprocessing.helpers import bbox_bg_only, compute_fgbg_ratio, import_h5py
from fran.transforms.imageio import SimpleTorchLoader, TorchReader
from fran.transforms.intensitytransforms import RandRandGaussianNoised
from fran.transforms.misc_transforms import DummyTransform, LoadTorchDict, MetaToDict
from fran.transforms.batch_affine import BatchRandAffined3D
from fran.utils.folder_names import FolderNames
from fran.utils.misc import convert_remapping
from lightning import LightningDataModule
from lightning.pytorch import LightningDataModule
from monai.config.type_definitions import KeysCollection
from monai.data import DataLoader, Dataset, GridPatchDataset, PatchIterd
from monai.data.dataset import LMDBDataset
from monai.data.meta_tensor import MetaTensor
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (
    RandCropByPosNegLabeld,
    ResizeWithPadOrCropd,
    SpatialPadd,
)
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, Resized
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.utility.dictionary import (
    EnsureChannelFirstd,
    MapLabelValued,
    ToDeviceD,
)
from torch.utils.data import RandomSampler
from tqdm.auto import tqdm as pbar
from utilz.cprint import cprint
from utilz.fileio import load_dict, load_yaml
from utilz.helpers import find_matching_fn, resolve_device
from utilz.stringz import (
    ast_literal_eval,
    headline,
    info_from_filename,
    strip_extension,
)

common_vars_filename = os.environ["FRAN_CONF"] + "/config.yaml"
COMMON_PATHS = load_yaml(common_vars_filename)

tr = ipdb.set_trace


def _is_grid_patch_padded(coords, original_spatial_shape) -> bool:
    """
    True when a grid patch goes outside original image bounds and required padding.
    """
    if coords is None or original_spatial_shape is None:
        return False

    coords_arr = np.asarray(coords)
    shape_arr = np.asarray(original_spatial_shape)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        return False
    if coords_arr.shape[0] == shape_arr.shape[0] + 1:
        spatial_coords = coords_arr[1:]
    else:
        spatial_coords = coords_arr[: shape_arr.shape[0]]

    starts = spatial_coords[:, 0]
    stops = spatial_coords[:, 1]
    return bool(np.any(starts < 0) or np.any(stops > shape_arr))


class PatchIterdWithPaddingFlag:
    """
    Shim around MONAI PatchIterd that annotates each yielded patch dictionary
    with `is_padded` based on coords vs original spatial shape.
    """

    def __init__(self, base_patch_iter: PatchIterd):
        self.base_patch_iter = base_patch_iter

    def __call__(self, data):
        if isinstance(data, (list, tuple)):
            inputs = data
        else:
            inputs = (data,)

        for item in inputs:
            for patch_dict, coords in self.base_patch_iter(item):
                d = dict(patch_dict)
                d["is_padded"] = _is_grid_patch_padded(
                    coords=d.get("patch_coords", coords),
                    original_spatial_shape=d.get("original_spatial_shape"),
                )
                yield d, coords


def int_to_ratios(n_fg_labels, fgbg_ratio=3):
    ratios = [1] + [fgbg_ratio / n_fg_labels] * n_fg_labels
    return ratios


def list_to_fgbg(class_ratios):
    bg = class_ratios[0]
    fg = class_ratios[1:]
    fg = reduce(add, fg)
    return fg, bg


class RandomPatch(MapTransform, RandomizableTransform):
    """
    to be used by DataManagerPatch
    """

    def randomize(self, data=None):
        n_patches = data["n_patches"]
        self.indx = self.R.randint(0, n_patches)

    def __call__(self, data: list):
        self.randomize(data)
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = data[key][self.indx]
        return d


class LoadHDF5ShardIndexd(MapTransform):
    """
    Resolve case -> HDF5 shard and load fg/bg flat indices for sampling.
    """

    def __init__(
        self,
        keys: KeysCollection,
        manifest_rel_path: str = "hdf5_shards/src_192_192_128/manifest.json",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.manifest_rel_path = Path(manifest_rel_path)
        self._manifest_cache = {}

    def _cached_manifest(self, data_folder: Path):
        folder_key = str(data_folder)
        cached = self._manifest_cache.get(folder_key)
        if cached is not None:
            return cached

        manifest_fn = data_folder / self.manifest_rel_path
        with open(manifest_fn, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        case_to_shard: dict[str, str] = {}
        for shard_info in manifest["shards"]:
            shard_name = shard_info["shard"]
            shard_path = Path(shard_name)
            if not shard_path.is_absolute():
                shard_path = manifest_fn.parent / shard_path
            for case_id in shard_info["case_ids"]:
                case_to_shard[str(case_id)] = str(shard_path)

        src_dims = tuple(int(v) for v in manifest.get("src_dims", (192, 192, 128)))
        cached = {
            "case_to_shard": case_to_shard,
            "src_dims": src_dims,
        }
        self._manifest_cache[folder_key] = cached
        return cached

    def __call__(self, data):
        d = dict(data)
        case_id = str(d["case_id"])
        data_folder = Path(d["data_folder"])
        manifest = self._cached_manifest(data_folder)
        shard_path = manifest["case_to_shard"][case_id]

        h5py = import_h5py()
        case_path = f"/cases/{case_id}"
        with h5py.File(shard_path, "r") as h5f:
            case_grp = h5f[case_path]
            fg = np.asarray(case_grp["lm_fg_indices"][:], dtype=np.int64).reshape(-1)
            bg = np.asarray(case_grp["lm_bg_indices"][:], dtype=np.int64).reshape(-1)
            src_dims = tuple(int(v) for v in case_grp["lm"].shape)

        d["hdf5_shard_path"] = str(shard_path)
        d["hdf5_case_path"] = case_path
        d["src_dims"] = src_dims
        d["lm_fg_indices"] = fg
        d["lm_bg_indices"] = bg
        return d


class RandCropByFlatIndicesd(RandomizableTransform, MapTransform):
    """
    RandCropByPosNegLabeld-like multi-sample output using precomputed flat indices.
    """

    def __init__(
        self,
        keys: KeysCollection,
        roi_size,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        fg_indices_key: str = "lm_fg_indices",
        bg_indices_key: str = "lm_bg_indices",
        src_dims_key: str = "src_dims",
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self)
        self.roi_size = tuple(int(v) for v in roi_size)
        self.pos = float(pos)
        self.neg = float(neg)
        self.num_samples = int(num_samples)
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.src_dims_key = src_dims_key

    def _sample_pool(self, fg: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, bool]:
        choose_fg = self.R.rand() < self.pos / (self.pos + self.neg)
        if (choose_fg and fg.size > 0) or bg.size == 0:
            return fg, True
        return bg, False

    def _compute_crop(
        self, center: tuple[int, ...], src_dims: tuple[int, ...]
    ) -> tuple[tuple[slice, ...], tuple[int, ...], tuple[int, ...]]:
        slices = []
        starts = []
        ends = []
        for c, src_dim, roi_dim in zip(center, src_dims, self.roi_size):
            roi_dim = min(int(roi_dim), int(src_dim))
            start = int(c) - roi_dim // 2
            end = start + roi_dim
            if start < 0:
                start = 0
                end = roi_dim
            if end > src_dim:
                end = int(src_dim)
                start = max(0, end - roi_dim)
            slices.append(slice(start, end))
            starts.append(start)
            ends.append(end)
        return tuple(slices), tuple(starts), tuple(ends)

    def __call__(self, data):
        d = dict(data)
        src_dims = tuple(int(v) for v in d[self.src_dims_key])
        fg = np.asarray(d[self.fg_indices_key], dtype=np.int64).reshape(-1)
        bg = np.asarray(d[self.bg_indices_key], dtype=np.int64).reshape(-1)
        out = []
        for _ in range(self.num_samples):
            sample = dict(d)
            pool, sample_is_fg = self._sample_pool(fg=fg, bg=bg)
            sampled_flat_index = int(pool[self.R.randint(0, pool.size)])
            center = tuple(
                int(v) for v in np.unravel_index(sampled_flat_index, src_dims)
            )
            crop_slices, crop_start, crop_end = self._compute_crop(center, src_dims)
            sample["crop_center"] = center
            sample["crop_slices"] = crop_slices
            sample["crop_start"] = crop_start
            sample["crop_end"] = crop_end
            sample["sample_is_fg"] = bool(sample_is_fg)
            sample["sampled_flat_index"] = sampled_flat_index
            out.append(sample)
        return out


class LoadHDF5Cropd(MapTransform):
    """
    Load only the requested crop from HDF5 shard image/lm datasets.
    """

    def __init__(
        self,
        keys: KeysCollection,
        shard_path_key: str = "hdf5_shard_path",
        case_path_key: str = "hdf5_case_path",
        crop_slices_key: str = "crop_slices",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.shard_path_key = shard_path_key
        self.case_path_key = case_path_key
        self.crop_slices_key = crop_slices_key

    def __call__(self, data):
        d = dict(data)
        shard_path = Path(d[self.shard_path_key])
        case_path = str(d[self.case_path_key])
        crop_slices = tuple(d[self.crop_slices_key])

        h5py = import_h5py()
        with h5py.File(shard_path, "r") as h5f:
            case_grp = h5f[case_path]
            image = np.asarray(case_grp["image"][crop_slices])
            lm = np.asarray(case_grp["lm"][crop_slices])

        filename_or_obj = f"{shard_path}:{case_path}"
        meta = {
            "filename_or_obj": filename_or_obj,
            "case_id": d.get("case_id"),
            "crop_start": d.get("crop_start"),
            "crop_end": d.get("crop_end"),
            "sampled_flat_index": d.get("sampled_flat_index"),
            "sample_is_fg": d.get("sample_is_fg"),
        }
        d["image"] = MetaTensor(torch.as_tensor(image), meta=dict(meta))
        d["lm"] = MetaTensor(torch.as_tensor(lm), meta=dict(meta))
        return d


class DataManagerDual(LightningDataModule):
    """
    Train + valid only.
    """

    def __init__(
        self,
        project_title,
        configs: dict,
        batch_size: int,
        cache_rate=0.0,
        device="cuda",
        ds_type=None,
        save_hyperparameters=True,
        data_folder: Optional[str | Path] = None,
        manager_class_train: Optional[type] = None,
        manager_class_valid: Optional[type] = None,
        train_indices=None,
        val_indices=None,
        val_sampling=1.0,
        debug=False,
        dual_ssd=True,
    ):
        super().__init__()
        self.project = Project(project_title)
        self.configs = configs
        self._batch_size = int(batch_size)
        self.cache_rate = cache_rate
        self.device = device
        self.ds_type = ds_type
        self.keys_tr = None
        self.keys_val = None
        self.data_folder = data_folder if data_folder is not None else None
        self.manager_class_train = manager_class_train
        self.manager_class_valid = manager_class_valid
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.val_sampling = float(val_sampling)
        self.debug = debug
        self.dual_ssd = dual_ssd
        self.batch_affine = self._create_batch_affine()

        if save_hyperparameters:
            self.save_hyperparameters(
                "project_title",
                "configs",
                "train_indices",
                "val_indices",
                "val_sampling",
                "dual_ssd",
                logger=False,
            )

    # ---- core lifecycle -------------------------------------------------

    def prepare_data(self):
        self._build_managers()
        self._call_prepare_data()
        if self.train_indices is not None:
            cprint(
                f"Limiting training dataset size to{self.train_indices}", color="yellow"
            )
            self.train_manager.select_cases_from_inds(self.train_indices)
            self.train_manager.data = self.train_manager.create_staged_data_dicts(
                self.train_manager.cases
            )
            if self.val_indices is None:
                self.val_indices = max(1, int(len(self.train_manager.cases) * 0.2))
        if self.val_indices is not None:
            self.valid_manager.select_cases_from_inds(self.val_indices)
            self.valid_manager.data = self.valid_manager.create_staged_data_dicts(
                self.valid_manager.cases
            )

    def setup(self, stage=None):
        self._call_setup(stage)

    def train_dataloader(self):
        return self.train_manager.dl

    def val_dataloader(self):
        return self.valid_manager.dl

    def _create_batch_affine(self):
        if not self.configs["dataset_params"].get("batch_affine", False):
            return None
        affine3d = self.configs["affine3d"]
        return BatchRandAffined3D(
            keys=["image", "lm"],
            mode=["bilinear", "nearest"],
            prob=affine3d["p"],
            rotate_range=affine3d["rotate_range"],
            scale_range=affine3d["scale_range"],
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        trainer = getattr(self, "trainer", None)
        if trainer is not None and trainer.training and self.batch_affine is not None:
            batch = self.batch_affine(batch)
        return batch

    def state_dict(self) -> dict:
        return {
            "batch_size": int(self._batch_size),
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "val_sampling": float(self.val_sampling),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not state_dict:
            return
        if "batch_size" in state_dict:
            self._batch_size = int(state_dict["batch_size"])
        if "train_indices" in state_dict:
            self.train_indices = state_dict["train_indices"]
        if "val_indices" in state_dict:
            self.val_indices = state_dict["val_indices"]
        if "val_sampling" in state_dict:
            self.val_sampling = float(state_dict["val_sampling"])

    # ---- datasets -------------------------------------------------------

    @property
    def train_ds(self):
        return self.train_manager.ds

    @property
    def valid_ds(self):
        return self.valid_manager.ds

    # ---- batch size propagation ----------------------------------------

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, v: int) -> None:
        v = int(v)
        if v == self._batch_size:
            return
        self._batch_size = v
        for m in self._iter_managers():
            m.batch_size = v
            m.set_effective_batch_size()
            m.create_train_dataloader() if m.split == "train" else m.create_valid_dataloader()

    # ---- internal helpers ----------------------------------------------

    def _iter_managers(self):
        # Dual managers only
        return (self.train_manager, self.valid_manager)

    def _build_managers(self):
        inf_tr, inf_val = self.infer_manager_classes(self.configs)
        cls_tr = self.manager_class_train or inf_tr
        cls_val = self.manager_class_valid or inf_val

        self.train_manager = cls_tr(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="train",
            device=self.device,
            ds_type=self.ds_type,
            keys=self.keys_tr,
            data_folder=self.data_folder,
            debug=self.debug,
            dual_ssd=self.dual_ssd,
        )
        self.valid_manager = cls_val(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="valid",
            device=self.device,
            ds_type=self.ds_type,
            keys=self.keys_val,
            data_folder=self.data_folder,
            val_sampling=self.val_sampling,
            debug=self.debug,
            dual_ssd=self.dual_ssd,
        )

    def _call_prepare_data(self):
        for m in self._iter_managers():
            m.prepare_data()
        self.data_folder = self.train_manager.data_folder

    def _call_setup(self, stage=None):
        for m in self._iter_managers():
            m.setup(stage)

    def infer_manager_classes(self, configs) -> Tuple[type, type]:
        train_mode = configs["plan_train"]["mode"]
        valid_mode = configs["plan_valid"]["mode"]

        mode_to_class = {
            "source": DataManagerSource,
            "whole": DataManagerWhole,
            "pbd": DataManagerPatch,
            "sourcepbd": DataManagerPatch,
            "lbd": DataManagerLBD,
            "kbd": DataManagerKBD,
            "baseline": DataManagerBaseline,
        }

        for mode in (train_mode, valid_mode):
            if mode not in mode_to_class:
                raise ValueError(
                    f"Unrecognized mode: {mode}. Must be one of {list(mode_to_class.keys())}"
                )

        return mode_to_class[train_mode], mode_to_class[valid_mode]


class DataManager(LightningDataModule):
    def __init__(
        self,
        project,
        configs: dict,
        batch_size=8,
        cache_rate=0.0,
        split="train",  # Add sp,lit parameter
        device="cuda:0",
        ds_type=None,
        save_hyperparameters=False,
        keys=None,
        collate_fn=None,
        data_folder: Optional[str | Path] = None,
        val_sampling=1.0,
        debug=False,
        dual_ssd=True,
    ):

        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters(
                "project", "configs", "split", "dual_ssd", logger=False
            )
        device = resolve_device(device)

        self.project = project
        self.configs = configs
        self.batch_size = batch_size
        self.cache_rate = cache_rate
        self.device = device
        self.ds_type = ds_type
        self.split = split
        self.keys = keys
        self.val_sampling = float(val_sampling)
        self.dual_ssd = dual_ssd
        self.set_plan()

        self.maybe_fix_remapping_dtype()
        self.set_preprocessing_params()
        self.set_effective_batch_size()
        self.set_data_folder(data_folder)
        self.set_collate_fn(collate_fn)
        self.debug = debug

    def set_data_folder(self, data_folder):
        if data_folder is None:
            self.data_folder = self.derive_data_folder(plan=self.plan)
        else:
            self.data_folder = Path(data_folder)
            assert self.data_folder.is_dir(), (
                f"Dataset folder {self.data_folder} does not exist or is not a directory"
            )
        # self.data_folder = self.derive_data_folder(mode=self.plan["mode"])

    def select_cases_from_inds(self, inds):
        if isinstance(inds, int):
            self.cases = self.cases[:inds]
        elif isinstance(inds, float):
            inds = int(len(self.cases) * inds)
            self.cases = self.cases[:inds]
        elif isinstance(inds, list | pd.Index):
            if isinstance(inds[0], str):
                cases_final = []
                for case_ in self.cases:
                    fname = case_.split(".")[0]
                    case_id = info_from_filename(fname, full_caseid=True)["case_id"]
                    if case_id in inds:
                        cases_final.append(case_)
                self.cases = cases_final
            else:
                raise NotImplementedError

    def set_preprocessing_params(self):
        global_properties = load_dict(self.project.global_properties_filename)
        self.dataset_params = self.configs["dataset_params"]
        self.dataset_params["intensity_clip_range"] = global_properties[
            "intensity_clip_range"
        ]
        transform_factors = self.configs["transform_factors"]
        self.dataset_params["mean_fg"] = global_properties["mean_fg"]
        self.dataset_params["std_fg"] = global_properties["std_fg"]
        self._assimilate_tfm_factors(transform_factors)

    def maybe_fix_remapping_dtype(self):
        if isinstance(self.plan["remapping_train"], dict):
            self.plan["remapping_train"] = convert_remapping(
                self.plan["remapping_train"]
            )

    def set_plan(self):
        if "train" in self.split:
            plan_str = "plan_train"
        elif "valid" in self.split:
            plan_str = "plan_valid"
        elif "test" in self.split:
            plan_str = "plan_test"
        else:
            raise ValueError(f"Unrecognized split: {self.split}")

        self.plan = self.configs[plan_str]

    def set_collate_fn(self, collate_fn=None):
        if collate_fn is not None:
            self.collate_fn = collate_fn
        else:
            self._set_collate_fn()

    def _set_collate_fn(self):
        raise NotImplementedError

    def __str__(self):
        return "DataManager instance with parameters: " + ", ".join(
            [f"{k}={v}" for k, v in vars(self).items()]
        )

    def __repr__(self):
        return (
            f"DataManager("
            + ", ".join([f"{k}={v}" for k, v in vars(self).items()])
            + ")"
        )

    def _assimilate_tfm_factors(self, transform_factors):
        for key, value in transform_factors.items():
            dici = {"value": value[0], "prob": value[1]}
            setattr(self, key, dici)

    def create_transforms(self):
        """
        Creates and assigns transformations based on provided keys.

        Args:
            keys (str): Comma-separated string of transform keys to include

        Returns:
            None: Sets self.transforms with composed transforms

        Transform Abbreviations and Full Names:
            Dev: EnsureTyped - puts data on GPU
            E: EnsureChannelFirstd - Ensures channel dimension is first
            N: NormaliseClipd - Normalizes and clips image intensities
            RP: RandomPatch - Applies random patch sampling
            F1: RandFlipd (axis=0) - Random flip along spatial axis 0
            F2: RandFlipd (axis=1) - Random flip along spatial axis 1
            IntensityTfms: Collection of intensity transforms:
                - RandScaleIntensityd: Random intensity scaling
                - RandRandGaussianNoised: Random Gaussian noise addition
                - RandShiftIntensityd: Random intensity shifting
                - RandAdjustContrastd: Random contrast adjustment
            Affine: RandAffined - Random affine transformations (rotation, scaling)
            ResizePC: ResizeWithPadOrCropd - Resize with padding or cropping
            ResizeW: Resized - Resize to specified spatial size
            L: LoadImaged - Load image and label data
            Ld: LoadTorchDict - Load torch dictionary with indices
            Ind: MetaToDict - Convert metadata to dictionary format
            Rtr: RandCropByPosNegLabeld (training) - Random crop with positive/negative sampling
            Rva: RandCropByPosNegLabeld (validation) - Random crop for validation
            Z: RandZWindowd - Random z-slice selection this is for LTSM sequences
            None: Sets self.transforms with composed transforms

        """
        # Initialize transforms dictionary and list
        # Parse transform keys
        Dev = ToDeviceD(keys=["image", "lm"], device=self.device)
        E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )

        RP = RandomPatch(keys=["image", "lm"])
        # self.transforms_dict["RP"] = RP

        F1 = RandFlipd(
            keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=0, lazy=True
        )
        # self.transforms_dict["F1"] = F1

        F2 = RandFlipd(
            keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=1, lazy=True
        )
        # self.transforms_dict["F2"] = F2

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
        # self.transforms_dict["IntensityTfms"] = IntensityTfms

        Affine = RandAffined(
            keys=["image", "lm"],
            mode=["bilinear", "nearest"],
            prob=self.configs["affine3d"]["p"],
            rotate_range=self.configs["affine3d"]["rotate_range"],
            scale_range=self.configs["affine3d"]["scale_range"],
        )
        remapping_train = self.plan.get("remapping_train")
        if not is_excel_None(
            remapping_train
        ):  # note this is a very expensive transform
            orig_labels = self.plan["remapping_train"][0]
            dest_labels = self.plan["remapping_train"][1]
            Remap = MapLabelValued(
                keys=["lm"], orig_labels=orig_labels, target_labels=dest_labels
            )
        else:
            Remap = DummyTransform(keys=["lm"])
        ResizePC = ResizeWithPadOrCropd(
            keys=["image", "lm"],
            spatial_size=self.plan["patch_size"],
            lazy=False,
        )

        ResizeP = SpatialPadd(
            keys=["image", "lm"],
            spatial_size=self.plan["patch_size"],
            mode="constant",
            lazy=True,
        )

        ResizeW = Resized(
            keys=["image", "lm"],
            spatial_size=self.plan["patch_size"],
            mode=["linear", "nearest"],
            lazy=True,
        )
        # Continue similarly for the remaining transforms like L, Ld, Ind, Rtr, Rva...

        L = LoadImaged(
            keys=["image", "lm"],
            image_only=True,
            ensure_channel_first=False,
            simple_keys=True,
        )
        L.register(TorchReader())

        # self.transforms_dict["Ld"] = Ld

        Ind = MetaToDict(keys=["lm"], meta_keys=["lm_fg_indices", "lm_bg_indices"])
        # self.transforms_dict["Ind"] = Ind

        src_tag = "_".join(str(int(v)) for v in self.src_dims)
        Ld = LoadHDF5ShardIndexd(
            keys=["case_id", "data_folder"],
            manifest_rel_path=f"hdf5_shards/src_{src_tag}/manifest.json",
        )
        Rtr = RandCropByFlatIndicesd(
            keys=["lm_fg_indices", "lm_bg_indices", "src_dims"],
            roi_size=self.src_dims,
            pos=self.dataset_params["fgbg_ratio"],
            neg=1,
            num_samples=self.plan["samples_per_file"],
        )
        L2 = LoadHDF5Cropd(keys=["hdf5_shard_path", "hdf5_case_path", "crop_slices"])
        Rva = DummyTransform(keys=["image"])

        self.transforms_dict = {
            "Dev": Dev,
            "E": E,
            "N": N,
            "RP": RP,
            "F1": F1,
            "F2": F2,
            "IntensityTfms": IntensityTfms,
            "Affine": Affine,
            "ResizePC": ResizePC,
            "ResizeP": ResizeP,
            "ResizeW": ResizeW,
            "L": L,
            "Ld": Ld,
            "L2": L2,
            "Ind": Ind,
            "Remap": Remap,
            "Rtr": Rtr,
            "Rva": Rva,
        }

        # Conditionally create transforms based on inclusion list

    def set_effective_batch_size(self):
        if (
            not "samples_per_file" in self.plan or not self.split == "train"
        ):  # if split is valid, grid sampling is done and effective batch_size should be same as batch size
            self.plan["samples_per_file"] = 1

        self.effective_batch_size = int(
            np.maximum(1, self.batch_size / self.plan["samples_per_file"])
        )
        print(
            "Given {0} Samples per file and {1} batch_size on the GPU, effective batch size (number of file tensors loaded then sampled for {2} is:\n {3} ".format(
                self.plan["samples_per_file"],
                self.batch_size,
                self.split,
                self.effective_batch_size,
            )
        )

    def set_transforms(self, keys):
        self.transforms = self.tfms_from_dict(keys)

    def tfms_from_dict(self, keys: str):
        keys2 = keys.replace(" ", "")
        keys_list = keys2.split(",")
        tfms = []
        for key in keys_list:
            try:
                if (
                    key == "Affine"
                    and self.split == "train"
                    and self.dataset_params.get("batch_affine", False)
                ):
                    continue
                tfm = self.transforms_dict[key]
                if key == "IntensityTfms":
                    tfms.extend(tfm)
                else:
                    tfms.append(tfm)
            except KeyError as e:
                print("All keys are: ", self.transforms_dict.keys())
                print(f"Transform {key} not found.")
                raise e

        print("{0} transforms are: {1}".format(self.split, tfms))
        tfms = Compose(tfms)
        return tfms

    def prepare_data(self):
        """Base prepare_data method that validates mode and gets appropriate cases"""
        dataset_mode = self.plan["mode"]
        assert dataset_mode in [
            "whole",
            "baseline",
            "patch",
            "source",
            "sourcepbd",
            "pbd",
            "lbd",
            "kbd",
            "dot",
        ], f"Set a value for mode in 'whole', 'patch' or 'source', got {dataset_mode}"

        # Get all cases but only use the ones for this split
        if not hasattr(self, "cases"):
            self.cases_from_project_split()
        # Create data dictionaries for this split
        self.data = self.create_staged_data_dicts(self.cases)

    def create_staged_data_dicts(self, cases):
        data = self.create_data_dicts(cases)
        if self.dual_ssd and len(data) > 0:
            if self.has_hdf5_shard_manifest():
                data = self.copy_hdf5_shards_to_rapid_access_folder2(data)
                return data
            required_keys = {"image", "lm", "indices"}
            if required_keys.issubset(data[0].keys()):
                data = self.copy_data_dicts_to_rapid_access_folder2(data)
            else:
                cprint(
                    "Skipping dual_ssd staging; data dicts do not contain image/lm/indices paths.",
                    color="yellow",
                )
        return data

    @property
    def hdf5_shard_manifest_rel_path(self):
        src_tag = "_".join(str(int(v)) for v in self.src_dims)
        return Path("hdf5_shards") / f"src_{src_tag}" / "manifest.json"

    @property
    def hdf5_shard_manifest_path(self):
        return self.data_folder / self.hdf5_shard_manifest_rel_path

    def has_hdf5_shard_manifest(self):
        try:
            return self.hdf5_shard_manifest_path.exists()
        except (KeyError, TypeError):
            return False

    def _rapid_access_folder2_data_folder(self):
        src_root = Path(COMMON_PATHS["rapid_access_folder"])
        dst_root = Path(COMMON_PATHS["rapid_access_folder2"])
        try:
            data_rel = self.data_folder.relative_to(src_root)
        except ValueError:
            data_rel = Path(self.data_folder.name)
        return dst_root / data_rel

    def copy_hdf5_shards_to_rapid_access_folder2(self, data):
        cprint(
            "Copying half of the HDF5 shards to rapid_access_folder2",
            color="green",
        )
        src_root = Path(COMMON_PATHS["rapid_access_folder"])
        dst_root = Path(COMMON_PATHS["rapid_access_folder2"])
        staged_data_folder = self._rapid_access_folder2_data_folder()
        staged_manifest_fn = staged_data_folder / self.hdf5_shard_manifest_rel_path

        manifest = json.loads(self.hdf5_shard_manifest_path.read_text())
        manifest_parent = self.hdf5_shard_manifest_path.parent
        for i, shard_info in enumerate(pbar(manifest["shards"])):
            shard_path = Path(shard_info["shard"])
            if not shard_path.is_absolute():
                shard_path = manifest_parent / shard_path
            if i % 2 == 1:
                try:
                    shard_rel = shard_path.relative_to(src_root)
                except ValueError:
                    shard_rel = Path(shard_path.name)
                src_shard_path = shard_path
                shard_path = dst_root / shard_rel
                if (
                    not shard_path.exists()
                    or shard_path.stat().st_size != src_shard_path.stat().st_size
                ):
                    shard_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_shard_path, shard_path)
            shard_info["shard"] = str(shard_path)

        staged_manifest_fn.parent.mkdir(parents=True, exist_ok=True)
        staged_manifest_fn.write_text(json.dumps(manifest, indent=2))

        staged = []
        for dici in data:
            out = dict(dici)
            out["data_folder"] = str(staged_data_folder)
            staged.append(out)
        return staged

    def copy_data_dicts_to_rapid_access_folder2(self, data):
        cprint("Copying half of the data to rapid_access_folder2", color="green")
        src_root = Path(COMMON_PATHS["rapid_access_folder"])
        dst_root = Path(COMMON_PATHS["rapid_access_folder2"])
        keys = ("image", "lm", "indices")
        copied = list(data)
        for i, dici in enumerate(pbar(data)):
            if i % 2 == 0:
                continue
            out = dict(dici)
            for key in keys:
                out[key] = self._copy_value_to_rapid_access_folder2(
                    out[key], src_root, dst_root
                )
            copied[i] = out
        return copied

    def _copy_value_to_rapid_access_folder2(self, value, src_root, dst_root):
        src = Path(value)
        dst = dst_root / src.relative_to(src_root)
        if dst.exists():
            return str(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cprint(f"Copying {src} to {dst}", color="green")
        shutil.copy2(src, dst)
        return str(dst)

    def cases_from_project_split(self):
        nnz_allowed = self.plan.get("nnz_allowed", False)
        train_cases, valid_cases = self.project.get_train_val_files(
            self.dataset_params["fold"],
            self.plan["datasources"],
            nnz_allowed=nnz_allowed,
        )

        # Store only the cases for this split
        self.cases = train_cases if self.split == "train" else valid_cases
        assert len(self.cases) > 0, "There are no cases, aborting!"

    def create_data_dicts(self, fnames):
        fnames = [strip_extension(fn) for fn in fnames]
        fnames = [fn + ".pt" for fn in fnames]
        images_fldr = self.data_folder / ("images")
        lms_fldr = self.data_folder / ("lms")
        inds_fldr = self.infer_inds_fldr(self.plan)
        images = list(images_fldr.glob("*.pt"))

        data = []

        for fn in pbar(fnames):
            fn = Path(fn)
            img_fn = find_matching_fn(fn.name, images, ["all"])[0]
            lm_fn = find_matching_fn(fn.name, lms_fldr, ["all"])[0]
            indices_fn = inds_fldr / img_fn.name
            assert img_fn.exists(), "Missing image {}".format(img_fn)
            assert lm_fn.exists(), "Missing labelmap fn {}".format(lm_fn)
            dici = {
                "case_id": img_fn.stem,
                "data_folder": str(self.data_folder),
                "image": str(img_fn),
                "lm": str(lm_fn),
                "indices": str(indices_fn),
            }
            data.append(dici)
        return data

    def infer_inds_fldr(self, plan):
        fg_indices_exclude = plan["fg_indices_exclude"]
        if is_excel_None(fg_indices_exclude):
            fg_indices_exclude = None
            indices_subfolder = "indices"
        else:
            if isinstance(fg_indices_exclude, str):
                fg_indices_exclude = ast_literal_eval(fg_indices_exclude)
            fg_indices_exclude = listify(fg_indices_exclude)
            indices_subfolder = "indices_fg_exclude_{}".format(
                "".join([str(x) for x in fg_indices_exclude])
            )
        return self.data_folder / (indices_subfolder)

    def derive_data_folder(self, plan):
        mode = plan["mode"]
        key = "data_folder_{}".format(mode)
        folders = FolderNames(self.project, plan).folders
        data_folder = folders[key]
        data_folder = Path(data_folder)
        if not data_folder.exists() or len(list(data_folder.rglob("*.pt"))) == 0:
            raise Exception(f"Data folder {data_folder} does not exist")
        return data_folder

    def _num_workers(self):
        if isinstance(self.ds, GridPatchDataset):
            return 0, False
        else:
            num_workers = min(12, self.effective_batch_size * 2)
            persistent_workers = False
            return num_workers, persistent_workers

    def create_train_dataloader(self):
        num_workers, persistent_workers = self._num_workers()
        self.dl = DataLoader(
            self.ds,
            batch_size=self.effective_batch_size,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=persistent_workers,
            pin_memory=True if self.debug == False else False,
            shuffle=True,
        )

    def create_valid_dataloader(self):

        num_workers, persistent_workers = self._num_workers()
        sampler = None
        if self.val_sampling < 1.0:
            n_samples = max(1, int(len(self.ds) * self.val_sampling))
            sampler = RandomSampler(
                self.ds,
                replacement=False,
                num_samples=n_samples,
            )
        self.dl = DataLoader(
            self.ds,
            batch_size=self.effective_batch_size,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=persistent_workers,
            pin_memory=True if self.debug == False else False,
            shuffle=False,
            sampler=sampler,
        )

    def create_dataloader(self):
        if self.split == "train":
            self.create_train_dataloader()
        else:
            self.create_valid_dataloader()

    def setup(self, stage: str = None) -> None:
        # Create transforms for this split

        headline(f"Setting up {self.split} dataset. DS type is: {self.ds_type}")
        print("Src Dims: ", self.plan["src_dims"])
        print("Patch Size: ", self.plan["patch_size"])
        print("Using fg indices: ", self.plan["use_fg_indices"])

        self.create_transforms()
        self.set_transforms(self.keys)
        print("Transforms are set up: ", self.keys)

        self.create_dataset()
        self.create_dataloader()

    def create_dataset(self):
        if not hasattr(self, "data") or len(self.data) == 0:
            print("No data. DS is not being created at this point.")
            return 0
        """Create a single dataset based on split type"""
        print(f"[DEBUG] Number of cases: {len(self.data)}")
        example_case = self.data[0] if len(self.data) > 0 else {}
        if isinstance(example_case, dict):
            example_ref = example_case.get("image", example_case.get("case_id", "None"))
        else:
            example_ref = str(example_case)
        print(f"[DEBUG] Example case: {example_ref}")
        self.ds = self._create_modal_ds()

    def _create_modal_ds(self):
        if is_excel_None(self.ds_type):
            ds = Dataset(data=self.data, transform=self.transforms)
            print("Vanilla Pytorch Dataset set up.")
        elif self.cache_rate > 0.0:
            ds = Dataset(
                data=self.data,
                transform=self.transforms,
                # cache_dir=self.cache_folder,
            )
        elif ds_type == "lmdb":
            # BUG: LMDBDataset will slow training down. fix it  (see #8)
            ds = LMDBDataset(
                data=self.data,
                transform=self.transforms,
                cache_dir=self.cache_folder,
                db_name=f"{self.split}_cache",
            )
        else:
            raise NotImplementedError
        return ds

    @property
    def src_dims(self):
        return self.plan["src_dims"]

    @property
    def cache_folder(self):
        parent_folder = Path(COMMON_PATHS["cache_folder"]) / (
            self.project.project_title
        )
        return parent_folder / (self.data_folder.name) / (self.split)

    @classmethod
    def from_folder(
        cls,
        data_folder: str,
        split: str,
        project,
        configs: dict,
        batch_size=8,
        **kwargs,
    ):
        """
        Create a DataManager instance from a folder containing images and labels.

        Args:
            folder_path (str): Path to folder containing 'images' and 'lms' subfolders
            split (str): Either 'train' or 'valid'
            project: Project instance
            configs (dict): Configuration dictionary
            batch_size (int): Batch size for dataloaders
            **kwargs: Additional arguments passed to DataManager constructor

        Returns:
            DataManager: Instance initialized with data from the specified folder
        Note: After this do not call setup() and instead jump straight to prepare_data()
        """
        data_folder2 = Path(data_folder)
        assert data_folder2.exists(), f"Folder {data_folder2} does not exist"
        assert split in ["train", "valid"], "Split must be either 'train' or 'valid'"

        # Create instance
        instance = cls(
            project=project, configs=configs, batch_size=batch_size, **kwargs
        )

        # Override data folder
        instance.data_folder = data_folder

        # Get files from images and lms folders
        images_folder = data_folder2 / "images"
        lms_folder = data_folder2 / "lms"
        assert images_folder.exists(), f"Images folder {images_folder} does not exist"
        assert lms_folder.exists(), f"Labels folder {lms_folder} does not exist"

        # Create data dictionaries
        image_files = sorted(list(images_folder.glob("*.pt")))
        cls.cases = [d.name for d in image_files]

        return instance


class DataManagerSource(DataManager):
    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        self.keys_tr = "Ld,Rtr,L2,E,F1,F2,Affine,ResizePC,N,IntensityTfms"
        self.keys_val = "L,E,N,Remap,ResizeP"
        if self.keys is None:
            if self.split == "train":
                self.keys = self.keys_tr
            elif self.split == "valid":
                self.keys = self.keys_val
        self.override_batch_size_valid_split(split=self.split)

    def _set_collate_fn(self):
        if self.split == "train":
            self.collate_fn = source_collated
        elif self.split == "valid":
            self.collate_fn = patch_collated
        else:
            raise NotImplementedError

    def __str__(self):
        return "DataManagerSource instance with parameters: " + ", ".join(
            [f"{k}={v}" for k, v in vars(self).items()]
        )

    def __repr__(self):
        return f"DataManagerSource"

    def override_batch_size_valid_split(self, split="valid"):
        if split == "valid":
            self.batch_size = self.effective_batch_size = 1
            self.collate_fn = None


class DataManagerWhole(DataManager):
    def __init__(self, project, configs: dict, batch_size=8, **kwargs):
        super().__init__(project, configs, batch_size, **kwargs)
        self.keys_tr = "L,E,F1,F2,Affine,ResizeW,N,IntensityTfms"
        self.keys_val = "L,E,ResizeW,N"
        if self.keys is None:
            if self.split == "train":
                self.keys = self.keys_tr
            elif self.split == "valid":
                self.keys = self.keys_val

    def _set_collate_fn(self):
        self.collate_fn = whole_collated

    def __str__(self):
        return "DataManagerWhole instance with parameters: " + ", ".join(
            [f"{k}={v}" for k, v in vars(self).items()]
        )

    def __repr__(self):
        return (
            f"DataManagerWhole("
            + ", ".join([f"{k}={v}" for k, v in vars(self).items()])
            + ")"
        )

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
            img_fn = find_matching_fn(fn.name, images, ["all"])[0]
            lm_fn = find_matching_fn(fn.name, lms_fldr, ["all"])[0]
            assert img_fn.exists(), "Missing image {}".format(img_fn)
            assert lm_fn.exists(), "Missing labelmap fn {}".format(lm_fn)
            dici = {"image": str(img_fn), "lm": str(lm_fn)}
            data.append(dici)
        return data


class DataManagerLBD(DataManagerSource):
    def __repr__(self):
        return (
            f"DataManagerLBD(plan={self.plan}, "
            f"dataset_params={self.dataset_params}, "
            f"lbd_folder={self.project.lbd_folder})"
        )

    def __str__(self):
        return (
            "LBD Data Manager with plan {} and dataset parameters: {} "
            "(using LBD folder: {})".format(
                self.plan, self.dataset_params, self.project.lbd_folder
            )
        )


class DataManagerKBD(DataManagerLBD):
    def __repr__(self):
        return (
            f"DataManagerKBD(plan={self.plan}, "
            f"dataset_params={self.dataset_params}, "
            f"kbd_folder={self.project.kbd_folder})"
        )

    def __str__(self):
        return (
            "KBD Data Manager with plan {} and dataset parameters: {} "
            "(using KBD folder: {})".format(
                self.plan, self.dataset_params, self.project.kbd_folder
            )
        )


class DataManagerShort(DataManager):
    def prepare_data(self):
        super().prepare_data()
        self.train_list = self.train_list[:32]

    def train_dataloader(self, num_workers=4, **kwargs):
        return super().train_dataloader(num_workers, **kwargs)


# CODE: in the below class move Rtr after Affine and get rid of Re to see if it affects training speed / model accuracy


class DataManagerPatch(DataManagerSource):
    def __init__(self, project, configs: dict, batch_size=8, **kwargs):
        super().__init__(project, configs, batch_size, **kwargs)
        if self.keys is None:
            self.set_tfm_keys()

    def _set_collate_fn(self):
        if self.split == "test":
            raise NotImplementedError
        else:
            self.collate_fn = patch_collated

    def prepare_data(self):
        """Override prepare_data to ensure proper sequence"""
        self.load_indices_info()  # Now we can safely load bboxes
        self.fg_bg_prior = compute_fgbg_ratio(self.dff, self.plan["nnz_allowed"])
        super().prepare_data()

    def __str__(self):
        return "DataManagerPatch instance with parameters: " + ", ".join(
            [
                f"{k}={v}"
                for k, v in vars(self).items()
                if k not in ["bboxes", "transforms_dict"]
            ]
        )

    def __repr__(self):
        return f"DataManagerPatch(project={self.project}, configs={self.configs}, batch_size={self.batch_size})"

    def set_tfm_keys(self):  # sets own tfm_keys because RP is an addition in this class
        self.keys_tr = "RP, L,Remap,E,N,F1,F2,Affine,ResizePC,IntensityTfms"
        self.keys_val = "RP,L,Remap,E,ResizePC,N "
        self.keys_test = "L,E,N,Remap,ResizeP"  # experimental
        if self.split == "train":
            self.keys = self.keys_tr
        elif self.split == "valid":
            self.keys = self.keys_val
        elif self.split == "test":
            self.keys = self.keys_test
        else:
            raise ValueError

    def load_indices_info(self):
        bbox_fn = self.data_folder / "resampled_dataset_properties.csv"
        try:
            self.dff = pd.read_csv(bbox_fn)
        except FileNotFoundError as f:
            raise FileNotFoundError(
                f"{bbox_fn} does not exist. Run summarize_indices_folder on the indices subfolder.\n{f}"
            )

    def set_effective_batch_size(self):
        self.effective_batch_size = (
            self.batch_size
        )  # never sample a file more than once ofc

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

    def img_patch_size_match(self) -> bool:
        if isinstance(self.dff["shape"][0], str):
            self.dff["shape"] = self.dff["shape"].apply(
                lambda x: np.array(ast.literal_eval(x))
            )
        arr = np.stack(self.dff["shape"])
        all_equal = np.all(arr == arr[0])

        return all_equal

    @classmethod
    def _create_df_from_folder(cls, data_folder: Path):
        images_fldr = data_folder / ("images")
        lms_fldr = data_folder / ("lms")
        images = list(images_fldr.glob("*.pt"))
        dicis_all = []
        for img_fn in images:
            case_id = info_from_filename(img_fn.name, full_caseid=True)["case_id"]
            img_fn_name = img_fn.name
            lm_match = lms_fldr / img_fn_name
            assert lm_match.exists(), (
                "Missing labelmap fn {}. In Patch Data Manager, it is IMPERATIVE that image and lm names are exact matches".format(
                    lm_match
                )
            )
            dici = {"case_id": case_id, "image": str(img_fn), "lm": str(lm_match)}
            dicis_all.append(dici)

        df = pd.DataFrame(dicis_all)
        return df

    def cases_from_project_split(self):
        nnz_allowed = self.plan.get("nnz_allowed", False)
        train_cases, valid_cases = self.project.get_train_val_case_ids(
            self.dataset_params["fold"],
            self.plan["datasources"],
            nnz_allowed=nnz_allowed,
        )

        # Store only the cases for this split
        self.cases = train_cases if self.split == "train" else valid_cases

    def create_data_dicts(self, cids):
        # this does not use cids
        df = self._create_df_from_folder(self.data_folder)
        data = []
        for cid in self.cases:
            dici0 = df.loc[df["case_id"] == cid]
            d = {
                "case_id": dici0["case_id"].iloc[0],
                "image": dici0["image"].tolist(),
                "lm": dici0["lm"].tolist(),
                "n_patches": len(dici0),
            }
            data.append(d)
        return data

    def create_transforms(self):
        super().create_transforms()
        if self.img_patch_size_match() == True:
            self.ResizePC = DummyTransform(
                keys=["lm"]
            )  # there will be no resizing if patch size and image size are same

        self.transforms_dict["L"] = SimpleTorchLoader(
            keys=["image", "lm"], allow_missing_keys=False
        )

    def setup(self, stage: str = None):
        fgbg_ratio = self.dataset_params["fgbg_ratio"]
        fgbg_ratio_adjusted = fgbg_ratio / self.fg_bg_prior
        self.dataset_params["fgbg_ratio"] = float(fgbg_ratio_adjusted)
        super().setup(stage)

    @property
    def src_dims(self):
        return self.plan["patch_size"]


class DataManagerBaseline(DataManagerLBD):
    """
    Small dataset of size =batchsize comprising a single batch. No augmentations. Used to get a baseline
    It has no training augmentations. Whether the flag is True or False doesnt matter.
    Note: It inherits from LBD dataset.
    """

    def __init__(self, project, configs: dict, batch_size=8, **kwargs):
        super().__init__(project, configs, batch_size, **kwargs)
        if self.keys is None:
            self.set_tfm_keys()
            self.keys = self.keys_tr
        self.collate_fn = whole_collated

    def __str__(self):
        return "DataManagerBaseline instance with parameters: " + ", ".join(
            [f"{k}={v}" for k, v in vars(self).items()]
        )

    def __repr__(self):
        return (
            f"DataManagerBaseline("
            + ", ".join([f"{k}={v}" for k, v in vars(self).items()])
            + ")"
        )

    def set_effective_batch_size(self):
        self.effective_batch_size = self.batch_size

    def set_tfm_keys(self):
        self.keys_val = "L,Ld,E,ResizePC,N"
        self.keys_tr = self.keys_val

    def prepare_data(self):
        super().prepare_data()
        self.data_train = self.data_train[: self.batch_size]
        self.data_valid = self.data_valid[: self.batch_size]

    @property
    def cache_folder(self):
        parent_folder = Path(COMMON_PATHS["cache_folder"]) / (
            self.project.project_title
        )
        return parent_folder / (self.data_folder.name + "_baseline")


# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
if __name__ == "__main__":
    from pprint import pp

    from fastcore.basics import warnings
    from fran.configs.parser import ConfigMaker
    from fran.transforms.imageio import LoadTorchd
    from utilz.imageviewers import ImageMaskViewer

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *

    project_title = "kits23"
    proj = Project(project_title=project_title)

    CL = ConfigMaker(proj)
    CL.setup(2)
    conf = CL.configs
# %%
# SECTION:-------------------- LIDC-------------------------------------------------------------------------------------- <CR>
    batch_size = 4
    ds_type = "lmdb"
    ds_type = None
    proj_tit = proj.project_title
    conf["dataset_params"]["cache_rate"] = 0.0

    D = DataManagerDual(
        project_title=proj_tit,
        configs=conf,
        batch_size=batch_size,
        ds_type=ds_type,
        dual_ssd=False,
    )

# %%
    D.prepare_data()
    D.setup("fit")
    tmv = D.valid_manager
    tmt = D.train_manager
    tmv.transforms_dict
# %%
    dl = tmt.dl
    iteri = iter(dl)
    for x, batch in enumerate(iteri):
        batch = next(iteri)
        print(batch['image'].shape)
# %%

    # while iteri:
    batch['lm'].shape
    batch['image'].shape
# %%
    ImageMaskViewer([batch["image"][0,0], batch["lm"][0,0]])
# %%
    #     print(batch['image'].shape)
    ds = D.valid_manager.ds
    for n in range(len(ds)):
        dat = ds[n]

# %%
    P = D.valid_manager
    td = P.transforms_dict
# %%
    n = 2
    data = D.valid_manager.data[n]
    dici = P.transforms_dict["RP"](data)
    dici = P.transforms_dict["L"](dici)
    pp(dici["image"].meta)
# %%

    dici = td["Remap"](dici)
    dici = td["E"](dici)
    dici = td["ResizePC"](dici)

# %%
    for batch in iteri:
        print(batch["image"].shape)
        # print(batch["patch_coords"])
# %%
    batch = next(iteri)
    batch["image"].device

    batch.keys()
# %%
# %%
# SECTION:-------------------- BONES-------------------------------------------------------------------------------------- <CR> <CR>

    batch_size = 3
    ds_type = None
    D = DataManagerMulti(
        project_title=proj_bones.project_title,
        configs=config_bones,
        batch_size=batch_size,
        ds_type=ds_type,
        manager_class_train=DataManagerPatch,
        manager_class_valid=DataManagerPatch,
    )

# %%
    D.prepare_data()
    D.setup()
    tmv = D.valid_manager
    tmt = D.train_manager
    tme = D.test_manager
    tmv.transforms_dict

    # %dat%
    dat = tmt.data[0]
    dat2 = tmt.transforms_dict["RP"](dat)
    im = torch.load(dat2["image"], weights_only=False)
    lm = torch.load(dat2["lm"], weights_only=False)
    dat3 = {"image": im, "lm": lm}

    td = tmt.transforms_dict
    dat4 = td["Remap"](dat3)
    dat5 = td["E"](dat4)
    dat6 = td["F1"](dat5)
    dat7 = td["ResizePC"](dat6)

    S = SimpleTorchLoader(keys=["image", "lm"], allow_missing_keys=False)

    dat3 = S(dat2)
# %%

    dl2 = D.train_dataloader()
    iteri2 = iter(dl2)
    batch = next(iteri2)
    batch["image"].shape
# %%
    dat = tmt.data[0]
    dat["image"].shape
    dici = tmt.transforms_dict["RP"](dat)
    # dici =  tmt.transforms_dict["Ld"](dat)

# %%
# SECTION:-------------------- Patch manager checks-------------------------------------------------------------------------------------- <CR> <CR>

    data_folder = (
        "/r/datasets/preprocessed/lidc/patches/spc_080_080_150_rspbb76320a_128128096"
    )

    batch_size = 2
# %%
    P = DataManagerPatch(
        project=proj,
        configs=conf,
        batch_size=batch_size,
        data_folder=data_folder,
        ds_type=None,
    )

# %%
    P.prepare_data()
    P.setup("fit")
# %%
    dici = P.ds[0]

    data = P.data[0]
# %%
    dici = P.transforms_dict["RP"](data)
    dici = P.transforms_dict["L"](dici)
# %%
    n = 1
    im = batch["image"][n][0]
    lm = batch["lm"][n][0]
    coords = batch["patch_coords"][n]
    print(im.meta["filename_or_obj"])
    print(coords)
    print(im.max())
# %%
    dici = P.transforms_dict["RP"](data)
    img_fn = dici["image"]

# %%
    im = im.permute(2, 0, 1)
    lm = lm.permute(2, 0, 1)
    ImageMaskViewer([im, lm])
# %%
    dl2 = D.train_dataloader()
    iteri2 = iter(dl2)
    # while iteri:
    #     print(batch['image'].shape)
    img_fns = tmv.data[0]["image"]
    img = torch.load(img_fns, weights_only=False)
    ImageMaskViewer([img, img])
# %%
    batch2 = next(iteri2)

    batch2.keys()
# %%
    n = 0
    im = batch2["image"][n][0]
    lm = batch2["lm"][n][0]
    im = im.permute(2, 0, 1)
    lm = lm.permute(2, 0, 1)
    ImageMaskViewer([im, lm])

# %%
    ds = tmv.ds
    dat = ds[0]
    dici = ds.data[0]
    tmv.tfms_list

# %%

# %%
    D.train_ds[0]
# %%
    ds1 = PersistentDataset(
        data=D.valid_manager.data,
        transform=D.valid_manager.transforms,
        cache_dir=D.valid_manager.cache_folder,
    )
    dici = ds1[0]
# %%
# %%
# SECTION:-------------------- LIVER-------------------------------------------------------------------------------------- <CR> <CR>

# %%
    D = DataManagerMulti(
        project_title=proj_litsmc.project_title,
        configs=config_litsmc,
        batch_size=batch_size,
        ds_type=ds_type,
    )

# %%

# SECTION:-------------------- LBD-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
    batch_size = 2
    ds_type = "lmdb"
    ds_type = None
    config_litsmc["dataset_params"]["mode"] = None
    config_litsmc["dataset_params"]["cache_rate"] = 0

    config_litsmc["plan_train"]
# %%
    D = DataManagerMulti(
        project_title=proj_litsmc.project_title,
        configs=config_litsmc,
        batch_size=batch_size,
        ds_type=ds_type,
    )

# %%
# SECTION:-------------------- FromFolder-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>

    Dev = EnsureTyped(keys=["image", "lm"], device=1)
    dat2 = Dev(dat)

# SECTION:-------------------- DataManagerWhole-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>
# %%
    # Test DataManagerWhole with DataManagerMulti
    D = DataManagerMulti(
        project=proj_tot, configs=config_tot, batch_size=4, ds_type=None
    )
    D.prepare_data()
    D.setup()
    tmv = D.train_manager
# %%

    # Now use train_manager or valid_manager to access the data
    dl = D.train_dataloader()
    bb = D.train_ds[0]

    iteri = iter(dl)
    b = next(iteri)
    b = D.train_ds[0]
    im = b["image"]
    lm = b["lm"]
    ImageMaskViewer([im[0], lm[0]])

# %%
# %%

    config_nodes["dataset_params"]["cache_rate"] = 0.5
    # D3 = DataManagerLBD(project=proj_nodes,config=config_nodes,split='valid',ds_type='cache',cache_rate=0.5)
    D3 = DataManagerLBD.from_folder(
        data_folder="/r/datasets/preprocessed/tmp",
        split="valid",
        project=proj_nodes,
        config=config_nodes,
        ds_type="lmdb",
    )
    D3.prepare_data()
    D3.setup()

# SECTION:-------------------- DataManagerPlain-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
# %%
    batch_size = 2
    D = DataManagerMulti(
        project=proj_tot, configs=config_tot, batch_size=batch_size, ds_type=None
    )
    # D.effective_batch_size = int(D.batch_size / D.plan["samples_per_file"])
# %%
    D.prepare_data()
    D.setup()
    b = D.train_ds[0]
    b["image"].shape

    D = tmv.transforms_dict["Dev"]

    b2 = D(b[0])
# %%
    lm = b["lm"]
    # %
# SECTION:-------------------- DataManagerSource ------------------------------------------------------------------------------------------------------ <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>

# %%
    batch_size = 2
    D = DataManagerMulti(
        project=proj_tot, configs=config_tot, batch_size=batch_size, ds_type=None
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
    im = b["image"]
    lm = b["lm"]

# %%
# %%
# SECTION:-------------------- Patch-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>

    batch_size = 2
    P = DataManagerPatch(
        project=proj_bones,
        configs=config_bones,
        batch_size=batch_size,
        ds_type=None,
        data_folder="/r/datasets/preprocessed/bones/fixed_spacing/spc_100_100_100",
    )

    P.prepare_data()
# %%
# %%
    dl = D.train_dataloader()
    iteri = iter(dl)
    b = next(iteri)
    im = b["image"]
    lm = b["lm"]
    ImageMaskViewer([im[0, 0], lm[0, 0]])
# %%
    for i, dd in enumerate(D.train_ds):
        print(i)

# %%
    cids = D.train_cids
    patches_per_id = []

# %%
    dici = D.data_train[0]
    dici = D.data[0]
    D.transforms_dict.keys()
    D.transforms_dict[""](dici)
# %%
    RD = RandomPatch(keys=["image", "lm"])
    L = D.transforms_dict["L"]

    dici2 = RD(dici)
    L(dici2)

# %%
# SECTION:-------------------- TROUBLESHOOTING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>

    D.prepare_data()
    D.setup()
    tmv = D.train_manager
    tmv.transforms_dict
# %%
    ds = tmv.ds
    dici = ds.data[0]
    dici2 = tmv.transforms(dici)
    tmv.tfms_list

    dv = resolve_device(0)
    Dev = ToDeviceD(keys=["image", "lm"], device=dv)
    dat3 = Dev(dat2)
# %%
# SECTION:-------------------- ROUGH-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>

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
        lazy=False,
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
    D.val_indices = int(len(D.train_manager.cases) * 0.2)
