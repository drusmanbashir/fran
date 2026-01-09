# %%

import os
import re
from tqdm.auto import tqdm
from collections import defaultdict
from functools import reduce
from operator import add
from pathlib import Path

import ipdb
import numpy as np
import torch
from fastcore.basics import listify, operator, store_attr, warnings
from lightning import LightningDataModule
from monai.data import DataLoader, Dataset
from monai.data.dataset import CacheDataset, LMDBDataset, PersistentDataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (RandCropByPosNegLabeld,
                                                 ResizeWithPadOrCropd)
from monai.transforms.intensity.dictionary import (RandAdjustContrastd,
                                                   RandScaleIntensityd,
                                                   RandShiftIntensityd)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, Resized
from monai.transforms.transform import RandomizableTransform
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from utilz.fileio import load_dict, load_yaml
from utilz.helpers import find_matching_fn, folder_name_from_list
from utilz.string import (ast_literal_eval, cleanup_fname, info_from_filename,
                          strip_extension)

from fran.configs.parser import ConfigMaker, is_excel_None
from fran.data.collate import whole_collated
from fran.data.dataset import NormaliseClipd, fg_in_bboxes
from fran.managers import Project
from fran.preprocessing.helpers import bbox_bg_only
from fran.transforms.imageio import TorchReader
from fran.transforms.intensitytransforms import RandRandGaussianNoised
from fran.transforms.misc_transforms import LoadTorchDict, MetaToDict
from fran.transforms.spatialtransforms import ExtractContiguousSlicesd


def process_items(items):
    data = defaultdict(list)
    filenames = defaultdict(list)

    for item in items:
        for key, value in item.items():
            data[key].append(value)
            try:
                filenames[key].append(value.meta["filename_or_obj"])
            except Exception:
                filenames[key].append(None)

    return dict(data), dict(filenames)


def source_collated(batch):
    all_data = defaultdict(list)
    all_filenames = defaultdict(list)

    for item in batch:
        data, filenames = process_items(item)
        for k, v in data.items():
            all_data[k].extend(v)
        for k, v in filenames.items():
            all_filenames[k].extend(v)

    output = {}
    for key in all_data:
        values = all_data[key]
        try:
            # Stack if all elements are tensors
            if all(isinstance(x, torch.Tensor) for x in values):
                stacked = torch.stack(values, 0)
                # Attach filenames to .meta if possible
                if hasattr(stacked, "meta"):
                    stacked.meta["filename_or_obj"] = (
                        all_filenames[key][0] if len(batch) == 1 else all_filenames[key]
                    )
                output[key] = stacked
            else:
                output[key] = values
        except Exception as e:
            raise RuntimeError(f"Error processing key '{key}': {e}")

    return output


# common_vars_filename = os.environ["FRAN_COMMON_PATHS"]
# COMMON_PATHS = load_yaml(common_vars_filename)

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


class DataManagerDual(LightningDataModule):
    """
    A higher-level DataManager that manages separate training and validation DataManagers
    """

    def __init__(
        self,
        project_title,
        config: dict,
        batch_size=8,
        cache_rate=0.0,
        ds_type=None,
        save_hyperparameters=True,
        keys_tr=None,
        keys_val=None,
        data_folder=None,
    ):
        super().__init__()
        project = Project(project_title)
        if save_hyperparameters:
            self.save_hyperparameters(
                "project_title", "config", logger=False
            )  # logger = False otherwise it clashes with UNet Manager
        manager_class_train, manager_class_valid = self.infer_manager_classes(config)

        # Create separate managers for training and validation
        self.train_manager = manager_class_train(
            project=project,
            config=config,
            batch_size=batch_size,
            cache_rate=cache_rate,
            ds_type=ds_type,
            split="train",
            keys_tr=keys_tr,
            data_folder=data_folder,
        )

        self.valid_manager = manager_class_valid(
            project=project,
            config=config,
            batch_size=batch_size,
            cache_rate=cache_rate,
            ds_type=None,
            split="valid",
            keys_val=keys_val,
            data_folder=data_folder,
        )

    def prepare_data(self):
        """Prepare both training and validation data"""
        self.train_manager.prepare_data()
        self.valid_manager.prepare_data()

    def setup(self, stage=None):
        """Set up both managers"""
        self.train_manager.setup(stage)
        self.valid_manager.setup(stage)

    def train_dataloader(self):
        """Return training dataloader"""
        return self.train_manager.dl

    def val_dataloader(self):
        """Return validation dataloader"""
        return self.valid_manager.dl

    @property
    def train_ds(self):
        """Access to training dataset"""
        return self.train_manager.ds

    @property
    def valid_ds(self):
        """Access to validation dataset"""
        return self.valid_manager.ds

    def infer_manager_classes(self, config):
        """
        Infer the appropriate DataManager class based on the mode in config

        Args:
            config (dict): Configuration dictionary containing plan_train and plan_valid

        Returns:
            class: The appropriate DataManager class

        Raises:
            AssertionError: If train and valid modes don't match
            ValueError: If mode is not recognized
        """
        train_mode = config["plan_train"]["mode"]
        valid_mode = config["plan_valid"]["mode"]

        # Ensure train and valid modes match
        assert (
            train_mode == valid_mode
        ), f"Train mode '{train_mode}' and valid mode '{valid_mode}' must match"

        # Map modes to manager classes
        mode_to_class = {
            "source": DataManagerSource,
            "whole": DataManagerWhole,
            "patch": DataManagerPatch,
            "lbd": DataManagerLBD,
            "baseline": DataManagerBaseline,
            "pbd": DataManagerWID,
        }

        if train_mode not in mode_to_class:
            raise ValueError(
                f"Unrecognized mode: {train_mode}. Must be one of {list(mode_to_class.keys())}"
            )

        return mode_to_class[train_mode], mode_to_class[valid_mode]


class DataManager(LightningDataModule):
    def __init__(
        self,
        project,
        config: dict,
        batch_size=8,
        cache_rate=0.0,
        ds_type=None,
        split="train",  # Add sp,lit parameter
        save_hyperparameters=False,
        keys_tr=None,
        keys_val=None,
        data_folder=None,  # Optional parent folder containing images/ and lms/ subfolders
    ):

        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters("project", "config", "split", logger=False)
        store_attr()
        self.plan = config[f"plan_{split}"]
        global_properties = load_dict(project.global_properties_filename)
        self.dataset_params = config["dataset_params"]
        self.dataset_params["intensity_clip_range"] = global_properties[
            "intensity_clip_range"
        ]
        transform_factors = config["transform_factors"]
        self.dataset_params["mean_fg"] = global_properties["mean_fg"]
        self.dataset_params["std_fg"] = global_properties["std_fg"]
        self.batch_size = batch_size
        self.cache_rate = cache_rate
        self.ds_type = ds_type
        self.set_effective_batch_size()
        if data_folder is None:
            self.data_folder = self.derive_data_folder()
        else:
            self.data_folder = Path(data_folder)
            assert (
                self.data_folder.is_dir()
            ), "Dataset folder {} does not exists or is not a folder".format(
                self.data_folder
            )
        self.assimilate_tfm_factors(transform_factors)
        self.set_tfm_keys(keys_tr, keys_val)
        self.set_collate_fn()

    def set_collate_fn(self):
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

    def set_tfm_keys(self, keys_tr=None, keys_val=None):
        if keys_tr is None:
            self.keys_tr = "L,Ld,E,Rtr,F1,F2,Affine,Re,N,IntensityTfms"
        else:
            self.keys_tr = keys_tr
        if keys_val is None:
            self.keys_val = "L,Ld,E,N,Re"
        else:
            self.keys_val = keys_val

    def assimilate_tfm_factors(self, transform_factors):
        for key, value in transform_factors.items():
            dici = {"value": value[0], "prob": value[1]}
            setattr(self, key, dici)

    def create_transforms(self, keys):
        """
        Creates and assigns transformations based on provided keys.

        Args:
            keys (str): Comma-separated string of transform keys to include

        Returns:
            None: Sets self.transforms with composed transforms

        Transform Abbreviations and Full Names:
            Affine: RandAffined - Random affine transformations (rotation, scaling)
            E: EnsureChannelFirstd - Ensures channel dimension is first
            Ex: ExtractContiguousSlicesd - Extracts 3 contiguous slices (z-1, z, z+1)
            F1: RandFlipd (axis=0) - Random flip along spatial axis 0
            F2: RandFlipd (axis=1) - Random flip along spatial axis 1
            Ind: MetaToDict - Convert metadata to dictionary format
            IntensityTfms: Collection of intensity transforms:
                - RandAdjustContrastd: Random contrast adjustment
                - RandRandGaussianNoised: Random Gaussian noise addition
                - RandScaleIntensityd: Random intensity scaling
                - RandShiftIntensityd: Random intensity shifting
            L: LoadImaged - Load image and label data
            Ld: LoadTorchDict - Load torch dictionary with indices
            N: NormaliseClipd - Normalizes and clips image intensities
            None: Sets self.transforms with composed transforms
            RP: RandomPatch - Applies random patch sampling
            Re: ResizeWithPadOrCropd - Resize with padding or cropping
            Resize: Resized - Resize to specified spatial size
            Rtr: RandCropByPosNegLabeld (training) - Random crop with positive/negative sampling
            Rva: RandCropByPosNegLabeld (validation) - Random crop for validation
        """
        # Initialize transforms dictionary and list
        self.plan["patch_size"] = self.plan["patch_size"][:2]
        self.transforms_dict = {}
        self.tfms_list = []
        # Parse transform keys
        include_keys = [key.strip() for key in keys.split(",")]
        # Conditionally create transforms based on inclusion list
        if "E" in include_keys:
            E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
            self.transforms_dict["E"] = E
        if "Ex" in include_keys:
            Ex = ExtractContiguousSlicesd(keys=["image_fns", "lm_fldr", "n_slices"])
            self.transforms_dict["Ex"] = Ex

        if "N" in include_keys:
            N = NormaliseClipd(
                keys=["image"],
                clip_range=self.dataset_params["intensity_clip_range"],
                mean=self.dataset_params["mean_fg"],
                std=self.dataset_params["std_fg"],
            )
            self.transforms_dict["N"] = N

        if "RP" in include_keys:
            RP = RandomPatch()
            self.transforms_dict["RP"] = RP

        if "F1" in include_keys:
            F1 = RandFlipd(
                keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=0, lazy=True
            )
            self.transforms_dict["F1"] = F1

        if "F2" in include_keys:
            F2 = RandFlipd(
                keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=1, lazy=True
            )
            self.transforms_dict["F2"] = F2

        if "IntensityTfms" in include_keys:
            IntensityTfms = [
                RandScaleIntensityd(
                    keys="image", factors=self.scale["value"], prob=self.scale["prob"]
                ),
                RandRandGaussianNoised(
                    keys=["image"],
                    std_limits=self.noise["value"],
                    prob=self.noise["prob"],
                ),
                RandShiftIntensityd(
                    keys="image", offsets=self.shift["value"], prob=self.shift["prob"]
                ),
                RandAdjustContrastd(
                    ["image"], gamma=self.contrast["value"], prob=self.contrast["prob"]
                ),
            ]
            self.transforms_dict["IntensityTfms"] = IntensityTfms

        if "Affine" in include_keys:
            Affine = RandAffined(
                keys=["image", "lm"],
                mode=["bilinear", "nearest"],
                prob=self.config["affine3d"]["p"],
                rotate_range=self.config["affine3d"]["rotate_range"],
                scale_range=self.config["affine3d"]["scale_range"],
            )
            self.transforms_dict["Affine"] = Affine

        if "Resize" in include_keys:
            Resize = Resized(
                keys=["image", "lm"],
                spatial_size=self.plan["patch_size"],
                mode=["linear", "nearest"],
                lazy=True,
            )

            self.transforms_dict["Resize"] = Re

        if "Re" in include_keys:
            Re = ResizeWithPadOrCropd(
                keys=["image", "lm"],
                spatial_size=self.plan["patch_size"],
                lazy=True,
            )
            self.transforms_dict["Re"] = Re

        # Continue similarly for the remaining transforms like L, Ld, Ind, Rtr, Rva...

        if "L" in include_keys:
            L = LoadImaged(
                keys=["image", "lm"],
                image_only=True,
                ensure_channel_first=False,
                simple_keys=True,
            )
            L.register(TorchReader())
            self.transforms_dict["L"] = L

        if "Ld" in include_keys:
            Ld = LoadTorchDict(
                keys=["indices"], select_keys=["lm_fg_indices", "lm_bg_indices"]
            )
            self.transforms_dict["Ld"] = Ld

        if "Ind" in include_keys:
            Ind = MetaToDict(keys=["lm"], meta_keys=["lm_fg_indices", "lm_bg_indices"])
            self.transforms_dict["Ind"] = Ind

        if "Rtr" in include_keys:
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

        if "Rva" in include_keys:
            Rva = RandCropByPosNegLabeld(
                keys=["image", "lm"],
                label_key="lm",
                image_key="image",
                image_threshold=-2600,
                spatial_size=self.plan["patch_size"],
                pos=1,
                neg=1,
                num_samples=1,
                lazy=True,
                allow_smaller=True,
            )
            self.transforms_dict["Rva"] = Rva
        for k in include_keys:
            if k == "IntensityTfms":
                self.tfms_list.extend(self.transforms_dict[k])
            else:
                self.tfms_list.append(self.transforms_dict[k])
        self.transforms = Compose(self.tfms_list)

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
        """Base prepare_data method that validates mode and gets appropriate cases"""
        dataset_mode = self.plan["mode"]
        assert dataset_mode in [
            "whole",
            "baseline",
            "patch",
            "source",
            "pbd",
            "lbd",
        ], f"Set a value for mode in 'whole', 'patch' or 'source', got {dataset_mode}"

        # Get all cases but only use the ones for this split
        if not hasattr(self, "cases"):
            self.cases_from_project_split()
        # Create data dictionaries for this split
        self.data = self.create_data_dicts(self.cases)

    def cases_from_project_split(self):
        train_cases, valid_cases = self.project.get_train_val_files(
            self.dataset_params["fold"], self.plan["datasources"]
        )

        # Store only the cases for this split
        self.cases = train_cases if self.split == "train" else valid_cases
        assert len(self.cases) > 0, "There are no cases, aborting!"

    def create_data_dicts(self, cases):
        cases = [strip_extension(fn) for fn in cases]
        image_subfoldrs = list(self.data_folder.glob("images/*"))
        lm_subfoldrs = list(self.data_folder.glob("lms/*"))
        data = []
        if len(cases)==0 or cases is None:
            raise ValueError("No cases found")
        for case_ in tqdm(cases) if cases else []:
            img_matched = [fn for fn in image_subfoldrs if case_ == fn.name]
            lm_matched = [fn for fn in lm_subfoldrs if case_ == fn.name]
            assert (
                len(img_matched) == 1 and len(lm_matched) == 1
            ), "Multiple images for case {}".format(case_)
            img_fldr = img_matched[0]
            lm_fldr = lm_matched[0]
            img_fns = list(img_fldr.glob("*"))
            n_slices = len(img_fns)
            dici = {"image_fns": img_fns, "lm_fldr": lm_fldr, "n_slices": n_slices}
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

    def derive_data_folder(self):
        raise NotImplementedError

    def create_dataloader(self):
        self.dl = DataLoader(
            self.ds,
            batch_size=self.effective_batch_size,
            num_workers=self.effective_batch_size * 2,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )

    def setup(self, stage: str = None) -> None:
        # Create transforms for this split
        keys = self.keys_tr if self.split == "train" else self.keys_val
        self.create_transforms(keys)
        print("Setting up datasets. Training ds type is: ", self.ds_type)

        print(f"Setting up {self.split} dataset. DS type is: {self.ds_type}")
        self.create_dataset()
        self.create_dataloader()

    def create_dataset(self):
        """Create a single dataset based on split type"""
        print(f"[DEBUG] Number of cases: {len(self.cases)}")
        print(f"[DEBUG] Example case: {self.cases[0] if self.cases else 'None'}")
        if not hasattr(self, "data") or len(self.data) == 0:
            print("No data. DS is not being created at this point.")
            return 0

        if self.split == "train":
            self.ds = self._create_train_ds()
        else:
            self.ds = self._create_valid_ds()

    def _create_train_ds(self):
        if is_excel_None(self.ds_type):
            ds = Dataset(data=self.data, transform=self.transforms)
            print("Vanilla Pytorch Dataset set up.")

        elif self.ds_type == "cache":
            ds = CacheDataset(
                data=self.data,
                transform=self.transforms,
                cache_rate=self.cache_rate,
            )
        elif self.ds_type == "lmdb":
            ds = LMDBDataset(
                data=self.data,
                transform=self.transforms,
                cache_dir=self.cache_folder,
                db_name=f"{self.split}_cache",
            )
        else:
            raise NotImplementedError
        return ds

    def _create_valid_ds(self):
        """
        valid-ds is a GridPatchDataset to make training runs comparable
        """
        ds = PersistentDataset(
            data=self.data,
            transform=self.transforms,
            cache_dir=self.cache_folder,
        )
        return ds

        # self.plan['patch_size']= self.plan['patch_size'][:2]
        #
        # tr()
        # patch_iter = PatchIterd(keys =['image','lm'], patch_size=  self.plan['patch_size'] )
        # ds = GridPatchDataset(data=ds1 ,patch_iter=patch_iter)
        # return ds

    @property
    def src_dims(self):
        if self.dataset_params["zoom"] == True:
            src_dims = self.dataset_params["src_dims"]
        else:
            src_dims = self.plan["patch_size"]
        return src_dims

    @property
    def cache_folder(self):
        parent_folder = Path(COMMON_PATHS["cache_folder"]) / (
            self.project.project_title
        )
        return parent_folder / (self.data_folder.name)

    @classmethod
    def from_folder(
        cls, data_folder: str, split: str, project, config: dict, batch_size=8, **kwargs
    ):
        """
        Create a DataManager instance from a folder containing images and labels.

        Args:
            folder_path (str): Path to folder containing 'images' and 'lms' subfolders
            split (str): Either 'train' or 'valid'
            project: Project instance
            config (dict): Configuration dictionary
            batch_size (int): Batch size for dataloaders
            **kwargs: Additional arguments passed to DataManager constructor

        Returns:
            DataManager: Instance initialized with data from the specified folder
        Note: After this do not call setup() and instead jump straight to prepare_data()
        """
        data_folder = Path(data_folder)
        assert data_folder.exists(), f"Folder {data_folder} does not exist"
        assert split in ["train", "valid"], "Split must be either 'train' or 'valid'"

        # Create instance
        instance = cls(project=project, config=config, batch_size=batch_size, **kwargs)

        # Override data folder
        instance.data_folder = data_folder

        # Get files from images and lms folders
        images_folder = data_folder / "images"
        lms_folder = data_folder / "lms"
        assert images_folder.exists(), f"Images folder {images_folder} does not exist"
        assert lms_folder.exists(), f"Labels folder {lms_folder} does not exist"

        # Create data dictionaries
        image_files = sorted(list(images_folder.glob("*.pt")))
        cls.cases = [d.name for d in image_files]
        # Assign to train or validation based on split
        # if split == 'train':
        #     instance.train_cases = cases
        # else:
        #     instance.valid_cases =cases

        return instance


class DataManagerSource(DataManager):
    # def __init__(self, project, config: dict, batch_size=8, **kwargs):
    #     super().__init__(project, config, batch_size, **kwargs)

    def set_collate_fn(self):
        self.collate_fn = source_collated

    def __str__(self):
        return "DataManagerSource instance with parameters: " + ", ".join(
            [f"{k}={v}" for k, v in vars(self).items()]
        )

    def __repr__(self):
        return (
            f"DataManagerSource("
            + ", ".join([f"{k}={v}" for k, v in vars(self).items()])
            + ")"
        )

    def derive_data_folder(self):
        assert (
            self.plan["mode"] == "source"
        ), f"Dataset mode must be 'source' for DataManagerSource, got '{self.plan['mode']}'"
        prefix = "spc"
        spacing = self.plan["spacing"]
        parent_folder = self.project.fixed_spacing_folder
        data_folder = folder_name_from_list(prefix, parent_folder, spacing)
        return data_folder

    # def prepare_data(self):
    #     super().prepare_data()
    #     self.data_train = self.create_data_dicts(self.train_cases)
    #     self.data_valid = self.create_data_dicts(self.valid_cases)

    def create_transforms(self, keys="all"):
        super().create_transforms(keys)


class DataManagerWhole(DataManagerSource):

    def __init__(self, project, config: dict, batch_size=8, **kwargs):
        super().__init__(project, config, batch_size, **kwargs)
        self.keys_tr = "L,E,F1,F2,Affine,Resize,N,IntensityTfms"
        self.keys_val = "L,E,Resize,N"

    def set_collate_fn(self):
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

    def derive_data_folder(self):
        assert (
            self.plan["mode"] == "whole"
        ), f"Dataset mode must be 'whole' for DataManagerWhole, got '{self.plan['mode']}'"
        prefix = "sze"
        spatial_size = self.plan["patch_size"]
        parent_folder = self.project.fixed_size_folder
        data_folder = folder_name_from_list(prefix, parent_folder, spatial_size)
        return data_folder

    def create_data_dicts(self, fnames):
        fnames = [strip_extension(fn) for fn in fnames]
        fnames = [fn + ".pt" for fn in fnames]
        fnames = fnames
        images_fldr = self.data_folder / ("images")
        lms_fldr = self.data_folder / ("lms")
        images = list(images_fldr.glob("*.pt"))
        data = []
        # for fn in fnames[400:432]:
        for fn in tqdm(fnames):
            fn = Path(fn)
            img_fn = find_matching_fn(fn.name, images, "all")[0]
            lm_fn = find_matching_fn(fn.name, lms_fldr, "all")[0]
            assert img_fn.exists(), "Missing image {}".format(img_fn)
            assert lm_fn.exists(), "Missing labelmap fn {}".format(lm_fn)
            dici = {"image": img_fn, "lm": lm_fn}
            data.append(dici)
        return data


class DataManagerLBD(DataManagerSource):
    def derive_data_folder(self, data_folder=None):
        assert (
            self.plan["mode"] == "lbd"
        ), f"Dataset mode must be 'lbd' for DataManagerLBD, got '{self.plan['mode']}'"
        spacing = ast_literal_eval(self.plan["spacing"])
        parent_folder = self.project.lbd_folder
        folder_suffix = "plan" + str(self.dataset_params[f"plan_{self.split}"])
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

    # def prepare_data(self):
    #     super().prepare_data()
    #     self.data_train = self.create_data_dicts(self.train_cases[:32])
    #     self.data_valid = self.create_data_dicts(self.valid_cases[:16])


class DataManagerWID(DataManagerLBD):
    def derive_data_folder(self, dataset_mode=None):
        spacing = self.plan["spacing"]
        parent_folder = self.project.pbd_folder
        folder_suffix = "plan" + str(self.dataset_params[f"plan_{self.split}"])
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
        return (
            f"DataManagerPBD(plan={self.plan}, "
            f"dataset_params={self.dataset_params}, "
            f"pbd_folder={self.project.pbd_folder})"
        )

    def __str__(self):
        return (
            "Patient-bound  Data Manager (PBD) with plan {} and dataset parameters: {} "
            "(using PBD folder: {})".format(
                self.plan, self.dataset_params, self.project.pbd_folder
            )
        )


class DataManagerShort(DataManager):
    def prepare_data(self):
        super().prepare_data()
        self.train_list = self.train_list[:32]

    def train_dataloader(self, num_workers=4, **kwargs):
        return super().train_dataloader(num_workers, **kwargs)


class DataManagerPatch(DataManagerSource):
    def __init__(self, project, config: dict, batch_size=8, **kwargs):
        super().__init__(project, config, batch_size, **kwargs)

    def prepare_data(self):
        """Override prepare_data to ensure proper sequence"""
        self.data_folder = self.derive_data_folder()
        self.load_bboxes()  # Now we can safely load bboxes
        self.fg_bg_prior = fg_in_bboxes(self.bboxes)

        self.train_cases, self.valid_cases = self.project.get_train_val_files(
            self.dataset_params["fold"], self.plan_train["datasources"]
        )
        print("Creating train data dicts from BBoxes")
        self.data_train = self.create_data_dicts(self.train_cases)
        print("Creating val data dicts from BBoxes")
        self.data_valid = self.create_data_dicts(self.valid_cases)

    def __str__(self):
        return "DataManagerPatch instance with parameters: " + ", ".join(
            [
                f"{k}={v}"
                for k, v in vars(self).items()
                if k not in ["bboxes", "transforms_dict"]
            ]
        )

    def __repr__(self):
        return f"DataManagerPatch(project={self.project}, config={self.config}, batch_size={self.batch_size}, fg_bg_prior={self.fg_bg_prior})"

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
            strip_extension(fn.name) + "_" + str(indx) + ".pt"
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

    def set_tfm_keys(self):
        if not is_excel_None(self.plan_train["remapping_train"]):
            self.keys_tr = "RP,L,Ld,P,E,Rtr,F1,F2,Affine,Re,N,IntensityTfms"
        else:
            self.keys_tr = "RP,L,Ld,E,Rva,F1,F2,Affine,Re,N,IntensityTfms"
        self.keys_val = "RP,L,Ld,E,N"

    def load_bboxes(self):
        bbox_fn = self.data_folder / "bboxes_info"
        self.bboxes = load_dict(bbox_fn)

    # def prepare_data(self):
    #     self.train_cids, self.valid_cids = self.project.get_train_val_files(
    #         self.dataset_params["fold"]
    #     )
    #     print("Creating train data dicts")
    #     self.data_train = self.create_data_dicts(self.train_cids)
    #     print("Creating val data dicts")
    #     self.data_valid = self.create_data_dicts(self.valid_cids)

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

    def create_data_dicts(self, fnames):
        patches = []
        fnames = self.train_cases
        patches = []
        for fname in tqdm(fnames):
            cid = info_from_filename(fname, True)["case_id"]
            dici = {"case_id": fname}
            patch_fns = self.get_patch_files(self.bboxes, cid)
            dici.update(patch_fns)
            patches.append(dici)
        return patches

    def create_transforms(self, keys="all"):
        super().create_transforms(keys)

    def derive_data_folder(self):
        assert (
            self.plan_train["mode"] == "patch"
        ), f"Dataset mode must be 'patch' for DataManagerPatch, got '{self.plan_train['mode']}'"
        parent_folder = self.project.patches_folder
        plan_name = "plan" + str(self.dataset_params["plan"])
        source_plan_name = self.plan_train["source_plan"]
        source_plan = self.config[source_plan_name]
        spacing = ast_literal_eval(source_plan["spacing"])
        subfldr1 = folder_name_from_list("spc", parent_folder, spacing)
        patch_size = ast_literal_eval(self.plan_train["patch_size"])
        return folder_name_from_list("dim", subfldr1, patch_size, plan_name)

    def setup(self, stage: str = None):
        fgbg_ratio = self.dataset_params["fgbg_ratio"]
        fgbg_ratio_adjusted = fgbg_ratio / self.fg_bg_prior
        self.dataset_params["fgbg_ratio"] = fgbg_ratio_adjusted
        super().setup(stage)

    @property
    def src_dims(self):
        return self.plan_train["patch_size"]


class DataManagerBaseline(DataManagerLBD):
    """
    Small dataset of size =batchsize comprising a single batch. No augmentations. Used to get a baseline
    It has no training augmentations. Whether the flag is True or False doesnt matter.
    Note: It inherits from LBD dataset.
    """

    def __init__(self, project, config: dict, batch_size=8, **kwargs):
        super().__init__(project, config, batch_size, **kwargs)
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
        self.keys_val = "L,Ld,E,Re,N"
        self.keys_tr = self.keys_val

    def derive_data_folder(self, dataset_mode=None):
        assert (
            self.plan_train["mode"] == "baseline"
        ), f"Dataset mode must be 'baseline' for DataManagerBaseline, got '{self.plan_train['mode']}'"
        # return data_folder
        source_plan_name = self.plan_train["source_plan"]
        source_plan = self.config[source_plan_name]

        source_ds_type = source_plan["mode"]
        if source_ds_type == "lbd":
            parent_folder = self.project.lbd_folder
        else:
            raise NotImplemented
        spacing = ast_literal_eval(source_plan["spacing"])

        data_folder = folder_name_from_list(
            "spc", parent_folder, spacing, source_plan_name
        )
        assert data_folder.exists(), "Dataset folder {} does not exists".format(
            data_folder
        )
        return data_folder

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


class DataManagerDual2(DataManagerDual):
    """
    A higher-level DataManager that manages separate training and validation DataManagers
    """

    def __init__(
        self,
        project_title,
        config: dict,
        batch_size=8,
        cache_rate=0.0,
        ds_type=None,
        save_hyperparameters=True,
        keys_tr="Ex,Rva,Affine,F1,F2 ,Re,N,IntensityTfms",
        keys_val="Ex, Rva,Re, N",
        data_folder=None,
    ):
        super().__init__(
            project_title=project_title,
            config=config,
            batch_size=batch_size,
            cache_rate=cache_rate,
            ds_type=ds_type,
            save_hyperparameters=save_hyperparameters,
            keys_val=keys_val,
            keys_tr=keys_tr,
            data_folder=data_folder,
        )


# %%
if __name__ == "__main__":
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>

    import torch

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")
    torch.set_float32_matmul_precision("medium")
    project_title = "litsmc"
    proj_litsmc = Project(project_title=project_title)

    C = ConfigMaker(proj_litsmc)
    C.setup(1)
    conf_litsmc = C.configs

    project_title = "totalseg"
    proj_tot = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    )
    configuration_filename = None

    C2 = ConfigMaker(proj_tot)
    C2.setup(1)
    conf_tot = C2.configs

    global_props = load_dict(proj_tot.global_properties_filename)

    conf_litsmc["plan_train"]["patch_size"] = [256, 256]
# %%

    data_fldr = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_ex070/slices")
    batch_size = 8
    ds_type = "lmdb"
    D = DataManagerDual2(
        project_title=proj_litsmc.project_title,
        config=conf_litsmc,
        batch_size=batch_size,
        ds_type=ds_type,
        data_folder=data_fldr,
    )

    tm = D.train_manager
    tm.data_folder
# %%
    # D.train_manager.plan['patch_size']=[128,128]
    # D.valid_manager.plan['patch_size']

    D.prepare_data()
    D.setup()
    tm = D.train_manager

    # dici = tm.ds[0]
    dl = D.train_dataloader()
    iteri = iter(dl)
    bt = next(iteri)
    bt["image"].shape
    bt["lm"].shape
# %%

    img = bt["image"][0]
    lm = bt["lm"][0]
    ImageMaskViewer([img, lm])
# %%
    tm.plan["patch_size"] = tm.plan["patch_size"][:2]

    E = ExtractContiguousSlicesd()
    dici = E(data)
# %%
    Rva = RandCropByPosNegLabeld(
        keys=["image", "lm"],
        label_key="lm",
        image_key="image",
        image_threshold=-2600,
        spatial_size=tm.plan["patch_size"],
        pos=1,
        neg=1,
        num_samples=1,
        lazy=False,
        allow_smaller=True,
    )
# %%
    data = tm.data[4]
    dici = E(data)
    dici = Rva(dici)
    dici[0]["image"].shape
    # % %%
    img = dici[0]["image"]
    lm = dici[0]["lm"]

# %%
    img = bt["image"][0]
    lm = bt["lm"][0]
    ImageMaskViewer([img, lm])

# %%
    dici = E(dici)
    Ld = LoadImaged()
    dici2 = Ld(dici)
    z = 1

# %%

# %%
    d
    D.setup()

    ds = D.train_ds
# %%
    dici = ds[0]
    dici[0]["image"].shape
    dici[0]["lm"].shape
# %%
    tm = D.train_manager
    tm.data_folder

    tm.cases
# %%

    # Now use train_manager or valid_manager to access the data
    dl = D.train_dataloader()
    bb = D.train_ds[0]

    iteri = iter(dl)
    b = next(iteri)

# %%
    fldr = tm.cache_folder
    fldr = "/s/tmp"
    ds = LMDBDataset(
        data=tm.data,
        transform=tm.transforms,
        cache_dir=fldr,
        db_name=f"{tm.split}_cache",
    )
# %%

    ds = CacheDataset(
        data=tm.data,
        transform=tm.transforms,
        cache_rate=tm.cache_rate,
    )
# %%

    tags = ["proj_title", "case_id", "date", "desc"]
    name = cleanup_fname(fname)
    parts = name.split("_")
    output_dic = {}
    for key, val in zip(tags, parts):
        output_dic[key] = val
    if full_caseid == True:
        output_dic["case_id"] = output_dic["proj_title"] + "_" + output_dic["case_id"]

# %%

    fnames = [strip_extension(fn) for fn in fnames]
    fnames = [fn + ".pt" for fn in fnames]
    fnames = fnames
    images_fldr = self.data_folder / ("images")
    lms_fldr = self.data_folder / ("lms")
    inds_fldr = self.infer_inds_fldr(self.plan)
    images = list(images_fldr.glob("*.pt"))
    data = []
# %%
    fn = Path("litq_48_20200107.pt")
    fn.name
    aa = [im for im in images if fn.name in im.name]
# %%

    images_fldr = tm.data_folder / ("images")
    lms_fldr = tm.data_folder / ("lms")
    inds_fldr = tm.infer_inds_fldr(tm.plan)
    images = list(images_fldr.glob("*.pt"))

# %%

    cases = images
    cases = [strip_extension(fn) for fn in cases]
    cases = [fn + ".pt" for fn in cases]
    cases = cases
    images_fldr = self.data_folder / ("images")
    lms_fldr = self.data_folder / ("lms")
    inds_fldr = self.infer_inds_fldr(self.plan)
    images = list(images_fldr.glob("*.pt"))
    data = []

    for fn in tqdm(cases):
        fn = Path(fn)
        img_fn = find_matching_fn(
            fn.name, images, ["case_id", "date"], allow_multiple_matches=True
        )
        lm_fn = find_matching_fn(
            fn.name, lms_fldr, ["case_id", "date"], allow_multiple_matches=True
        )
        indices_fn = inds_fldr / img_fn.name
        assert img_fn.exists(), "Missing image {}".format(img_fn)
        assert lm_fn.exists(), "Missing labelmap fn {}".format(lm_fn)
        dici = {"image": img_fn, "lm": lm_fn, "indices": indices_fn}
        data.append(dici)
# %%
    case_ = tm.cases[0]
    case_ = cleanup_fname(case_)
    image_subfoldrs = list(tm.data_folder.glob("images/*"))
    matches = [fn for fn in image_subfoldrs if case_ in fn.name]
    image_subfoldrs = [str(f) for f in image_subfoldrs]
    image_subfoldrs = [f.split("images/")[1] for f in image_subfoldrs]

# %%

    cases = tm.cases
    cases = [strip_extension(fn) for fn in cases]
    image_subfoldrs = list(tm.data_folder.glob("images/*"))
    lm_subfoldrs = list(tm.data_folder.glob("lms/*"))

    data = []
    for case_ in pbar(cases):
        img_matched = [fn for fn in image_subfoldrs if case_ == fn.name]
        lm_matched = [fn for fn in lm_subfoldrs if case_ == fn.name]
        assert (
            len(img_matched) == 1 and len(lm_matched) == 1
        ), "Multiple images for case {}".format(case_)
        img_fldr = img_matched[0]
        lm_fldr = lm_matched[0]
        dici = {"image": img_fldr, "lm": lm_fldr}
        data.append(dici)

    tm.data = data

    img_fldr = dici["image"]
    n_slices = len(list(img_fldr.glob("*")))
