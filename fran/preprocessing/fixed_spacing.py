# %%
import itertools
import operator
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from fastcore.all import Union, save_pickle, store_attr
from fastcore.foundation import GetAttr
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import (EnsureChannelFirstd,
                                                 FgBgToIndicesd, ToDeviced)
from torch.utils.data import Dataset

from fran.data.dataloader import img_lm_metadata_lists_collated
from fran.data.dataset import NormaliseClipd
from fran.managers.project import get_ds_remapping
from fran.preprocessing.datasetanalyzers import bboxes_function_version
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import ChangeDType
from fran.transforms.misc_transforms import (ChangeDtyped, DictToMeta, HalfPrecisiond,
                                             Recast, RemapSITK)
from fran.transforms.spatialtransforms import ResizeDynamicd
from fran.utils.common import *
from fran.utils.fileio import load_dict, maybe_makedirs, save_dict, save_json
from fran.utils.helpers import folder_name_from_list, multiprocess_multiarg
from fran.utils.string import strip_extension


def generate_bboxes_from_lms_folder(
    masks_folder, bg_label=0, debug=False, num_processes=16
):
    label_files = masks_folder.glob("*pt")
    arguments = [
        [x, bg_label] for x in label_files
    ]  # 0.2 factor for thresholding as kidneys are small on low-res imaging and will be wiped out by default threshold 3000
    bboxes = multiprocess_multiarg(
        func=bboxes_function_version,
        arguments=arguments,
        num_processes=num_processes,
        debug=debug,
    )
    bbox_fn = masks_folder.parent / ("bboxes_info")
    print("Storing bbox info in {}".format(bbox_fn))
    save_dict(bboxes, bbox_fn)


class ResampleDatasetniftiToTorch(GetAttr):
    _default = "project"

    def __init__(
        self,
        project,
        minimum_final_spacing,
        enforce_isotropy=True,
        half_precision=False,
        clip_centre=True,
        device="cpu",
    ) -> None:
        """
        minimum_final_spacing is only used when enforce_isotropy is True
        """
        store_attr("project,half_precision,clip_centre,device")

        if enforce_isotropy == True:
            self.spacing = [
                np.maximum(
                    minimum_final_spacing,
                    np.mean(self.global_properties["spacing_median"][1:]),
                ),
            ] * 3  # ignores first index (z) and averages over x and y
            print(
                "Enfore isotropy is true. Setting same spacing based on dataset medians"
            )
        else:
            print("Enfore isotropy is False. Setting spacing based on dataset medians")
            self.spacing = self.global_properties["spacing_median"]

    def resample_cases(
        self,
        multiprocess=True,
        num_processes=8,
        overwrite=False,
        debug=False,
    ):
        print("Resampling dataset to spacing: {0}".format(self.spacing))
        output_subfolders = [
            self.resampling_output_folder / ("images"),
            self.resampling_output_folder / ("lms"),
        ]
        maybe_makedirs(output_subfolders)

        ds = ResamplerDataset(
            project=self.project,
            spacing=self._spacing,
            half_precision=self.half_precision,
            device=self.device,
        )
        dl = DataLoader(
            dataset=ds,
            num_workers=4,
            collate_fn=img_lm_metadata_lists_collated,
            batch_size=4 if debug == False else 1,
        )
        self.results = []
        for id, batch in enumerate(dl):
            images, lms = batch["image"], batch["lm"]
            for img, lm in zip(images, lms):
                assert img.shape == lm.shape, "Mismatch in shape".format(
                    img.shape, lm.shape
                )
                assert img.dim() == 4, "Images should be CxHxWxD"
                self.save_tensor(img[0], output_subfolders[0], overwrite)
                self.save_tensor(lm[0], output_subfolders[1], overwrite)
                self.results.append(self.get_tensor_stats(img))
        self.results = pd.DataFrame(self.results).values
        if self.results.shape[-1] == 3:  # only store if entire dset is processed
            self._store_resampled_dataset_properties()
        else:
            print(
                "Since some files skipped, dataset stats are not being stored. Run ResampleDatasetniftiToTorch.get_tensor_folder_stats separately"
            )
        update_resampling_configs(self.spacing, self.resampling_output_folder)

    def save_tensor(self, tnsr, output_folder, overwrite):
        tnsr = tnsr.contiguous()
        fname = Path(tnsr.meta["filename"])
        fname_name = strip_extension(fname.name) + ".pt"
        fname_full = output_folder / fname_name
        if overwrite == True or not fname_full.exists():
            print("Writing preprocess tensor to ", fname_full)
            torch.save(tnsr, fname_full)

    def get_tensor_stats(self, tnsr):
        dic = {
            "max": tnsr.max().item(),
            "min": tnsr.min().item(),
            "median": np.median(tnsr),
        }
        return dic

    def _store_resampled_dataset_properties(self):
        resampled_dataset_properties = dict()
        resampled_dataset_properties["dataset_spacing"] = self.spacing
        resampled_dataset_properties["dataset_max"] = self.results[:, 0].max().item()
        resampled_dataset_properties["dataset_min"] = self.results[:, 1].min().item()
        resampled_dataset_properties["dataset_std"] = self.results[:, 1].std().item()
        resampled_dataset_properties["dataset_median"] = np.median(self.results[:, 2])
        resampled_dataset_properties_fname = (
            self.resampling_output_folder / "resampled_dataset_properties.json"
        )
        maybe_makedirs(self.resampling_output_folder)
        print(
            "Writing preprocessing output properties to {}".format(
                resampled_dataset_properties_fname
            )
        )
        save_dict(resampled_dataset_properties, resampled_dataset_properties_fname)

    def generate_bboxes_from_masks_folder(
        self, bg_label=0, debug=False, num_processes=8
    ):
        masks_folder = self.resampling_output_folder / ("lms")
        print("Generating bbox info from {}".format(masks_folder))
        generate_bboxes_from_lms_folder(
            masks_folder,
            bg_label,
            debug,
            num_processes,
        )

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, spacing: Union[list, np.ndarray]):
        self._spacing = spacing
        self.resampling_output_folder = spacing

    @property
    def resampling_output_folder(self):
        """The resampling_output_folder property."""
        return self._resampling_output_folder

    @resampling_output_folder.setter
    def resampling_output_folder(self, value):
        if isinstance(value, (int, float)):
            value = [
                value,
            ] * 3
        assert all(
            [isinstance(value, (list, tuple)), len(value) == 3]
        ), "Provide a list with x,y,z spacing"
        self._resampling_output_folder = folder_name_from_list(
            prefix="spc",
            parent_folder=self.fixed_spacing_folder,
            values_list=value,
        )
        print(
            "Based on output spacing {0},\n setting resampling output folder to : {1}".format(
                self.spacing, self._resampling_output_folder
            )
        )

    def update_specsfile(self):
        specs = {
            "spacing": self.spacing,
            "resampling_output_folder": str(self.resampling_output_folder),
        }
        specs_file = self.resampling_output_folder.parent / ("resampling_configs")

        try:
            saved_specs = load_dict(specs_file)
            matches = [specs == dic for dic in saved_specs]
            if not any(matches):
                saved_specs.append(specs)
                save_dict(saved_specs, specs_file)
        except:
            saved_specs = [specs]
            save_dict(saved_specs, specs_file)

    def get_tensor_folder_stats(self, debug=True):
        img_filenames = (self.resampling_output_folder / ("images")).glob("*")
        args = [[img_fn] for img_fn in img_filenames]
        results = multiprocess_multiarg(get_tensorfile_stats, args, debug=debug)
        self.results = pd.DataFrame(results).values
        self._store_resampled_dataset_properties()


def get_tensorfile_stats(filename):
    tnsr = torch.load(filename)
    return get_tensor_stats(tnsr)


def get_tensor_stats(tnsr):
    dic = {
        "max": tnsr.max().item(),
        "min": tnsr.min().item(),
        "median": np.median(tnsr),
    }
    return dic


def verify_resampling_configs(resampling_configs_fn):
    output_specs = []
    try:
        saved_specs = load_dict(resampling_configs_fn)
        print(
            "Verifying existing spacing configurations and deleting defunct entries if needed."
        )
        print(
            "Number of fixed_spacing configurations on file: {}".format(
                len(saved_specs)
            )
        )
        for dic in saved_specs:
            if dic["resampling_output_folder"].exists():
                print(str(dic["resampling_output_folder"]) + " verified..")
                output_specs.append(dic)
            else:
                print(
                    str(dic["resampling_output_folder"])
                    + " does not exist. Removing from specs"
                )
        save_pickle(output_specs, resampling_configs_fn)
    except:
        print("Resampling configs file either does not exist or is invalid")


def update_resampling_configs(spacing, resampling_output_folder):
    specs = {
        "spacing": spacing,
        "resampling_output_folder": str(resampling_output_folder),
    }
    specs_file = resampling_output_folder.parent / ("resampling_configs.json")
    verify_resampling_configs(specs_file)
    try:
        output_specs = load_dict(specs_file)
    except:
        print("Creating new reesampling configs file.")
        output_specs = []
    matches = [specs == dic for dic in output_specs]
    if not any(matches):
        output_specs.append(specs)
        save_json(output_specs, specs_file)
    else:
        print("Set of specs already exist in a folder. Nothing is changed.")


class ResamplerDataset(GetAttr, Dataset):
    _default = "project"

    def __init__(
        self,
        project,
        spacing,
        half_precision=False,
        clip_center=False,
        store_label_inds=False,
        mean_std_mode: str = "dataset",
        device="cuda",
    ):

        assert mean_std_mode in [
            "dataset",
            "fg",
        ], "Select either dataset mean/std or fg mean/std for normalization"
        self.project = project
        self.df = self.filter_completed_cases()
        self.spacing = spacing
        self.half_precision = half_precision
        self.clip_center = clip_center
        self.device = device
        super(GetAttr).__init__()
        self.set_normalization_values(mean_std_mode)
        self.create_transforms()

    def filter_completed_cases(self):
        df = self.project.df.copy()  # speed up things
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        cp = self.df.iloc[index]
        ds = cp["ds"]
        remapping = get_ds_remapping(ds, self.global_properties)

        img_fname = cp["image"]
        mask_fname = cp["lm"]
        img = sitk.ReadImage(img_fname)
        mask = sitk.ReadImage(mask_fname)
        dici = {
            "image": img,
            "lm": mask,
            "image_fname": img_fname,
            "lm_fname": mask_fname,
            "remapping": remapping,
        }
        dici = self.transform(dici)
        return dici

    def create_transforms(self):
        R = RemapSITK(keys=["lm"], remapping_key="remapping")
        L = LoadSITKd(keys=["image", "lm"], image_only=True)
        T = ToDeviced(keys=["image", "lm"], device=self.device)
        Re = Recast(keys=["image", "lm"])

        Ind = FgBgToIndicesd(keys=["lm"], image_key="image", image_threshold=-2600)
        Ai = DictToMeta(
            keys=["image"], meta_keys=["image_fname"], renamed_keys=["filename"]
        )
        Am = DictToMeta(
            keys=["lm"],
            meta_keys=["lm_fname", "remapping", "lm_fg_indices", "lm_bg_indices"],
            renamed_keys=["filename", "remapping", "lm_fg_indices", "lm_bg_indices"],
        )
        E = EnsureChannelFirstd(
            keys=["image", "lm"], channel_dim="no_channel"
        )  # funny shape output mismatch
        Si = Spacingd(keys=["image"], pixdim=self.spacing, mode="trilinear")
        Rz = ResizeDynamicd(keys=["lm"], key_spatial_size="image", mode="nearest")

        # Sm = Spacingd(keys=["lm"], pixdim=self.spacing,mode="nearest")
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.global_properties["intensity_clip_range"],
            mean=self.mean,
            std=self.std,
        )
        Ch = ChangeDtyped(keys=['lm'],target_dtype = torch.uint8)

        

        tfms = [R, L, T, Re, Ind, Ai, Am, E, Si, Rz,Ch]

        if self.clip_center == True:
            tfms.extend([N])
        if self.half_precision == True:
            H = HalfPrecisiond(keys=["image"])
            tfms.extend([H])
        self.transform = Compose(tfms)

    def set_normalization_values(self, mean_std_mode):
        if mean_std_mode == "dataset":
            self.mean = self.global_properties["mean_dataset_clipped"]
            self.std = self.global_properties["std_dataset_clipped"]
        else:
            self.mean = self.global_properties["mean_fg"]
            self.std = self.global_properties["std_fg"]


# %%
if __name__ == "__main__":
    from fran.utils.common import *

    project = Project("tmp")
    Rx = ResamplerDataset(project, spacing=[1.5, 1.5, 1.5])

# %%
    index = 0

    cp = Rx.df.iloc[index]
    ds = cp["ds"]
    remapping = get_ds_remapping(ds, Rx.global_properties)

    img_fname = cp["image"]
    mask_fname = cp["lm"]
    img = sitk.ReadImage(img_fname)
    mask = sitk.ReadImage(mask_fname)
    dici = {
        "image": img,
        "lm": mask,
        "image_fname": img_fname,
        "lm_fname": mask_fname,
        "remapping": remapping,
    }

    dici = Rx.transform(dici)
# %%
    R = RemapSITK(keys=["lm"], remapping_key="remapping")
    L = LoadSITKd(keys=["image", "lm"], image_only=True)
    T = ToDeviced(keys=["image", "lm"], device=Rx.device)
    Re = Recast(keys=["image", "lm"])

    Ind = FgBgToIndicesd(keys=["lm"], image_key="image", image_threshold=-2600)
    Ai = DictToMeta(
        keys=["image"], meta_keys=["image_fname"], renamed_keys=["filename"]
    )
    Am = DictToMeta(
        keys=["lm"],
        meta_keys=["lm_fname", "remapping", "lm_fb_indices", "lm_fg_indices"],
        renamed_keys=["filename", "remapping", "lm_fb_indices", "lm_fg_indices"],
    )
    E = EnsureChannelFirstd(
        keys=["image", "lm"], channel_dim="no_channel"
    )  # funny shape output mismatch
    Si = Spacingd(keys=["image"], pixdim=Rx.spacing, mode="trilinear")
    Rz = ResizeDynamicd(keys=["lm"], key_spatial_size="image", mode="nearest")

    # Sm = Spacingd(keys=["lm"], pixdim=Rx.spacing,mode="nearest")
    N = NormaliseClipd(
        keys=["image"],
        clip_range=Rx.global_properties["intensity_clip_range"],
        mean=Rx.mean,
        std=Rx.std,
    )

    tf = Compose([R, L, T, Re, Ind])
    dici = tf(dici)
# %%
