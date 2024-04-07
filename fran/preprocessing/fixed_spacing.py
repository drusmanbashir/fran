# %%
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

from monai.data import Dataset
from fran.data.dataloader import img_lm_metadata_lists_collated
from fran.data.dataset import NormaliseClipd
from fran.managers.project import get_ds_remapping
from fran.preprocessing.datasetanalyzers import bboxes_function_version
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import ChangeDType
from fran.transforms.misc_transforms import (ChangeDtyped, DictToMeta, HalfPrecisiond,
                                             Recast, RemapSITK)
from fran.transforms.spatialtransforms import ResizeToTensord
from fran.utils.common import *
from fran.utils.fileio import load_dict, maybe_makedirs, save_dict, save_json
from fran.utils.helpers import folder_name_from_list, multiprocess_multiarg
from fran.utils.string import strip_extension

def get_tensor_stats( tnsr):
        dic = {
            "max": tnsr.max().item(),
            "min": tnsr.min().item(),
            "median": np.median(tnsr),
        }
        return dic


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


class _Preprocessor(GetAttr):
    _default = "project"

    def __init__(
        self,
        project,
        spacing,
        device="cpu",
    ) -> None:
        store_attr("project,spacing,device")

   
    def save_pt(self, tnsr, subfolder):
        tnsr = tnsr.contiguous()
        fn = Path(tnsr.meta["filename"])
        fn = Path(tnsr.meta["filename"])
        fn_name = strip_extension(fn.name)+".pt"
        fn = self.output_folder / subfolder / fn_name
        torch.save(tnsr, fn)

    def _store_dataset_properties(self):
        resampled_dataset_properties = dict()
        resampled_dataset_properties["dataset_spacing"] = self.spacing
        resampled_dataset_properties["dataset_max"] = self.results[:, 0].max().item()
        resampled_dataset_properties["dataset_min"] = self.results[:, 1].min().item()
        resampled_dataset_properties["dataset_std"] = self.results[:, 1].std().item()
        resampled_dataset_properties["dataset_median"] = np.median(self.results[:, 2])
        resampled_dataset_properties_fname = (
            self.output_folder / "resampled_dataset_properties.json"
        )
        maybe_makedirs(self.output_folder)
        print(
            "Writing preprocessing output properties to {}".format(
                resampled_dataset_properties_fname
            )
        )
        save_dict(resampled_dataset_properties, resampled_dataset_properties_fname)

    def process(
                self,

            ):
                print("resampling dataset to spacing: {0}".format(self.spacing))
                self.create_output_folders()
                self.results =[]
                self.shapes=[]
                for i, batch in pbar(enumerate(self.dl)):
                    images, lms = batch["image"], batch["lm"]
                    for image, lm in zip(images, lms):
                        assert image.shape == lm.shape, "mismatch in shape".format(
                            image.shape, lm.shape
                        )
                        assert image.dim() == 4, "images should be cxhxwxd"

                        self.save_pt(image[0], "images")
                        self.save_pt(lm[0], "lms")
                        self.results.append(get_tensor_stats(image))
                        self.shapes.append(image.shape[1:])

                self.results = pd.DataFrame(self.results).values
                if self.results.shape[-1] == 3:  # only store if entire dset is processed
                    self._store_dataset_properties()
                    generate_bboxes_from_lms_folder(self.output_folder/("lms"))
                else:
                    print(
                        "since some files skipped, dataset stats are not being stored. run resampledatasetniftitotorch.get_tensor_folder_stats separately"
                    )

    def get_tensor_folder_stats(self, debug=True):
        img_filenames = (self.output_folder / ("images")).glob("*")
        args = [[img_fn] for img_fn in img_filenames]
        results = multiprocess_multiarg(get_tensorfile_stats, args, debug=debug)
        self.results = pd.DataFrame(results).values
        self._store_dataset_properties()

    def _store_dataset_properties(self):
        resampled_dataset_properties = self.create_properties_dict()

        resampled_dataset_properties_fname = (
            self.output_folder / "resampled_dataset_properties.json"
        )
        maybe_makedirs(self.output_folder)
        print(
            "Writing preprocessing output properties to {}".format(
                resampled_dataset_properties_fname
            )
        )
        save_dict(resampled_dataset_properties, resampled_dataset_properties_fname)

    def create_properties_dict(self):
        self.shapes = np.array(self.shapes)
        resampled_dataset_properties = dict()
        resampled_dataset_properties["median_shape"] =np.median(self.shapes,0).tolist()
        resampled_dataset_properties["dataset_spacing"] = self.spacing
        resampled_dataset_properties["dataset_max"] = self.results[:, 0].max().item()
        resampled_dataset_properties["dataset_min"] = self.results[:, 1].min().item()
        resampled_dataset_properties["dataset_std"] = self.results[:, 1].std().item()
        resampled_dataset_properties["dataset_median"] = np.median(self.results[:, 2])
        return resampled_dataset_properties


    def create_output_folders(self):
        maybe_makedirs([self.output_folder / ("lms"), self.output_folder / ("images")])

class ResampleDatasetniftiToTorch(_Preprocessor):
    def __init__(self, project, spacing, device="cpu",half_precision=False):
        super().__init__(project, spacing, device=device)
        self.output_folder = folder_name_from_list(
            prefix="spc",
            parent_folder=self.fixed_spacing_folder,
            values_list=spacing,
        )
        self.half_precision = half_precision

    def create_dl(self,debug=False):
        self.ds = ResamplerDataset(
            project=self.project,
            spacing=self.spacing,
            half_precision=self.half_precision,
            device=self.device,
        )
        self.ds.setup()
        self.dl = DataLoader(
            dataset=self.ds,
            num_workers=4,
            collate_fn=img_lm_metadata_lists_collated,
            batch_size=4 if debug == False else 1,
        )

    def process(
        self,

    ):

        super().process()
        update_resampling_configs(self.spacing, self.output_folder)

    def generate_bboxes_from_masks_folder(
        self, bg_label=0, debug=False, num_processes=8
    ):
        masks_folder = self.output_folder / ("lms")
        print("Generating bbox info from {}".format(masks_folder))
        generate_bboxes_from_lms_folder(
            masks_folder,
            bg_label,
            debug,
            num_processes,
        )

    def update_specsfile(self):
        lbd_output_folder = self.project.lbd_folder/(self.output_folder.name)
        specs = {
            "spacing": self.spacing,
            "output_folder": str(self.output_folder),
            "lbd_output_folder":str(lbd_output_folder)
        }
        specs_file = self.output_folder.parent / ("resampling_configs")

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
        img_filenames = (self.output_folder / ("images")).glob("*")
        args = [[img_fn] for img_fn in img_filenames]
        results = multiprocess_multiarg(get_tensorfile_stats, args, debug=debug)
        self.results = pd.DataFrame(results).values
        self._store_dataset_properties()


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
            if dic["output_folder"].exists():
                print(str(dic["output_folder"]) + " verified..")
                output_specs.append(dic)
            else:
                print(
                    str(dic["output_folder"])
                    + " does not exist. Removing from specs"
                )
        save_pickle(output_specs, resampling_configs_fn)
    except:
        print("Resampling configs file either does not exist or is invalid")


def update_resampling_configs(spacing, output_folder):
    specs = {
        "spacing": spacing,
        "output_folder": str(output_folder),
    }
    specs_file = output_folder.parent / ("resampling_configs.json")
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

    def setup(self):
        self.create_transforms()
        data = self.create_data_dicts()
        Dataset.__init__(self, data,self.transform)


    def create_data_dicts(self):
        data = []
        for index in range(len(self)):
            cp = self.df.iloc[index]
            ds = cp["ds"]
            remapping = get_ds_remapping(ds, self.global_properties)

            img_fname = cp["image"]
            mask_fname = cp["lm"]
            dici = {
                "image": img_fname,
                "lm": mask_fname,
                "remapping": remapping,
            }
            data.append(dici)
        return data



    def filter_completed_cases(self):
        df = self.project.df.copy()  # speed up things
        tr()
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        dici = self.data[index]
        img_fname = dici["image"]
        mask_fname = dici["lm"]
        remapping = dici['remapping']
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
        Rz = ResizeToTensord(keys=["lm"], key_template_tensor="image", mode="nearest")

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

    project = Project("lidc2")
    Rx = ResamplerDataset(project, spacing=[.8,.8,1.5])

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
    I.Resampler.shapes = np.array(I.Resampler.shapes)
    fn_dict = I.Resampler.output_folder / "info.json"

    dici = {
        "median_shape":np.median(I.Resampler.shapes,0).tolist(),
        "spacing": I.Resampler.spacing,
    }
    save_dict(dici,fn_dict)



    resampled_dataset_properties["median_shape"] =np.median(I.Resampler.shapes,0).tolist()
# %%
    tnsr = torch.load("/s/fran_storage/datasets/preprocessed/fixed_spacing/litsmc/spc_080_080_150/lms/drli_001ub.nii.gz")
    fn = Path(tnsr.meta["filename"])
    fn_name = strip_extension(fn.name)+".pt"
    fn_out = fn.parent/(fn_name)
    generate_bboxes_from_lms_folder(I.Resampler.output_folder/("lms"))
