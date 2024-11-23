# %%
from pathlib import Path
import ipdb

tr = ipdb.set_trace

from fran.preprocessing.preprocessor import Preprocessor
from fran.transforms.intensitytransforms import NormaliseClipd

import numpy as np
import pandas as pd
import torch
from fastcore.all import store_attr
from fastcore.foundation import GetAttr
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.spatial.dictionary import Spacingd
from monai.transforms.utility.dictionary import (
    EnsureChannelFirstd,
    FgBgToIndicesd,
    ToDeviced,
)

from fran.data.collate import dict_list_collated
from fran.preprocessing.dataset import ResamplerDataset
from fran.preprocessing import bboxes_function_version
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import ToCPUd
from fran.transforms.misc_transforms import (
    ChangeDtyped,
    DictToMeta,
    FgBgToIndicesd2,
    LabelRemapd,
    Recastd,
)
from fran.transforms.spatialtransforms import ResizeToTensord
from fran.utils.common import *
from fran.utils.fileio import load_dict, maybe_makedirs, save_dict, save_json
from fran.utils.helpers import create_df_from_folder, folder_name_from_list, multiprocess_multiarg, pbar
from fran.utils.string import info_from_filename, strip_extension


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

class ResampleDatasetniftiToTorch(Preprocessor):
    def __init__(self, project, spacing, data_folder=None, output_folder=None, half_precision=False):
        super().__init__(project, spacing, output_folder=output_folder,data_folder=data_folder)
        self.half_precision = half_precision

    def setup(self, device="cpu",overwrite=False):
        self.register_existing_files()
        if overwrite == False:
            self.remove_completed_cases()
        if len(self.df) > 0:
            self.ds = ResamplerDataset(
                df=self.df,
                project=self.project,
                spacing=self.spacing,
                half_precision=self.half_precision,
                device=device,
            )
            self.ds.setup()
            self.create_dl()

    def create_dl(self, num_workers=1, batch_size=4):
        # same function as labelbounded
        self.dl = DataLoader(
            dataset=self.ds,
            num_workers=4,
            collate_fn=dict_list_collated(
                keys=[
                    "image",
                    "lm",
                    "lm_fg_indices",
                    "lm_bg_indices",
                ]
            ),
            # collate_fn=img_lm_metadata_lists_collated,
            batch_size=batch_size,
            pin_memory=False,
        )

    def maybe_load_bboxes(self):
        fixed_sp_bboxes_fn = self.output_folder / ("bboxes_info.pkl")
        if fixed_sp_bboxes_fn.exists():
            bboxes = load_dict(fixed_sp_bboxes_fn)
        else:
            bboxes = None
        return bboxes

    def remove_completed_cases(self):
        # remove cases only if bboxes have been created
        existing_fnames = [fn.name for fn in self.existing_files]
        self.df = self.df.copy()
        for i in range(len(self.df)):
            row = self.df.loc[i]
            df_fname = Path(row.lm_symlink)
            df_fname = strip_extension(df_fname.name) + ".pt"
            if df_fname in existing_fnames:
                self.df.drop(i, inplace=True)

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
        lbd_output_folder = self.project.lbd_folder / (self.output_folder.name)
        specs = {
            "spacing": self.spacing,
            "output_folder": str(self.output_folder),
            "lbd_output_folder": str(lbd_output_folder),
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
    #
    # def get_tensor_folder_stats(self, debug=True):
    #     img_filenames = (self.output_folder / ("images")).glob("*")
    #     args = [[img_fn] for img_fn in img_filenames]
    #     results = multiprocess_multiarg(get_tensorfile_stats, args, debug=debug)
    #     self.results = pd.DataFrame(results).values
    #     self._store_dataset_properties()

    def set_input_output_folders(self,data_folder,output_folder):
        if output_folder is not None:
            self.output_folder = output_folder
        self.output_folder = folder_name_from_list(
            prefix="spc",
            parent_folder=self.fixed_spacing_folder,
            values_list=self.spacing,
        )
        self.data_folder = data_folder



class FGBGIndicesResampleDataset(ResampleDatasetniftiToTorch):
    def __init__(self, project, spacing,  half_precision=False):
        super().__init__(project, spacing, half_precision)

    def register_existing_files(self):
        self.existing_files = list(self.indices_subfolder.glob("*"))

    def process(
        self,
    ):
        if not hasattr(self, "dl"):
            print("No data loader created. No data to be processed")
            return 0
        print("resampling dataset to spacing: {0}".format(self.spacing))
        self.create_output_folders()
        for batch in pbar(self.dl):
            self.process_batch(batch)

    def process_batch(self, batch):
        images, lms, fg_inds, bg_inds = (
            batch["image"],
            batch["lm"],
            batch["lm_fg_indices"],
            batch["lm_bg_indices"],
        )
        for (
            image,
            lm,
            fg_ind,
            bg_ind,
        ) in zip(
            images,
            lms,
            fg_inds,
            bg_inds,
        ):
            assert image.shape == lm.shape, "mismatch in shape".format(
                image.shape, lm.shape
            )
            assert image.dim() == 4, "images should be cxhxwxd"
            inds = {
                "lm_fg_indices": fg_ind,
                "lm_bg_indices": bg_ind,
                "meta": image.meta,
            }
            self.save_indices(inds, self.indices_subfolder)


# %%
if __name__ == "__main__":
    from fran.utils.common import *

    from fran.managers import Project
    project = Project("litsmc")

# %%
#SECTION:-------------------- ResampleDatasetniftiToTorch--------------------------------------------------------------------------------------
    Rs = ResampleDatasetniftiToTorch(project, spacing=[0.8, 0.8, 1.5], data_folder="/s/xnat_shadow/crc/",output_folder="/s/xnat_shadow/crc/tensors/fixed_spacing/")
    Rs.setup(True)
    Rs.process()
# %%
    F = FGBGIndicesResampleDataset(project, spacing=[0.8, 0.8, 1.5] )
    F.setup()
    F.process()
    # R.register_existing_files()

    # %%

    # %%
    L = LoadSITKd(keys=["image", "lm"], image_only=True)
    R = LabelRemapd(keys=["lm"], remapping_key="remapping")
    T = ToDeviced(keys=["image", "lm"], device=Rs.ds.device)
    Re = Recastd(keys=["image", "lm"])

    Ind = FgBgToIndicesd2(keys=["lm"], image_key="image", image_threshold=-2600)
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
    Si = Spacingd(keys=["image"], pixdim=Rs.ds.spacing, mode="trilinear")
    Rz = ResizeToTensord(keys=["lm"], key_template_tensor="image", mode="nearest")

    # Sm = Spacingd(keys=["lm"], pixdim=Rs.ds.spacing,mode="nearest")
    N = NormaliseClipd(
        keys=["image"],
        clip_range=Rs.ds.global_properties["intensity_clip_range"],
        mean=Rs.ds.mean,
        std=Rs.ds.std,
    )
    Ch = ChangeDtyped(keys=["lm"], target_dtype=torch.uint8)

    # tfms = [R, L, T, Re, Ind, Ai, Am, E, Si, Rz,Ch]
    tfms = [L, R, T, Re, Ind, E, Si, Rz, Ch]
    # %%
    dici = Rs.ds[0]
    dici["lm"].meta
    dici = Rs.ds.data[0]

    # %%
    dici = L(dici)

    dici = R(dici)
    dici = T(dici)
    dici = Re(dici)
    dici = Ind(dici)
    dici = E(dici)
    dici = Si(dici)
    dici = Rz(dici)
    dici = Ch(dici)

    # %%
    print(dici["image"].meta["filename_or_obj"], dici["lm"].meta["filename_or_obj"])

    L = LoadSITKd(keys=["image", "lm"], image_only=True)
    T = ToDeviced(keys=["image", "lm"], device=Rx.device)
    Re = Recastd(keys=["image", "lm"])

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
        "median_shape": np.median(I.Resampler.shapes, 0).tolist(),
        "spacing": I.Resampler.spacing,
    }
    save_dict(dici, fn_dict)

    resampled_dataset_properties["median_shape"] = np.median(
        I.Resampler.shapes, 0
    ).tolist()
    # %%
    tnsr = torch.load(
        "/s/fran_storage/datasets/preprocessed/fixed_spacing/litsmc/spc_080_080_150/lms/drli_001ub.nii.gz"
    )
    fn = Path(tnsr.meta["filename"])
    fn_name = strip_extension(fn.name) + ".pt"
    fn_out = fn.parent / (fn_name)
    generate_bboxes_from_lms_folder(R.output_folder / ("lms"))
    # %%

    existing_fnames = [fn.name for fn in R.existing_files]
    df = R.df.copy()
    rows_new = []
    # %%
    for i in range(len(df)):
        row = df.loc[i]
        df_fname = Path(row.lm_symlink)
        df_fname = strip_extension(df_fname.name) + ".pt"
        if df_fname in existing_fnames:
            df.drop([i], inplace=True)
            # rows_new.append(row)

        # %%
        L.shapes = np.array(L.shapes)
        resampled_dataset_properties = dict()
        resampled_dataset_properties["median_shape"] = np.median(L.shapes, 0).tolist()
        resampled_dataset_properties["dataset_spacing"] = L.spacing
        resampled_dataset_properties["dataset_max"] = L.results["max"].max().item()
        resampled_dataset_properties["dataset_min"] = L.results["min"].min().item()
        resampled_dataset_properties["dataset_median"] = np.median(
            L.results["median"]
        ).item()

    # %%
    df = pd.DataFrame(np.arange(12).reshape(3, 4), columns=["A", "B", "C", "D"])
    df.drop(["A", "B"], axis=1)
    df
    dici = load_dict()
    save_json(
        resampled_dataset_properties,
        "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_plan3/resampled_dataset_properties.json",
    )
    # %%
    dl = I.R.dl
    iteri = iter(dl)
    batch = next(iteri)
    # %%
    U = ToCPUd(keys=["image", "lm", "lm_fg_indices", "lm_bg_indices"])
    batch = U(batch)
    images, lms, fg_inds, bg_inds = (
        batch["image"],
        batch["lm"],
        batch["lm_fg_indices"],
        batch["lm_bg_indices"],
    )
    for (
        image,
        lm,
        fg_ind,
        bg_ind,
    ) in zip(
        images,
        lms,
        fg_inds,
        bg_inds,
    ):
        assert image.shape == lm.shape, "mismatch in shape".format(
            image.shape, lm.shape
        )
        assert image.dim() == 4, "images should be cxhxwxd"

        inds = {
            "lm_fg_indices": fg_ind,
            "lm_bg_indices": bg_ind,
            "meta": image.meta,
        }
        I.R.save_indices(inds, I.R.indices_subfolder)
        I.R.save_pt(image[0], "images")
        I.R.save_pt(lm[0], "lms")
        I.R.extract_image_props(image)


# %%
