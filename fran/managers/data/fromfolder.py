# %%
from __future__ import annotations

import os
from pathlib import Path

from fran.data.dataset import NormaliseClipd
from fran.managers.data.training import DataManager
from fran.transforms.imageio import LoadSITKd
from monai.transforms.spatial.dictionary import (
    Orientationd,
    Spacingd,
)
from monai.transforms.utility.dictionary import (
    EnsureChannelFirstd,
)
from torchvision.datasets.folder import is_image_file
from utilz.fileio import is_sitk_file, load_yaml
from utilz.helpers import (
    find_matching_fn,
)
from utilz.stringz import (
    headline,
)

common_vars_filename = os.environ["FRAN_CONF"] + "/config.yaml"
COMMON_PATHS = load_yaml(common_vars_filename)


class DataManagerTestFF(DataManager):
    def __init__(
        self,
        project,
        configs,
        batch_size,
        device,
        data_folder,
        keys,
        collate_fn,
        remapping_gt=None,
    ):
        self.project = project

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
            self.plan["remapping_train"] = None

    def set_data_folder(self, data_folder):
        self.data_folder = Path(data_folder)

    def prepare_data(self):
        images_folder = self.data_folder / "images"
        lms_folder = self.data_folder / "lms"
        images = [fn for fn in images_folder.glob("*") if is_image_file(fn.name)]

        self.data = []
        for img_fn in images:
            lm_fn = find_matching_fn(
                img_fn.name, lms_folder, ["case_id"], allow_multiple_matches=False
            )[0]

            dici = {"image": img_fn, "lm": lm_fn}
            self.data.append(dici)
        sample_file = self.cases[0]
        self.file_type = "nifti" if is_sitk_file(sample_file) else "pt"

    def create_transforms(self):
        if self.file_type == "nifti":
            self.O = Orientationd(keys=["image", "lm"], axcodes="RAS")
            self.L = LoadSITKd(
                keys=["image", "lm"],
                image_only=True,
                ensure_channel_first=False,
                simple_keys=True,
            )
            self.transforms_dict["L"] = self.L
            self.transforms_dict["O"] = self.O
            self.keys = "L,E,S,N,Remap,ResizeP"  # experimental
        else:
            self.keys = "L,O, E,N,Remap"  # Remap is a dummy transform unless self.plan_train specifies it

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
            "O": Orientationd(keys=["image", "lm"], axcodes="RAS"),  # nOTE RAS
        }

    def setup(self):
        headline(f"Setting up test/valid dataset")
        print("Src Dims: ", self.plan["src_dims"])
        print("Patch Size: ", self.plan["patch_size"])
        keys_test = "L,E,N,Remap,ResizeP"
        self.create_transforms()
        self.set_transforms(self.keys)
