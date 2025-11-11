# ray_worker_base.py
from typing import Any, Dict
from SimpleITK import Not
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.utils import is_positive
import pandas as pd

import ipdb


import pandas as pd
from monai.transforms import Compose
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.utility.dictionary import (EnsureChannelFirstd,
                                                 MapLabelValueD, ToDeviced)
from monai.transforms.utils import is_positive
from utilz.fileio import *
from utilz.helpers import *
from utilz.imageviewers import *

from fran.transforms.imageio import LoadTorchd
from fran.transforms.misc_transforms import (DummyTransform, FgBgToIndicesd2)

MIN_SIZE = 32  # min size in a single dimension of any image

# plain, testable class (NOT a Ray actor)
from typing import Any, Dict

import pandas as pd
from fran.preprocessing.preprocessor import Preprocessor

tr = ipdb.set_trace

from typing import Any, Dict





class RayWorkerBase(Preprocessor):

    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        crop_to_label=None,
        device="cpu",
        tfms_keys  = "LT,E,D,C,R,Ind",
    ):
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
        )

        self.crop_to_label = crop_to_label  # redundant
        self.image_key = "image"
        self.lm_key = "lm"
        self.tnsr_keys = [self.image_key, self.lm_key]
        self.create_transforms(device=device)
        self.set_transforms(tfms_keys)

    def _create_data_dict(self,row):

        raise NotImplementedError

    def _process_row(self, row: pd.Series) -> Dict[str, Any]:
        data = self._create_data_dict(row)

        # Apply transforms
        data = self.transforms(data)
        image = data["image"]
        lm = (data["lm"])
        lm_fg_indices = data["lm_fg_indices"]
        lm_bg_indices = data["lm_bg_indices"]
        # Get metadata and indices
        # Process the case
        assert image.shape == lm.shape, "mismatch in shape"
        assert image.dim() == 4, "images should be cxhxwxd"
        assert image.numel() > MIN_SIZE**3, f"image size is too small {image.shape}"

        inds = {
            "lm_fg_indices": lm_fg_indices,
            "lm_bg_indices": lm_bg_indices,
            "meta": image.meta,
        }

        self.save_indices(inds, self.indices_subfolder)
        self.save_pt(image[0], "images")
        self.save_pt(lm[0], "lms")
        # self.extract_image_props(image)
        results = {
            "case_id": row.get("case_id"),
            "ok": True,
            "shape": list(image.shape),
        }
        return results
    def set_transforms(self, keys_tr: str):
        self.transforms = self.tfms_from_dict(keys_tr)

    def tfms_from_dict(self, keys: str):
        keys = keys.replace(" ", "").split(",")
        tfms = []
        for key in keys:
            tfm = self.transforms_dict[key]
            tfms.append(tfm)
        tfms = Compose(tfms)
        return tfms

    def set_input_output_folders(self, data_folder, output_folder):
        self.data_folder = data_folder
        self.output_folder = output_folder

    def _create_data_dicts_from_df(self, df):
        """Create data dictionaries from DataFrame."""
        data = []
        for index in range(len(df)):
            row = df.iloc[index]
            dici = self._dici_from_df_row(row, remapping)
            data.append(dici)

    def create_transforms(self, device):
        if self.plan["expand_by"]:
            margin = [int(self.plan["expand_by"] / sp) for sp in self.plan["spacing"]]
        else:
            margin = 0
        if self.crop_to_label is None:
            select_fn = is_positive
        else:
            select_fn = lambda lm: lm == self.crop_to_label
        # Transform attributes in alphabetical order
        self.C = CropForegroundd(
            keys=self.tnsr_keys,
            source_key=self.lm_key,
            select_fn=select_fn,
            allow_smaller=True,
            margin=margin,
        )
        self.D = ToDeviced(device=device, keys=self.tnsr_keys)
        self.E = EnsureChannelFirstd(keys=self.tnsr_keys, channel_dim="no_channel")
        self.Ind = FgBgToIndicesd2(
            keys=[self.lm_key],
            image_key=self.image_key,
            ignore_labels=self.plan["fg_indices_exclude"],
            image_threshold=-2600,
        )
        self.LT = LoadTorchd(keys=[self.image_key, self.lm_key])
        if self.plan["remapping_lbd"] is not None:
            self.R = MapLabelValueD(
                keys=[self.lm_key],
                orig_labels=self.plan["remapping_lbd"][0],
                target_labels=self.plan["remapping_lbd"][1],
            )
        else:
            self.R = DummyTransform(keys=[self.lm_key])
        # )
        self.transforms_dict = {
            "C": self.C,
            "D": self.D,
            "E": self.E,
            "Ind": self.Ind,
            "LT": self.LT,
            "R": self.R,
        }

    def process(self, mini_df):
        outs = []
        for i,row in mini_df.iterrows():
            try:
                outs.append(self._process_row(row))
            except Exception as e:
                
                img_fn = row.get("image")
                print(f"[{self.__class__.__name__}] error: {img_fn}: {e}")
                outs.append({"case_id": row.get("case_id"), "ok": False, "err": str(e)})
        return outs


#
# class RayWorkerBase(Preprocessor):
#     def __init__(self, project, plan, data_folder=None, output_folder=None, device="cpu", **kwargs):
#         super().__init__(project=project, plan=plan, data_folder=data_folder, output_folder=output_folder)
#         self.device = device
#         self.extra: Dict[str, Any] = kwargs
#         # subclass will set self.transforms
#         self.create_transforms(device=device)
#         self.create_output_folders()
#
#     def create_transforms(self, device="cpu"):
#         raise NotImplementedError
#
#     def _process_row(self, row: pd.Series) -> Dict[str, Any]:
#         raise NotImplementedError
#
#     # what the actor will call
#     def process(self, mini_df: "pd.DataFrame"):
#         out = []
#         for _, row in mini_df.iterrows():
#             try:
#                 out.append(self._process_row(row))
#             except Exception as e:
#                 name = getattr(row.get("image"), "name", row.get("image"))
#                 print(f"[{self.__class__.__name__}] error: {name}: {e}")
#                 out.append({"case_id": row.get("case_id"), "ok": False, "err": str(e)})
#         return out
