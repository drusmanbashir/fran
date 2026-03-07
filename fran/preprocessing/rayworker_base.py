# ray_worker_base.py
import traceback
from typing import Any, Dict

import ipdb
import pandas as pd
from monai.transforms import Compose
from monai.transforms.utility.dictionary import (EnsureChannelFirstd,
                                                 MapLabelValueD, ToDeviced)
from monai.transforms.utils import is_positive
from utilz.cprint import cprint

from fran.configs.parser import parse_excel_datasources, parse_excel_remapping
from fran.transforms.imageio import LoadTorchd
from fran.transforms.misc_transforms import (
    DummyTransform,
    FgBgToIndicesd2,
    GetLabelsd,
)
from fran.transforms.spatialtransforms import CropForegroundMinShaped

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
        debug=False,
        tfms_keys="LoadT,Chan,Dev,Crop,Remap,Labels,Indx",
    ):
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
        )

        self.crop_to_label = crop_to_label  # redundant
        self.debug = debug
        self.image_key = "image"
        self.lm_key = "lm"
        self.tnsr_keys = [self.image_key, self.lm_key]
        self.create_transforms(device=device)
        self.set_transforms(tfms_keys)
        self.tfms_keys = tfms_keys

    def _create_data_dict(self, row):

        raise NotImplementedError

    def _process_row(self, row: pd.Series) -> Dict[str, Any]:
        data = self._create_data_dict(row)

        # Apply transforms
        data = self.apply_transforms(data)
        labels_key = f"{self.lm_key}_labels"
        labels = data.get(labels_key)
        image = data["image"]
        lm = data["lm"]
        lm_fg_indices = data["lm_fg_indices"]
        lm_bg_indices = data["lm_bg_indices"]
        # Get metadata and indices

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
            "n_fg": len(lm_fg_indices),
            "n_bg": len(lm_bg_indices),
            "labels": labels,
        }
        return results




    def apply_transforms(self, data: dict):
        if self.debug==False:
            return self.apply_transforms_compose(data)
        else:
            return self.apply_transforms_debug(data)

    def apply_transforms_compose(self, data: dict):
        data = self.transforms(data)
        return data

    def apply_transforms_debug(self, data: dict):
        keys = self.tfms_keys
        keys = keys.replace(" ", "").split(",")

        for key in keys:
            if self.debug:
                cprint(f"{key}", color = "yellow")
                tr()
            tfm = self.transforms_dict[key]
            if isinstance(tfm, dict):
                ds = data["ds"]
                tfm = tfm[ds]
            data = tfm(data)
        return data

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
            dici = self._create_data_dict(row)
            data.append(dici)
        return data

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
        self.C = CropForegroundMinShaped(
            keys=self.tnsr_keys,
            source_key=self.lm_key,
            select_fn=select_fn,
            margin=margin,
            min_shape=self.plan["patch_size"],
        )
        
        self.Dev = ToDeviced(device=device, keys=self.tnsr_keys)
        self.Chan = EnsureChannelFirstd(keys=self.tnsr_keys, channel_dim="no_channel")
        self.Indx = FgBgToIndicesd2(
            keys=[self.lm_key],
            image_key=self.image_key,
            ignore_labels=self.plan.get("fg_indices_exclude", []),
            image_threshold=-2600,
        )
        self.Labels = GetLabelsd(lm_key=self.lm_key)
        self.LoadT = LoadTorchd(keys=[self.image_key, self.lm_key])
        self.Remap = self.create_monai_remapping_per_ds("remapping_lbd", self.lm_key)

        self.transforms_dict = {
            "Crop": self.C,
            "Dev": self.Dev,
            "Chan": self.Chan,
            "Indx": self.Indx,
            "Labels": self.Labels,
            "LoadT": self.LoadT,
            "Remap": self.Remap,  # dict: datasource -> transform
        }



    def create_monai_remapping_per_ds(self, remapping_key, lm_key) -> dict:
        dss = self.plan["datasources"]
        dss = parse_excel_datasources(dss)
        rem = self.plan[remapping_key]
        rem = parse_excel_remapping(rem)
        Remapping_tfms = {}
        for rez, ds in zip(rem, dss):
            if rez is not None:
                orig = list(rez.keys())
                target = list(rez.values())
                R = MapLabelValueD(
                    keys=[self.lm_key], orig_labels=orig, target_labels=target
                )
            else:
                R = DummyTransform(keys=[lm_key])
            Remapping_tfms[ds] = R
        return Remapping_tfms

    def process(self, mini_df):
        outs = []
        for i, row in mini_df.iterrows():
            try:
                outs.append(self._process_row(row))

            except Exception as e:
                print(
                    f"[{self.__class__.__name__}] error:"
                    f"\n  case_id={row.get('case_id')}"
                    f"\n  image={row.get('image')}"
                    f"\n  lm={row.get('lm')}"
                    f"\n  lm_imported={row.get('lm_imported')}"
                )
                traceback.print_exc()  # <- this is the key
                outs.append(
                    {
                        "case_id": row.get("case_id"),
                        "ok": False,
                        "err": repr(e),
                        "labels": None,
                    }
                )

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
# %%
# parse_excel_remapping
# parse_excel_datasources(self.plan["datasources"])
