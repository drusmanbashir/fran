# ray_worker_base.py
import traceback
from pathlib import Path
from typing import Any, Dict

import ipdb
import pandas as pd
from fran.configs.parser import parse_excel_datasources, parse_excel_remapping
from fran.data.dataregistry import DS
from fran.transforms.fg_indices import FgBgToIndicesSubsampled, FgBgToIndicesd2
from fran.transforms.imageio import LoadTorchd
from fran.transforms.misc_transforms import (
    DummyTransform,
    GetLabelsd,
)
from fran.transforms.spatialtransforms import CropForegroundMinShaped
from monai.transforms import Compose
from monai.transforms.utility.dictionary import (
    EnsureChannelFirstd,
    MapLabelValueD,
    ToDeviced,
)
from monai.transforms.utils import is_positive
from utilz.cprint import cprint
from utilz.stringz import strip_extension

MIN_SIZE = 32  # min size in a single dimension of any image

# plain, testable class (NOT a Ray actor)
from typing import Any, Dict

import pandas as pd
from fran.preprocessing.preprocessor import Preprocessor

tr = ipdb.set_trace

from typing import Any, Dict


class RayWorkerBase(Preprocessor):
    remapping_key = None

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
        remapping_key=None,
    ):
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
        )
        self.remapping_key = remapping_key or self.remapping_key
        self.crop_to_label = crop_to_label  # redundant
        self.debug = debug
        self.image_key = "image"
        self.lm_key = "lm"
        self.tnsr_keys = [self.image_key, self.lm_key]
        self.tfms_keys = tfms_keys
        self.create_transforms(device=device)
        self.set_transforms(tfms_keys)

    def _create_data_dict(self, row):

        raise NotImplementedError

    def _process_row(self, row: pd.Series) -> Dict[str, Any]:
        data = self._create_data_dict(row)

        # Apply transforms
        data = self.apply_transforms(data)
        preprocess_events = self._normalize_preprocess_events(
            data.get("_preprocess_events")
        )
        labels_key = f"{self.lm_key}_labels"
        labels = data.get(labels_key)
        image_src = row.get("image")
        fn_name = ""
        if image_src is not None:
            fn_name = strip_extension(Path(str(image_src)).name) + ".pt"
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
            "fn_name": fn_name,
            "ok": True,
            "shape": list(image.shape),
            "n_fg": len(lm_fg_indices),
            "n_bg": len(lm_bg_indices),
            "labels": labels,
            "_preprocess_events": preprocess_events,
        }
        return results

    @staticmethod
    def _normalize_preprocess_events(events):
        if events is None:
            return []
        if isinstance(events, dict):
            events = [events]
        if not isinstance(events, list):
            return []
        normalized = []
        for event in events:
            if not isinstance(event, dict):
                continue
            error_type = event.get("error_type")
            error_message = event.get("error_message")
            if error_type is None and error_message is None:
                continue
            normalized.append(
                {
                    "error_type": "WARNING" if error_type is None else str(error_type),
                    "error_message": "" if error_message is None else str(error_message),
                }
            )
        return normalized

    def apply_transforms(self, data: dict):
        if self.debug == False:
            return self.apply_transforms_compose(data)
        else:
            return self.apply_transforms_debug(data)

    def apply_transforms_compose(self, data: dict):
        tfm = self.transforms
        if isinstance(tfm, dict):
            ds = data.get("ds")
            if ds is None:
                raise KeyError(
                    "Datasource key 'ds' is missing in data dict; cannot choose per-ds Compose."
                )
            if ds not in tfm:
                raise KeyError(
                    f"No composed transform found for ds='{ds}'. Available: {sorted(tfm.keys())}"
                )
            tfm = tfm[ds]
        data = tfm(data)
        return data

    def apply_transforms_debug(self, data: dict):
        keys = self.tfms_keys
        keys = keys.replace(" ", "").split(",")

        for key in keys:
            if self.debug:
                cprint(f"{key}", color="yellow")
                tr()
            tfm = self.transforms_dict[key]
            if isinstance(tfm, dict):
                ds = data["ds"]
                tfm = tfm[ds]
            if isinstance(data, list | tuple):
                data = data[0]
            data = tfm(data)
        return data

    def set_transforms(self, keys_tr: str):
        self.transforms = self.tfms_from_dict(keys_tr)

    def tfms_from_dict(self, keys: str):
        keys = keys.replace(" ", "").split(",")
        tfms = [self.transforms_dict[key] for key in keys]
        ds_specific_tfms = [t for t in tfms if isinstance(t, dict)]
        if not ds_specific_tfms:
            return Compose(tfms)

        ds_keys = set()
        for t in ds_specific_tfms:
            ds_keys.update(t.keys())

        composed_per_ds = {}
        for ds in ds_keys:
            ds_chain = []
            for t in tfms:
                if isinstance(t, dict):
                    if ds not in t:
                        raise KeyError(
                            f"Missing transform for ds='{ds}' in datasource-specific transform."
                        )
                    ds_chain.append(t[ds])
                else:
                    ds_chain.append(t)
            composed_per_ds[ds] = Compose(ds_chain)
        return composed_per_ds

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
        self.Indx = FgBgToIndicesSubsampled(
            keys=[self.lm_key],
            image_key=self.image_key,
            ignore_labels=self.plan.get("fg_indices_exclude", []),
            image_threshold=-2600,
        )
        self.Labels = GetLabelsd(lm_key=self.lm_key)
        self.LoadT = LoadTorchd(keys=[self.image_key, self.lm_key])
        self.Remap = self.create_monai_remapping_per_ds(self.remapping_key, self.lm_key)

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
        assert self.remapping_key is not None, (
            "remapping_key must be provided,\n This class is not meant to be used without sublcassing and setting a remapping_key attr"
        )
        dss = self.plan["datasources"]
        dss = parse_excel_datasources(dss)

        rem = self.plan[remapping_key]
        rem = parse_excel_remapping(rem, dss)
        Remapping_tfms = {}
        for rez, ds in zip(rem, dss):
            ds_name = DS[ds].name

            if rez is not None:
                orig = list(rez.keys())
                target = list(rez.values())
                R = MapLabelValueD(
                    keys=[self.lm_key], orig_labels=orig, target_labels=target
                )
            else:
                R = DummyTransform(keys=[lm_key])
            Remapping_tfms[ds_name] = R
        return Remapping_tfms

    def process(self, mini_df):
        outs = []
        for i, row in mini_df.iterrows():
            try:
                outs.append(self._process_row(row))

            except Exception as e:
                trace = traceback.format_exc()
                print(
                    f"[{self.__class__.__name__}] error:"
                    f"\n  case_id={row.get('case_id')}"
                    f"\n  image={row.get('image')}"
                    f"\n  lm={row.get('lm')}"
                    f"\n  lm_imported={row.get('lm_imported')}"
                )
                print(trace.rstrip())
                fn_name = strip_extension(Path(str(row.get("image"))).name) + ".pt"
                outs.append(
                    {
                        "case_id": row.get("case_id"),
                        "fn_name": fn_name,
                        "ok": False,
                        "err": repr(e),
                        "labels": None,
                        "_preprocess_events": [],
                        "_preprocess_error": {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "traceback": trace,
                        },
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
