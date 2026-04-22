from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import ray
import torch
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.preprocessing.rayworker_base import RayWorkerBase
from fran.transforms.spatialtransforms import CropForegroundMinShaped
from fran.utils.affine import spacing_from_affine
from localiser.inference.base import (
    LocaliserInfererPT,
    load_yolo_specs,
)
from localiser.utils.bbox_helpers import bbox_from_file, crop_to_yolo_bbox
from monai.transforms.transform import MapTransform
from ultralytics import YOLO
from utilz.fileio import load_yaml
from utilz.stringz import info_from_filename

class CropByYolo(MapTransform):
    def __init__(
        self,
        keys=("image", "lm"),
        lm_key="lm",
        bbox_key="bbox",
        margin=20,
        allow_missing_keys=False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.lm_key = lm_key
        self.bbox_key = bbox_key
        self.margin = float(margin)

    def __call__(self, data):
        dici = dict(data)
        self._assert_3d(dici)
        first_crop = self._crop_keys(dici, dici[self.bbox_key])
        fg_before = self._fg_count(dici)
        expanded_crop = self._try_expanded_crop(
            data=dici,
            bbox=dici[self.bbox_key],
            first_crop=first_crop,
        )
        fg_after_expanded = self._fg_count(expanded_crop)
        self._log_mismatch(
            data=dici,
            fg_before=fg_before,
            fg_after=fg_after_expanded,
        )
        return expanded_crop

    def _crop_keys(self, data: dict, bbox: dict) -> dict:
        out = dict(data)
        for key in self.key_iterator(out):
            out[key] = crop_to_yolo_bbox(out[key], bbox)
        return out

    def _fg_count(self, data: dict) -> int:
        return int(torch.count_nonzero(data[self.lm_key]).item())

    def _try_expanded_crop(
        self,
        *,
        data: dict,
        bbox: dict,
        first_crop: dict,
    ) -> dict:
        expanded_bbox = self._expand_bbox_mm(
            bbox=bbox,
            lm=data[self.lm_key],
            margin_mm=self.margin,
        )
        expanded_crop = self._crop_keys(data, expanded_bbox)
        expanded_crop[self.bbox_key] = expanded_bbox
        return expanded_crop

    def _assert_3d(self, data: dict):
        for key in self.key_iterator(data):
            assert data[key].ndim == 3, (
                f"CropByYolo expects 3D tensors, got {key}.ndim={data[key].ndim}"
            )

    def _expand_bbox_mm(self, bbox: dict, lm, margin_mm: float) -> dict:
        bbox_out = copy.deepcopy(bbox)
        spacing = self._spacing_from_affine(lm)
        spatial_shape = self._spatial_shape(lm)
        axis_map = {"ap": 0, "width": 1, "height": 2}

        for key, axis in axis_map.items():
            start, end = bbox_out[key]
            start = float(start)
            end = float(end)
            vox_margin = int(np.ceil(margin_mm / spacing[axis]))
            delta = vox_margin / max(int(spatial_shape[axis]), 1)
            bbox_out[key] = (max(0.0, start - delta), min(1.0, end + delta))
        return bbox_out

    def _spacing_from_affine(self, lm) -> np.ndarray:
        return spacing_from_affine(lm.meta["affine"])

    @staticmethod
    def _spatial_shape(tensor):
        return tuple(int(v) for v in tensor.shape)

    def _log_context(self, data: dict):
        case_id = data.get("case_id")
        bbox_fn = data.get("bbox_fn")
        return case_id, bbox_fn

    def _log_mismatch(
        self,
        *,
        data: dict,
        fg_before: int,
        fg_after: int,
    ) -> None:
        case_id, bbox_fn = self._log_context(data)
        self._register_warning(
            data,
            "CropByYolo fg mismatch: "
            f"case_id={case_id} bbox_txt_path={bbox_fn} fg_before={fg_before} "
            f"fg_after={fg_after}",
        )

    @staticmethod
    def _register_warning(data: dict, message: str) -> None:
        events = data["_preprocess_events"]
        events.append({"error_type": "CropByYolo", "error_message": str(message)})
        data["_preprocess_events"] = events


class CropByYoloWithForegroundFallbackd(MapTransform):
    """
    Run YOLO crop first, then fallback to foreground crop if FG voxels are still lost.
    """

    def __init__(
        self,
        min_shape,
        keys=("image", "lm"),
        lm_key="lm",
        bbox_key="bbox",
        allow_missing_keys=False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.lm_key = lm_key
        self.bbox_key = bbox_key
        self.cropper_yolo = CropByYolo(
            keys=keys,
            lm_key=lm_key,
            bbox_key=bbox_key,
            margin=20,
            allow_missing_keys=allow_missing_keys,
        )
        self.cropper_fg = CropForegroundMinShaped(
            keys=keys,
            source_key=lm_key,
            allow_missing_keys=allow_missing_keys,
            min_shape=min_shape,
        )

    def __call__(self, data):
        original = dict(data)
        lm = data[self.lm_key]
        assert lm.ndim == 3, "CropByYoloWithForegroundFallbackd expects 3D tensors"
        fg_before = self._fg_count(original)
        yolo_out = self.cropper_yolo(dict(original))
        fallback_input = dict(original)
        fallback_input["_preprocess_events"] = list(yolo_out["_preprocess_events"])

        fallback_out = self.apply_crop_fg(fallback_input)
        fg_after_fallback = self._fg_count(fallback_out)
        fg_after_yolo = self._fg_count(yolo_out)

        case_id = original.get("case_id")
        bbox_fn = original.get("bbox_fn")
        message = (
            "CropByYolo fallback to CropForegroundMinShaped: "
            f"case_id={case_id} bbox={bbox_fn} "
            f"fg_before={fg_before} fg_after_yolo={fg_after_yolo} "
            f"fg_after_fallback={fg_after_fallback} "
            "verified_fg_preserved=False"
        )
        self._register_event(fallback_out, message)
        return fallback_out

    def apply_crop_fg(self, fallback_input):
        fallback_input = self._temporarily_channel_first(fallback_input)
        fallback_out = self.cropper_fg(fallback_input)
        fallback_out = self._restore_spatial_only(fallback_out)
        return fallback_out


    def _fg_count(self, data: dict) -> int:
        return int(torch.count_nonzero(data[self.lm_key]).item())

    def _temporarily_channel_first(self, data: dict) -> dict:
        d = dict(data)
        added_channel_keys = []
        for key in self.key_iterator(d):
            x = d[key]
            d[key] = x.unsqueeze(0)
            added_channel_keys.append(key)
        d["_fg_fallback_added_channel_keys"] = added_channel_keys
        return d

    def _restore_spatial_only(self, data: dict) -> dict:
        d = dict(data)
        added_channel_keys = d.pop("_fg_fallback_added_channel_keys")
        for key in added_channel_keys:
            d[key] = d[key].squeeze(0)
        return d

    @staticmethod
    def _register_event(data: dict, message: str) -> None:
        events = data["_preprocess_events"]
        events.append(
            {"error_type": "CropByYoloFallback", "error_message": str(message)}
        )
        data["_preprocess_events"] = events

class _RBDSamplerWorkerBase(RayWorkerBase):
    remapping_key = "remapping_lbd_kbd"

    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        crop_to_label=None,
        device="cpu",
        debug=False,
        tfms_keys="LoadT,Dev,CropByYolo,Chan,Remap,Labels,Indx",
    ):
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            crop_to_label=crop_to_label,
            device=device,
            debug=debug,
            tfms_keys=tfms_keys,
        )

    def create_transforms(self, device):
        super().create_transforms(device=device)
        self.cropper_yolo = CropByYolo(
            keys=["image", "lm"],
            lm_key="lm",
            bbox_key="bbox",
            margin=20,
        )
        self.CropByYolo = CropByYoloWithForegroundFallbackd(
            min_shape=self.plan["src_dims"],
            keys=["image", "lm"],
            lm_key="lm",
            bbox_key="bbox",
        )
        self.transforms_dict["CropByYolo"] = self.CropByYolo

    def _create_data_dict(self, row):
        bbox_fn = row["bbox_fn"]
        data = {
            "case_id": row["case_id"],
            "image": row["image"],
            "lm": row["lm"],
            "ds": row["ds"],
            "remapping": row["remapping"],
            "bbox_fn": bbox_fn,
            "bbox": bbox_from_file(bbox_fn),
        }
        return data

    @property
    def indices_subfolder(self):
        fg_indices_exclude = self.plan["fg_indices_exclude"]
        indices_subfolder = "indices_fg_exclude_{}".format(
            "".join([str(x) for x in fg_indices_exclude])
        )
        return self.output_folder / indices_subfolder


@ray.remote(num_cpus=1)
class RBDSamplerWorkerImpl(_RBDSamplerWorkerBase):
    pass


class RBDSamplerWorkerLocal(_RBDSamplerWorkerBase):
    pass


class RegionBoundedDataGenerator(LabelBoundedDataGenerator):
    actor_cls = RBDSamplerWorkerImpl
    local_worker_cls = RBDSamplerWorkerLocal
    remapping_key = "remapping_lbd_kbd"
    subfolder_key = "data_folder_kbd"

    def _index_bbox_files_by_case_id(self, bbox_files: list[Path]) -> dict[str, list[Path]]:
        mapping: dict[str, list[Path]] = {}
        for fn in bbox_files:
            case_keys = self._case_id_keys_from_bbox_file(fn)
            for case_id in case_keys:
                mapping.setdefault(case_id, []).append(fn)
        for case_id, fns in mapping.items():
            mapping[case_id] = sorted({Path(fn) for fn in fns}, key=lambda p: str(p))
        return mapping

    @staticmethod
    def _case_id_keys_from_bbox_file(fn: Path) -> set[str]:
        keys = {fn.stem.lower()}
        info = info_from_filename(fn.name, full_caseid=True)
        keys.add(str(info["case_id"]).lower())
        return keys

    def attach_bbox_fns(self):
      bbox_files = sorted(self.loc_folder.glob("*.txt"))
      by_case = self._index_bbox_files_by_case_id(bbox_files)
      resolved = {case_id: fns[0].resolve() for case_id, fns in by_case.items()}
      self.df["bbox_fn"] = self.df["case_id"].astype(str).str.lower().map(resolved)

    def missing_bbox_mask(self):
      return self.df["bbox_fn"].isna()

    def infer_yolo_bboxes(self):
      missing = self.missing_bbox_mask()
      imgs = self.df.loc[missing, "image"].tolist()

      conf_fldr = os.environ["FRAN_CONF"]
      best_runs = load_yaml(Path(conf_fldr) / "best_runs.yaml")
      yolo_fldr = best_runs["localiser"][0]
      self.yolo_specs = load_yolo_specs(yolo_fldr)

      ckpt = self.yolo_specs["ckpt"]
      model = YOLO(ckpt)
      classes_in_bbox = self.get_region_indices()
      imsize = self.yolo_specs["specs"]["imgsz"]

      self.I = LocaliserInfererPT(
          model,
          classes_in_bbox,
          imsize=imsize,
          window="a",
          projection_dim=(1, 2),
          out_folder=self.loc_folder,
          batch_size=64,
      )
      self.I.run(imgs, overwrite=False)

    def process(self):
      self.maybe_infer_bboxes()
      self.mini_dfs = self.split_dataframe_for_workers(self.df, self.num_processes)

      return super().process()


    def create_data_df(self):
      super().create_data_df()
      self.attach_bbox_fns()

    def maybe_infer_bboxes(self):
      self.infer_yolo_bboxes()
      self.attach_bbox_fns()


    def get_region_indices(self):
        regions = self.plan["localiser_regions"]
        regions = str(regions).replace(" ", "")
        regions_list = [r for r in regions.split(",") if r]

        names = self.yolo_specs["data"]["names"]
        class_to_ind = {str(v): int(k) for k, v in names.items()}

        classes_in_bbox = []
        for class_name, class_idx in class_to_ind.items():
            if any(region in class_name for region in regions_list):
                classes_in_bbox.append(class_idx)
        classes_in_bbox = sorted(set(classes_in_bbox))
        return classes_in_bbox

    @property
    def loc_folder(self):
        return Path(self.output_folder) / "localisers"



# %%
# SECTION:-------------------- setup-------------------------------------------------------------------------------------- if __name__ == "__main__":
if __name__ == "__main__":
    from pathlib import Path

    from fran.configs.parser import ConfigMaker
    from fran.managers import Project
    from fran.utils.common import *
    from fran.utils.folder_names import FolderNames
    from monai.transforms.io.dictionary import LoadImaged
    from fran.transforms.imageio import TorchReader
    from utilz.helpers import pp

    project_title = "kits23"
    P = Project(project_title=project_title)
    # P.maybe_store_projectwide_properties()
    # spacing = [1.5, 1.5, 1.5]

    C = ConfigMaker(P)
    C.setup(2)
    C.plans
    conf = C.configs
    print(conf["model_params"])

    plan = conf["plan_train"]
    pp(plan)
    spacing = plan["spacing"]
    # plan["remapping_imported"][0]
    existing_fldr = FolderNames(P, plan).folders.get("data_folder_source", None)
    pp(existing_fldr)
# %%

    overwrite=True
    num_processes = 16
    R = RegionBoundedDataGenerator(project=P, plan=plan, data_folder=existing_fldr)
    R.setup(num_processes=num_processes, overwrite=overwrite)
    # R.process()
# %%
    RR = RBDSamplerWorkerLocal(
        project=R.project,
        plan=R.plan,
        data_folder=R.data_folder,
        output_folder=R.output_folder,
    )

    row = R.df.iloc[0]
    dici = {
        "case_id": row["case_id"],
        "image": row["image"],
        "lm": row["lm"],
        "ds": row["ds"],
        "remapping": row["remapping"],
        "bbox_fn": row["bbox_fn"],
    }
    dici["bbox"] = bbox_from_file(dici["bbox_fn"])

# %%
    dici = RR.transforms_dict["LoadT"](dici)
    dici = RR.transforms_dict["Dev"](dici)
    dici = RR.transforms_dict["CropByYolo"](dici)
    dici = RR.transforms_dict["Chan"](dici)
    dici = RR.transforms_dict["Remap"][dici["ds"]](dici)
    dici = RR.transforms_dict["Labels"](dici)
    dici = RR.transforms_dict["Indx"](dici)

# %%
    image = dici["image"]
    lm = dici["lm"]
    lm_fg_indices = dici["lm_fg_indices"]
    lm_bg_indices = dici["lm_bg_indices"]
    labels = dici["lm_labels"]

    assert image.shape == lm.shape, "mismatch in shape"
    assert image.dim() == 4, "images should be cxhxwxd"

    inds = {
        "lm_fg_indices": lm_fg_indices,
        "lm_bg_indices": lm_bg_indices,
        "meta": image.meta,
    }

    # Optional local debug writes, same as worker side-effects.
    # RR.save_indices(inds, RR.indices_subfolder)
    # RR.save_pt(image[0], "images")
    # RR.save_pt(lm[0], "lms")

# %%
    row = R.df.iloc[0]
    dici = {"case_id": row["case_id"], "image": row["image"], "lm": row["lm"]}
    dici = Lp(dici)
    R.loc_fldr
    txt_files = R.loc_fldr.glob("*.txt")
    case_id = dici["case_id"]
# %%
    for txt_fl in txt_files:
        txt_fl_case_id = info_from_filename(txt_fl.name)["case_id"][0]
        if case_id == txt_fl_case_id:
            continue
    dici["bbox_fn"] = txt_fl

    bbox = bbox_from_file(dici["bbox_fn"])
    dici["bbox"] = bbox

# %%
    C = CropByYolo()
    C.margin = 50

    txt_fn = dici["bbox_fn"]

    text = Path(txt_fn).read_text()
    toks = text.split("\n")
    dici = {}

    dici = Lp(dici)


    bbox["width"] = (0.0898,0.5)

    lm = dici["lm"]
    lm_org = lm.clone()
    counts_org = lm.count_nonzero().item()

    img = crop_to_yolo_bbox(image, bbox)
    lm = crop_to_yolo_bbox(lm, bbox)


    counts_after = lm.count_nonzero()
    counts_after == counts_org


    bbo2 = C._expand_bbox_mm(bbox, lm, C.margin)

    bbox_out = copy.deepcopy(bbox)
    spacing = C._spacing_from_affine(lm)
    spatial_shape = C._spatial_shape(lm)
    axis_map = {"ap": 0, "width": 1, "height": 2}

# %%
    margin_mm=50
    for key, axis in axis_map.items():
        start, end = bbox_out[key]
        start = float(start)
        end = float(end)
        vox_margin = int(np.ceil(margin_mm / spacing[axis]))
        delta = vox_margin / max(int(spatial_shape[axis]), 1)
        bbox_out[key] = (max(0.0, start - delta), min(1.0, end + delta))
# %%


    ImageMaskViewer([img, lm])

    txt_file = [
        x for x in txt_files if case_id == info_from_filename(x.name)["case_id"][0]
    ][0]
    txt_file

# %%
    image = dici["image"]
    mask = dici["lm"]
    image.meta
    aff = image.meta["affine"]

# %%
    imsize = specs["imgsz"]

    I = LocaliserInfererPT(
        model,
        classes_in_bbox,
        imsize=imsize,
        window="a",
        projection_dim=(1, 2),
        out_folder=R.loc_fldr,
        batch_size=64,
    )
# %%
    imgs = R.df.image.tolist()
    len(imgs)
# %%
    preds = I.run(imgs)

    row = R.df.iloc[0]
# %%
    overwrite = False
    num_processes = 5
    debug_ = False
    R.setup(overwrite=overwrite, device="cpu", num_processes=num_processes, debug=debug_)
    R.process()
# %%
    R.mini_dfs = R.split_dataframe_for_workers(R.df, num_processes)
    mini_df = R.mini_dfs[0].iloc[:3]
# %%
    overwrite = False
    RR = RBDSamplerWorkerLocal(
        project=R.project,
        plan=R.plan,
        data_folder=R.data_folder,
        output_folder=R.output_folder,
    )
    RR.process(mini_df)
# %%


    case_id = "kits23_00486"

    row = R.df[R.df["case_id"].astype(str) == case_id].iloc[0]

    dici = {
        "case_id": row["case_id"],
        "image": row["image"],
        "lm": row["lm"],
        "ds": row["ds"],
        "remapping": row["remapping"],
        "bbox_fn": row["bbox_fn"],
    }
    dici["bbox"] = bbox_from_file(dici["bbox_fn"])

# %%
    RR = RBDSamplerWorkerLocal(
        project=R.project,
        plan=R.plan,
        data_folder=R.data_folder,
        output_folder=R.output_folder,
    )

    dici = RR.transforms_dict["LoadT"](dici)
    dici = RR.transforms_dict["Dev"](dici)

    C = CropByYolo(
        keys=["image", "lm"],
        lm_key="lm",
        bbox_key="bbox",
        margin=20,
        sanitize=True,
    )

    dici2 = C(dici)
# %%
    img = dici['image']
    lm = dici['lm']
    ImageMaskViewer([img, lm])

# %%

    dici["image"].shape, dici["lm"].shape
    dici2["image"].shape, dici2["lm"].shape

    dici["lm"].count_nonzero().item(), dici2["lm"].count_nonzero().item()

    dici2.get("_preprocess_events")

    # Fallback wrapper too:

    CF = CropByYoloWithForegroundFallbackd(
        min_shape=R.plan["src_dims"],
        keys=["image", "lm"],
        lm_key="lm",
        bbox_key="bbox",
    )

    dici3 = CF(dici)

    dici3["image"].shape, dici3["lm"].shape

    dici3["lm"].count_nonzero().item()

    dici3.get("_preprocess_events")
# %%
