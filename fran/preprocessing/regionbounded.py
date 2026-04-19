from __future__ import annotations

import copy
import logging
import os
from pathlib import Path

import numpy as np
import ray
import torch
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.preprocessing.rayworker_base import RayWorkerBase
import localiser.inference.base as localiser_base
from localiser.inference.base import (
    LocaliserInfererPT,
    bbox_from_file,
    crop_to_yolo_bbox,
    load_yolo_specs,
)
from monai.transforms.transform import MapTransform
from ultralytics import YOLO
from utilz.fileio import load_yaml
from utilz.stringz import ast_literal_eval, info_from_filename

# localiser.inference.base.bbox_from_file expects this module global but does
# not define it in current localiser builds.


class CropByYolo(MapTransform):
    def __init__(
        self,
        keys=("image", "lm"),
        lm_key="lm",
        bbox_key="bbox",
        margin=20,
        sanitize=True,
        allow_missing_keys=False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.lm_key = lm_key
        self.bbox_key = bbox_key
        self.margin = float(margin)
        self.sanitize = bool(sanitize)
        self.logger = logging.getLogger(__name__)

    def __call__(self, data):
        dici = dict(data)
        self._assert_3d(dici)
        first_crop = self._crop_keys(dici, dici[self.bbox_key])
        if not self.sanitize:
            return first_crop

        fg_before = self._fg_count(dici)
        fg_after = self._fg_count(first_crop)
        if fg_before == fg_after:
            return first_crop

        self._log_sanitize_mismatch(
            stage="first",
            data=dici,
            fg_before=fg_before,
            fg_after=fg_after,
        )

        expanded_crop = self._try_expanded_crop(
            data=dici,
            bbox=dici[self.bbox_key],
            first_crop=first_crop,
        )

        fg_after_expanded = self._fg_count(expanded_crop)
        if fg_after_expanded != fg_before:
            self._log_sanitize_mismatch(
                stage="expanded",
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
        affine = lm.meta["affine"]
        affine = torch.as_tensor(affine).detach().cpu().numpy()
        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        spacing = np.where(spacing > 0, spacing, 1.0)
        return spacing

    @staticmethod
    def _spatial_shape(tensor):
        return tuple(int(v) for v in tensor.shape)

    def _log_context(self, data: dict):
        case_id = data.get("case_id")
        bbox_fn = data.get("bbox_fn")
        return case_id, bbox_fn

    def _log_sanitize_mismatch(
        self,
        *,
        stage: str,
        data: dict,
        fg_before: int,
        fg_after: int,
    ) -> None:
        case_id,  bbox_fn = self._log_context(data)
        if stage == "first":
            self.logger.warning(
                "CropByYolo fg mismatch (first crop): case_id=%s bbox=%s fg_before=%d fg_after=%d; retrying with %.1fmm margin",
                case_id,
                bbox_fn,
                fg_before,
                fg_after,
                self.margin,
            )
            return

        if stage == "expanded":
            self.logger.error(
                "CropByYolo fg mismatch persists after expansion: case_id=%sbbox=%s fg_before=%d fg_after_expanded=%d",
                case_id,
                bbox_fn,
                fg_before,
                fg_after,
            )


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
        self.CropByYolo = CropByYolo(
            keys=["image", "lm"],
            lm_key="lm",
            bbox_key="bbox",
            margin=20,
            sanitize=True,
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
        fg_indices_exclude = self.plan.get("fg_indices_exclude")
        if fg_indices_exclude is None:
            fg_indices_exclude = []
        elif isinstance(fg_indices_exclude, int):
            fg_indices_exclude = [fg_indices_exclude]
        if len(fg_indices_exclude) > 0:
            indices_subfolder = "indices_fg_exclude_{}".format(
                "".join([str(x) for x in fg_indices_exclude])
            )
        else:
            indices_subfolder = "indices"
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

    def create_data_df(self):
        super().create_data_df()
        bbox_files = sorted(self.loc_folder.glob("*.txt"))
        if len(bbox_files) == 0:
            raise FileNotFoundError(f"No bbox txt files found in {self.loc_folder}")

        by_case = self._index_bbox_files_by_case_id(bbox_files)
        resolved_bbox_fns = []
        missing = []
        duplicates = {}
        for case_id in self.df["case_id"].tolist():
            case_key = str(case_id).lower()
            matches = by_case.get(case_key, [])
            if len(matches) == 0:
                missing.append(case_key)
            elif len(matches) > 1:
                duplicates[case_key] = [str(fn) for fn in matches]
            else:
                resolved_bbox_fns.append(matches[0].resolve())

        if missing or duplicates:
            msg = [f"Failed deterministic bbox matching in {self.loc_folder}."]
            if missing:
                msg.append(f"Missing bbox files for case_ids ({len(missing)}): {missing}")
            if duplicates:
                msg.append("Duplicate bbox matches found:")
                for case_id, fns in sorted(duplicates.items()):
                    msg.append(f"  - {case_id}: {fns}")
            raise ValueError("\n".join(msg))

        self.df = self.df.assign(bbox_fn=resolved_bbox_fns)

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
        for full_caseid in (False, True):
            try:
                info = info_from_filename(fn.name, full_caseid=full_caseid)
            except Exception:
                continue
            case_id = info.get("case_id")
            if case_id:
                keys.add(str(case_id).lower())
        return {k for k in keys if k}

    def process(self, derive_bboxes=False):
        self.infer_yolo_bboxes()
        return super().process(derive_bboxes=derive_bboxes)

    def infer_yolo_bboxes(self):
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
        imgs = self.df.image.tolist()
        self.I.run(imgs, overwrite=False)

    def get_region_indices(self):
        regions = self.plan["localiser_regions"]
        regions = str(regions).replace(" ", "")
        regions_list = [r for r in regions.split(",") if r]

        names = self.yolo_specs["data"]["names"]
        if isinstance(names, dict):
            class_to_ind = {str(v): int(k) for k, v in names.items()}
        else:
            class_to_ind = {str(name): idx for idx, name in enumerate(names)}

        classes_in_bbox = []
        for class_name, class_idx in class_to_ind.items():
            if any(region in class_name for region in regions_list):
                classes_in_bbox.append(class_idx)
        classes_in_bbox = sorted(set(classes_in_bbox))

        if len(classes_in_bbox) == 0:
            raise ValueError(
                f"No localiser classes matched localiser_regions='{self.plan['localiser_regions']}'."
            )
        return classes_in_bbox

    @property
    def loc_folder(self):
        return Path(self.data_folder) / "localisers"

    @property
    def loc_fldr(self):
        return self.loc_folder


# %%
# SECTION:-------------------- setup--------------------------------------------------------------------------------------
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
    conf_fldr = os.environ["FRAN_CONF"]
    regions = plan["localiser_regions"]
    best_runs = load_yaml(Path(conf_fldr) / "best_runs.yaml")

    yolo_fldr = best_runs["localiser"][0]
    yolo_specs = load_yolo_specs(yolo_fldr)
    ckpt = yolo_specs["ckpt"]
    data = yolo_specs["data"]
    specs = yolo_specs["specs"]
    class_names = data["names"]
    class_to_ind = {n: x for x, n in enumerate(class_names)}
# %%
    regions = regions.replace(" ", "")
    regions_list = regions.split(",")
    classes_in_bbox = []
    for k in class_to_ind.keys():
        for region in regions_list:
            if region in k:
                classes_in_bbox.append(class_to_ind[k])

# %%

    overwrite=True
    num_processes = 16
    R = RegionBoundedDataGenerator(project=P, plan=plan, data_folder=existing_fldr)
    R.setup(num_processes=num_processes, overwrite=overwrite)
    R.process()
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
    C.sanitize = True

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
