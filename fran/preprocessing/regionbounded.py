from __future__ import annotations

import json
import math
from pathlib import Path

from fran.preprocessing.helpers import import_h5py
import ipdb
from localiser.inference.base import bbox_from_file
import numpy as np
import ray
import torch
from fran.inference.cascade_yolo import LocaliserInfererPT
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.preprocessing.preprocessor import DEFAULT_HDF5_SRC_DIMS
from fran.preprocessing.rayworker_base import RayWorkerBase
from fran.transforms.spatialtransforms import (
    CropByYoloWithForegroundFallbackd,
    CropForegroundMinShaped,
    CropMaybePad,
)
from fran.utils.affine import spacing_from_affine
from localiser.utils.bbox_helpers import (
    EmptyBBoxDetectionsError,
    MissingBBoxClassMatchError,
    standardize_bboxes,
)
from monai.transforms.transform import MapTransform
from utilz.cprint import cprint
from utilz.stringz import info_from_filename


class CropByYolo(CropMaybePad):
    def __init__(
        self,
        keys=("image", "lm"),
        lm_key="lm",
        bbox_key="bbox",
        min_shape=(0, 0, 0),
        margin=20,
        allow_missing_keys=False,
    ):
        super().__init__(keys=keys, min_shape=min_shape, margin=margin)
        self.lm_key = lm_key
        self.bbox_key = bbox_key

    def __call__(self, data):
        dici = dict(data)
        fg_before = self._fg_count(dici)
        spacing = spacing_from_affine(dici[self.lm_key].meta["affine"])
        box_start, box_end = self._yolo_bbox_to_bounds(
            tuple(int(v) for v in dici[self.lm_key].shape[1:]),
            dici[self.bbox_key],
        )
        out = super().__call__(dici, box_start, box_end, spacing)
        out[self.bbox_key] = dici[self.bbox_key]
        fg_after_expanded = self._fg_count(out)
        self._log_mismatch(
            data=dici,
            fg_before=fg_before,
            fg_after=fg_after_expanded,
        )
        return out

    def _fg_count(self, data: dict) -> int:
        return int(torch.count_nonzero(data[self.lm_key]).item())

    @staticmethod
    def _yolo_bbox_to_bounds(img_shape, bbox_dici):
        width3d, ap3d, height3d = img_shape
        wd = bbox_dici["width"]
        height = bbox_dici["height"]
        ap = bbox_dici["ap"]
        return (
            math.floor(wd[0] * width3d),
            math.floor(ap[0] * ap3d),
            math.floor((1.0 - height[1]) * height3d),
        ), (
            math.ceil(wd[1] * width3d),
            math.ceil(ap[1] * ap3d),
            math.ceil((1.0 - height[0]) * height3d),
        )

    def _log_mismatch(
        self,
        *,
        data: dict,
        fg_before: int,
        fg_after: int,
    ) -> None:
        self._register_warning(
            data,
            "CropByYolo fg mismatch: "
            f"case_id={data.get('case_id')} bbox_source_path={data.get('bbox_fn')} fg_before={fg_before} "
            f"fg_after={fg_after}",
        )

    @staticmethod
    def _register_warning(data: dict, message: str) -> None:
        events = data["_preprocess_events"]
        events.append({"error_type": "CropByYolo", "error_message": str(message)})
        data["_preprocess_events"] = events


class _RBDSamplerWorkerBase(RayWorkerBase):
    remapping_key = "remapping_lbd_rbd"

    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        crop_to_label=None,
        device="cpu",
        debug=False,
        tfms_keys="LoadT,Chan,Dev,CropByYolo,Remap,Labels,Indx",
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
        margin = self.plan["expand_by"]
        self.CropByYolo = CropByYoloWithForegroundFallbackd(
            min_shape=self.plan["src_dims"],
            keys=["image", "lm"],
            lm_key="lm",
            bbox_key="bbox",
            margin=margin,
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
            "bbox": row["bbox"],
            "_preprocess_events": [],
        }
        return data

    @property
    def indices_subfolder(self):
        fg_indices_exclude = self.plan.get("fg_indices_exclude")
        if fg_indices_exclude is None:
            indices_subfolder = "indices"
        elif isinstance(fg_indices_exclude, int):
            indices_subfolder = f"indices_fg_exclude_{fg_indices_exclude}"
        else:
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
    remapping_key = "remapping_lbd_rbd"
    subfolder_key = "data_folder_rbd"

    def _index_bbox_files_by_case_id(
        self, bbox_files: list[Path]
    ) -> dict[str, list[Path]]:
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

    def _ensure_bbox_columns(self):
        if "bbox" not in self.df.columns:
            self.df["bbox"] = None
        if "bbox_fn" not in self.df.columns:
            self.df["bbox_fn"] = None

    def _standardize_cached_bbox_json(
        self, json_fn: Path, classes_in_bbox: list[int]
    ) -> dict:
        bbox = json.loads(json_fn.read_text())
        pads3tup = bbox["ap"]["meta"]["letterbox_padded"]
        try:
            return standardize_bboxes(
                bbox["ap"],
                bbox["lat"],
                pads3tup,
                classes_in_bbox,
                serialised=True,
            )
        except EmptyBBoxDetectionsError:
            return {"empty_bbox": True}
        except MissingBBoxClassMatchError as exc:
            raise MissingBBoxClassMatchError(
                "Failed to standardize cached bbox JSON for RBD preprocessing. "
                f"case_id={json_fn.stem} bbox_json={json_fn} "
                f"{exc}"
            ) from exc

    def attach_bboxes(self, classes_in_bbox: list[int]) -> None:
        self._ensure_bbox_columns()
        bbox_files = sorted(self.I.output_folder.glob("*.json"))
        by_case = self._index_bbox_files_by_case_id(bbox_files)
        case_ids = self.df["case_id"].astype(str)
        case_id_set = set(case_ids.tolist())
        by_case_found = {
            case_id: fns for case_id, fns in by_case.items() if case_id in case_id_set
        }

        resolved_fns = {}
        resolved_bboxes = {}
        for case_id, fns in by_case_found.items():
            if len(fns) != 1:
                raise ValueError(f"Duplicate bbox matches for case_id={case_id}: {fns}")
            json_fn = fns[0].resolve()
            resolved_fns[case_id] = json_fn
            resolved_bboxes[case_id] = self._standardize_cached_bbox_json(
                json_fn, classes_in_bbox
            )
        self.df["bbox_fn"] = case_ids.map(resolved_fns)
        self.df["bbox"] = case_ids.map(resolved_bboxes)

    def missing_bbox_mask(self):
        self._ensure_bbox_columns()
        return self.df["bbox"].isna()

    def _localiser_regions_list(self) -> list[str]:
        regions = self.plan.get("localiser_regions")
        if regions is None:
            return ["all"]
        elif isinstance(regions, (list, tuple, set)):
            return [str(region).strip() for region in regions if str(region).strip()]
        else:
            return [r for r in str(regions).replace(" ", "").split(",") if r]

    def maybe_infer_bboxes(self):
        regions = self._localiser_regions_list()
        self.I = LocaliserInfererPT(
            localiser_regions=regions,
            window="a",
            bs=16,
            devices=self.devices,
            debug=False,
        )
        self.yolo_specs = self.I.yolo_state_dict
        classes_in_bbox = self.get_region_indices()
        self.attach_bboxes(classes_in_bbox)

        missing = self.missing_bbox_mask()
        imgs = self.df.loc[missing, "image"].tolist()
        if len(imgs) == 0:
            return
        cprint(
            f"Total case {len(self.df)}. \nBBoxes on file: {len(self.df) - len(imgs)}. \nRemaining bboxes: {len(imgs)}. \nInferring missing bboxes with localiser for regions: {regions}",
            color="blue",
        )
        self.I.run(imgs, overwrite=False)
        self.attach_bboxes(classes_in_bbox)

    def process(
        self,
        overwrite=None,
        derive_bboxes=True,
        src_dims=DEFAULT_HDF5_SRC_DIMS,
        cases_per_shard=5,
        max_shard_bytes=None,
        overwrite_hdf5_shards=False,
        hdf5_compression="gzip",
        hdf5_compression_opts=1,
    ):
        self.maybe_infer_bboxes()
        self.mini_dfs = self.split_dataframe_for_workers(self.df, self.num_processes)
        return super().process(
            overwrite=overwrite,
            derive_bboxes=derive_bboxes,
            src_dims=src_dims,
            cases_per_shard=cases_per_shard,
            max_shard_bytes=max_shard_bytes,
            overwrite_hdf5_shards=overwrite_hdf5_shards,
            hdf5_compression=hdf5_compression,
            hdf5_compression_opts=hdf5_compression_opts,
        )

    def create_data_df(self):
        super().create_data_df()
        self._ensure_bbox_columns()

    def get_region_indices(self):
        regions = self.plan["localiser_regions"]
        names = self.yolo_specs["data"]["names"]
        if isinstance(names, dict):
            class_to_ind = {str(v): int(k) for k, v in names.items()}
        else:
            class_to_ind = {str(name): idx for idx, name in enumerate(names)}
        if regions is None:
            return sorted(class_to_ind.values())
        regions = str(regions).replace(" ", "")
        if regions.lower() == "all":
            return sorted(class_to_ind.values())
        regions_list = [r for r in regions.split(",") if r]

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
    from fran.configs.parser import ConfigMaker
    from fran.managers import Project
    from fran.utils.common import *
    from fran.utils.folder_names import FolderNames
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
    devices = [0]

    overwrite = False
    num_processes = 8
    R = RegionBoundedDataGenerator(project=P, plan=plan, data_folder=existing_fldr,devices=devices)
    R.setup(num_processes=num_processes, overwrite=overwrite)
# %%

        manifest_fn = R.hdf5_manifest_fn
        R.existing_output_fnames = set()
# %%
        if manifest_fn.exists():
            manifest = json.loads(manifest_fn.read_text())
            h5py = import_h5py()
            shards_folder = manifest_fn.parent
            for shard_meta in manifest["shards"]:
                shard_fn = shards_folder / shard_meta["shard"]
                with h5py.File(shard_fn, "r") as h5f:
                    cases_grp = h5f["cases"]
                    for case_id in shard_meta["case_ids"]:
                        if case_id not in cases_grp:
                            continue
                        case_grp = cases_grp[case_id]
                        required_keys = ("image", "lm", "lm_fg_indices", "lm_bg_indices")
                        if all(key in case_grp for key in required_keys):
                            R.existing_output_fnames.add(f"{case_id}.pt")
        print("Output folder: ", R.output_folder)
        print(
            "Image files fully processed in a previous session: ",
            len(R.existing_output_fnames),
        )
        R._register_existing_pt_files()


    R._register_existing_hdf5_shards()

# %%




# %%
    R.process()
# %%
    
    # %%  # T:block_start|RegionBoundedDataGenerator.maybe_infer_bboxes
#SECTION:-------------------- maybe_infer_bboxes--------------------------------------------------------------------------------------  # T:block_meta|RegionBoundedDataGenerator.maybe_infer_bboxes
    regions = R._localiser_regions_list()  # T:self_ref|regions = self._localiser_regions_list()
    R.I = LocaliserInfererPT(  # T:self_ref|self.I = LocaliserInfererPT(
        localiser_regions=regions,
        window="a",
        bs=16,
        devices=R.devices,  # T:self_ref|    devices=self.devices,
        debug=False,
    )
    R.yolo_specs = R.I.yolo_state_dict  # T:self_ref|self.yolo_specs = self.I.yolo_state_dict
    classes_in_bbox = R.get_region_indices()  # T:self_ref|classes_in_bbox = self.get_region_indices()
# %%
    R.attach_bboxes(classes_in_bbox)  # T:self_ref|self.attach_bboxes(classes_in_bbox)
    missing = R.missing_bbox_mask()  # T:self_ref|missing = self.missing_bbox_mask()
    imgs = R.df.loc[missing, "image"].tolist()  # T:self_ref|imgs = self.df.loc[missing, "image"].tolist()
    if len(imgs) == 0:
        pass  # T:early_return|    return
    cprint(
        f"Total case {len(R.df)}. \nBBoxes on file: {len(R.df) - len(imgs)}. \nRemaining bboxes: {len(imgs)}. \nInferring missing bboxes with localiser for regions: {regions}",  # T:self_ref|    f"Total case {len(self.df)}. \\nBBoxes on file: {len(self.df) - len(imgs)}. \\nRemaining bboxes: {len(imgs)}. \\nInferring missing bboxes with localiser for regions: {regions}",
        color="blue",
    )
    R.I.run(imgs, overwrite=False)  # T:self_ref|self.I.run(imgs, overwrite=False)
    R.attach_bboxes(classes_in_bbox)  # T:self_ref|self.attach_bboxes(classes_in_bbox)
    # end PythonMethodScratch  # T:block_end|RegionBoundedDataGenerator.maybe_infer_bboxes
        

# %%



    RR = RBDSamplerWorkerLocal(
        project=R.project,
        plan=R.plan,
        data_folder=R.data_folder,
        output_folder=R.output_folder,
    )
# %%

    row = R.df.iloc[0]
    RR.debug=True
# %%
# %%
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

    classes_in_bbox = R.get_region_indices()
# %%
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

    bbox["width"] = (0.0898, 0.5)

    lm = dici["lm"]
    lm_org = lm.clone()
    counts_org = lm.count_nonzero().item()

    dici2 = C(dici)
    img = dici2["image"]
    lm = dici2["lm"]

    counts_after = lm.count_nonzero()
    counts_after == counts_org

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
    num_processes = 1
    debug_ = False
    R.setup(
        overwrite=overwrite, device="cpu", num_processes=num_processes, debug=debug_
    )
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

#SECTION:-------------------- process--------------------------------------------------------------------------------------  # T:block_meta|RegionBoundedDataGenerator.process
    # R.maybe_infer_bboxes()  # T:self_ref|self.maybe_infer_bboxes()
    R.mini_dfs = R.split_dataframe_for_workers(R.df, R.num_processes)  # T:self_ref|self.mini_dfs = self.split_dataframe_for_workers(self.df, self.num_processes)
    process_result = super().process()  # T:return|return super().process()
    # end PythonMethodScratch  # T:block_end|RegionBoundedDataGenerator.process
# %%
    a= R.mini_dfs[0].iloc[:3]
# %%
    # %%  # T:block_start|RegionBoundedDataGenerator.maybe_infer_bboxes
#SECTION:-------------------- maybe_infer_bboxes--------------------------------------------------------------------------------------  # T:block_meta|RegionBoundedDataGenerator.maybe_infer_bboxes
    regions = R._localiser_regions_list()  # T:self_ref|regions = self._localiser_regions_list()
    R.I = LocaliserInfererPT(  # T:self_ref|self.I = LocaliserInfererPT(
        localiser_regions=regions,
        window="a",
        bs=16,
        devices=R.devices,  # T:self_ref|    devices=self.devices,
        debug=False,
    )
# %%
    R.yolo_specs = R.I.yolo_state_dict  # T:self_ref|self.yolo_specs = self.I.yolo_state_dict
    classes_in_bbox = R.get_region_indices()  # T:self_ref|classes_in_bbox = self.get_region_indices()
    R.attach_bboxes(classes_in_bbox)  # T:self_ref|self.attach_bboxes(classes_in_bbox)
    missing = R.missing_bbox_mask()  # T:self_ref|missing = self.missing_bbox_mask()
    imgs = R.df.loc[missing, "image"].tolist()  # T:self_ref|imgs = self.df.loc[missing, "image"].tolist()
    if len(imgs) == 0:
        pass  # T:early_return|    return
    cprint(
        f"Total case {len(R.df)}. \nBBoxes on file: {len(R.df) - len(imgs)}. \nRemaining bboxes: {len(imgs)}. \nInferring missing bboxes with localiser for regions: {regions}",  # T:self_ref|    f"Total case {len(self.df)}. \\nBBoxes on file: {len(self.df) - len(imgs)}. \\nRemaining bboxes: {len(imgs)}. \\nInferring missing bboxes with localiser for regions: {regions}",
        color="blue",
    )
    R.I.run(imgs, overwrite=False)  # T:self_ref|self.I.run(imgs, overwrite=False)
    R.attach_bboxes(classes_in_bbox)  # T:self_ref|self.attach_bboxes(classes_in_bbox)
    # end PythonMethodScratch  # T:block_end|RegionBoundedDataGenerator.maybe_infer_bboxes
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
    img = dici["image"]
    lm = dici["lm"]
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

    # %%  # T:block_start|RegionBoundedDataGenerator.process

    # %%
    bbox_files = []
    # %%  # T:block_start|RegionBoundedDataGenerator._index_bbox_files_by_case_id

    # %%
    overwrite = None
    derive_bboxes = True
    src_dims = DEFAULT_HDF5_SRC_DIMS
    cases_per_shard = 5
    max_shard_bytes = None
    overwrite_hdf5_shards = False
    hdf5_compression = "gzip"
    hdf5_compression_opts = 1
    # %%  # T:block_start|RegionBoundedDataGenerator.process
#SECTION:-------------------- process--------------------------------------------------------------------------------------  # T:block_meta|RegionBoundedDataGenerator.process
    R.maybe_infer_bboxes()  # T:self_ref|self.maybe_infer_bboxes()
    R.mini_dfs = R.split_dataframe_for_workers(R.df, R.num_processes)  # T:self_ref|self.mini_dfs = self.split_dataframe_for_workers(self.df, self.num_processes)
    pass  # T:early_return|return super().process(
        overwrite=overwrite,
        derive_bboxes=derive_bboxes,
        src_dims=src_dims,
        cases_per_shard=cases_per_shard,
        max_shard_bytes=max_shard_bytes,
        overwrite_hdf5_shards=overwrite_hdf5_shards,
        hdf5_compression=hdf5_compression,
        hdf5_compression_opts=hdf5_compression_opts,
    )
    # end PythonMethodScratch  # T:block_end|RegionBoundedDataGenerator.process
#SECTION:-------------------- _index_bbox_files_by_case_id--------------------------------------------------------------------------------------  # T:block_meta|RegionBoundedDataGenerator._index_bbox_files_by_case_id
