# %%
import json
import itertools as il
import sqlite3
from pathlib import Path

import ipdb
import numpy as np
import pandas as pd
import ray
import SimpleITK as sitk
import torch
from fastcore.foundation import GetAttr
from tqdm.auto import tqdm
from fran.data.dataregistry import DS
from fran.managers.db import add_plan_to_db
from fran.preprocessing import bboxes_function_version
from fran.preprocessing.helpers import (
    create_dataset_stats_artifacts,
    import_h5py,
    infer_dataset_stats_window,
    postprocess_artifacts_missing,
    sanitize_meta_for_monai,
)
from fran.utils.dataset_properties import analyze_tensor_data_folder
from fran.utils.string_works import is_excel_None
from utilz.fileio import maybe_makedirs, save_dict, save_json
from utilz.helpers import create_df_from_folder, multiprocess_multiarg
from utilz.rayz import shutdown_actors
from utilz.stringz import ast_literal_eval, info_from_filename, strip_extension


def bboxes_to_df(bboxes):
    rows = []
    for case in bboxes:
        case_id = case["case_id"]
        for stat in case["bbox_stats"]:
            label = stat["label"]
            bbs = stat["bounding_boxes"]
            cents = stat["centroids"]
            for i, (bb, c) in enumerate(zip(bbs, cents)):
                z0, z1 = bb[0].start, bb[0].stop
                y0, y1 = bb[1].start, bb[1].stop
                x0, x1 = bb[2].start, bb[2].stop

                rows.append(
                    {
                        "case_id": case_id,
                        "label": label,
                        "bbox_id": i,
                        "z0": z0,
                        "z1": z1,
                        "y0": y0,
                        "y1": y1,
                        "x0": x0,
                        "x1": x1,
                        "size_z": z1 - z0,
                        "size_y": y1 - y0,
                        "size_x": x1 - x0,
                        "cz": c[0],
                        "cy": c[1],
                        "cx": c[2],
                    }
                )

    df = pd.DataFrame(rows)
    return df


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
    df = bboxes_to_df(bboxes)
    bbox_fn = masks_folder.parent / ("bboxes_info.csv")
    print("Storing bbox info in {}".format(bbox_fn))
    df.to_csv(bbox_fn)


def _itk_binary_stats_like_fran(
    mask_arr: np.ndarray, separate_islands: bool = True
) -> dict:
    """
    Build Fran-compatible bbox stats using ITK/SimpleITK:
      {
        "voxel_counts": np.ndarray,           # [bg, comp1, comp2, ...]
        "bounding_boxes": list[slice tuple],  # [bg_full, comp1_bbox, ...]
        "centroids": np.ndarray               # [bg_centroid, comp1_centroid, ...] in z,y,x index order
      }
    """
    mask_arr = np.asarray(mask_arr)
    binary = (mask_arr > 0).astype(np.uint8)
    zdim, ydim, xdim = binary.shape

    # Background entry follows current Fran/cc3d convention.
    bg_bbox = (slice(0, zdim), slice(0, ydim), slice(0, xdim))

    img = sitk.GetImageFromArray(binary)
    if separate_islands:
        label_img = sitk.ConnectedComponent(img)
    else:
        label_img = img

    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(label_img)

    entries = []
    for lbl in ls.GetLabels():
        count = int(ls.GetNumberOfPixels(lbl))
        x0, y0, z0, sx, sy, sz = ls.GetBoundingBox(lbl)
        bbox = (slice(z0, z0 + sz), slice(y0, y0 + sy), slice(x0, x0 + sx))

        # Convert ITK physical centroid to voxel index, then to z,y,x order.
        c_phys = ls.GetCentroid(lbl)
        cx, cy, cz = label_img.TransformPhysicalPointToContinuousIndex(c_phys)
        centroid_zyx = (float(cz), float(cy), float(cx))
        entries.append((count, bbox, centroid_zyx))

    # Match cc3d.largest_k behavior: largest components first.
    entries.sort(key=lambda x: x[0], reverse=True)

    fg_count = int(sum(e[0] for e in entries))
    total_count = int(binary.size)
    bg_count = int(total_count - fg_count)

    # Efficient background centroid from totals minus fg weighted sums.
    total_sum_z = (zdim * (zdim - 1) / 2.0) * ydim * xdim
    total_sum_y = (ydim * (ydim - 1) / 2.0) * zdim * xdim
    total_sum_x = (xdim * (xdim - 1) / 2.0) * zdim * ydim
    fg_sum_z = sum(e[2][0] * e[0] for e in entries)
    fg_sum_y = sum(e[2][1] * e[0] for e in entries)
    fg_sum_x = sum(e[2][2] * e[0] for e in entries)

    if bg_count > 0:
        bg_centroid = np.array(
            [
                (total_sum_z - fg_sum_z) / bg_count,
                (total_sum_y - fg_sum_y) / bg_count,
                (total_sum_x - fg_sum_x) / bg_count,
            ],
            dtype=np.float64,
        )
    else:
        bg_centroid = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    voxel_counts = np.array([bg_count] + [e[0] for e in entries], dtype=np.int64)
    bounding_boxes = [bg_bbox] + [e[1] for e in entries]
    centroids = np.vstack(
        [bg_centroid] + [np.array(e[2], dtype=np.float64) for e in entries]
    )

    return {
        "voxel_counts": voxel_counts,
        "bounding_boxes": bounding_boxes,
        "centroids": centroids,
    }


def bboxes_function_version_nifti_itk(filename, bg_label=0):
    """
    NIfTI-only bbox generator using ITK that returns the same pattern Fran expects.
    """
    filename = Path(filename)
    lm = sitk.ReadImage(str(filename))
    arr = sitk.GetArrayFromImage(lm)

    labels = np.unique(arr)
    labels = labels[labels != bg_label]
    stats_all = []

    for label in labels:
        label_mask = (arr == label).astype(np.uint8)
        stats = {"label": int(label)}
        stats.update(_itk_binary_stats_like_fran(label_mask, separate_islands=True))
        stats_all.append(stats)

    all_fg = (arr != bg_label).astype(np.uint8)
    all_fg_stats = {"label": "all_fg"}
    all_fg_stats.update(_itk_binary_stats_like_fran(all_fg, separate_islands=False))
    stats_all.append(all_fg_stats)

    case_id = info_from_filename(filename.name, full_caseid=True)["case_id"]
    return {"case_id": case_id, "filename": filename, "bbox_stats": stats_all}


def generate_bboxes_from_lms_folder_itk(
    masks_folder, bg_label=0, debug=False, num_processes=16
):
    """
    Generate Fran-style bbox info from NIfTI labelmaps (lms/*.nii or *.nii.gz)
    using ITK bbox/shape stats, and store to <masks_folder.parent>/bboxes_info.
    """
    masks_folder = Path(masks_folder)
    label_files = sorted(
        list(masks_folder.glob("*.nii.gz")) + list(masks_folder.glob("*.nii"))
    )
    arguments = [[x, bg_label] for x in label_files]
    bboxes = multiprocess_multiarg(
        func=bboxes_function_version_nifti_itk,
        arguments=arguments,
        num_processes=num_processes,
        debug=debug,
    )
    bbox_fn = masks_folder.parent / ("bboxes_info")
    print("Storing bbox info in {}".format(bbox_fn))
    save_dict(bboxes, bbox_fn)


def get_tensorfile_stats(filename):
    tnsr = torch.load(filename, weights_only=False)
    return get_tensor_stats(tnsr)


def _labels_from_lm_file(filename):
    lm = torch.load(filename, weights_only=False)
    labels = torch.unique(lm).detach().cpu().tolist()
    return [int(v) for v in labels]


def store_label_count(output_folder, num_processes=16):
    output_folder = Path(output_folder)
    lms_folder = output_folder / "lms"
    lm_files = list(lms_folder.glob("*pt"))
    labels_all = set()
    if len(lm_files) > 0:
        arguments = [[fn] for fn in lm_files]
        labels_per_file = multiprocess_multiarg(
            func=_labels_from_lm_file,
            arguments=arguments,
            num_processes=num_processes,
            debug=False,
        )
        for labels in labels_per_file:
            labels_all.update(labels)
    labels_all = sorted(int(v) for v in labels_all)
    out_fn = output_folder / "labels_all.json"
    save_json(labels_all, out_fn)


def get_tensor_stats(tnsr) -> dict:
    dic = {
        "max": tnsr.max().item(),
        "min": tnsr.min().item(),
        "median": np.median(tnsr),
        "shape": [*tnsr.shape],
    }
    return dic


HDF5_SHARD_CHUNKS = {
    (192, 192, 128): {
        "image": (192, 192, 128),
        "lm": (192, 192, 128),
        "indices": (262144,),
    }
}


def _normalize_src_dims(src_dims):
    dims = tuple(int(v) for v in src_dims)
    if len(dims) != 3 or any(v <= 0 for v in dims):
        raise ValueError(f"src_dims must be 3 positive ints, got {src_dims}")
    return dims


def _hdf5_chunks_for(shape, key, src_dims: tuple=(192, 192, 128)):
    shape = tuple(int(v) for v in shape)
    # src_dims = _normalize_src_dims(src_dims)
    conf = HDF5_SHARD_CHUNKS.get(
        src_dims,
        {"image": src_dims, "lm": src_dims, "indices": (262144,)},
    )

    if key in ("image", "lm"):
        if len(shape) != 3:
            raise ValueError(f"{key} expected 3D shape, got {shape}")
        base = conf[key]
        return tuple(min(int(dim), int(chunk_dim)) for dim, chunk_dim in zip(shape, base))

    if key in ("indices", "lm_fg_indices", "lm_bg_indices"):
        if len(shape) != 1:
            raise ValueError(f"{key} expected 1D shape, got {shape}")
        idx_chunk = int(conf["indices"][0])
        return (max(1, min(int(shape[0]), idx_chunk)),)

    raise KeyError(f"Unknown HDF5 chunk key: {key}")


def _load_torch_for_hdf5(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _to_numpy_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _create_index_dataset(case_grp, name, data, src_dims, ds_kwargs):
    if int(data.shape[0]) > 0:
        case_grp.create_dataset(
            name,
            data=data,
            chunks=_hdf5_chunks_for(data.shape, name, src_dims),
            **ds_kwargs,
        )
        return
    case_grp.create_dataset(name, data=data, **ds_kwargs)


def _write_case_to_hdf5_shard(
    h5f,
    case_id,
    image_pt,
    lm_pt,
    indices_pt,
    src_dims,
    compression,
    compression_opts,
):
    image = _to_numpy_cpu(_load_torch_for_hdf5(image_pt))
    lm = _to_numpy_cpu(_load_torch_for_hdf5(lm_pt))
    indices = _load_torch_for_hdf5(indices_pt)

    if not isinstance(indices, dict):
        raise ValueError(f"indices file must be a dict: {indices_pt}")
    if "lm_fg_indices" not in indices or "lm_bg_indices" not in indices:
        raise KeyError(f"indices dict missing lm_fg_indices/lm_bg_indices: {indices_pt}")

    fg = _to_numpy_cpu(indices["lm_fg_indices"]).reshape(-1)
    bg = _to_numpy_cpu(indices["lm_bg_indices"]).reshape(-1)

    ds_kwargs = {}
    if compression is not None:
        ds_kwargs["compression"] = compression
        if compression_opts is not None:
            ds_kwargs["compression_opts"] = compression_opts
        ds_kwargs["shuffle"] = True

    cases_grp = h5f.require_group("cases")
    case_grp = cases_grp.create_group(case_id)
    case_grp.create_dataset(
        "image",
        data=image,
        chunks=_hdf5_chunks_for(image.shape, "image", src_dims),
        **ds_kwargs,
    )
    case_grp.create_dataset(
        "lm",
        data=lm,
        chunks=_hdf5_chunks_for(lm.shape, "lm", src_dims),
        **ds_kwargs,
    )
    _create_index_dataset(
        case_grp=case_grp,
        name="lm_fg_indices",
        data=fg,
        src_dims=src_dims,
        ds_kwargs=ds_kwargs,
    )
    _create_index_dataset(
        case_grp=case_grp,
        name="lm_bg_indices",
        data=bg,
        src_dims=src_dims,
        ds_kwargs=ds_kwargs,
    )

    case_grp.attrs["image_pt"] = str(image_pt)
    case_grp.attrs["lm_pt"] = str(lm_pt)
    case_grp.attrs["indices_pt"] = str(indices_pt)
    case_grp.attrs["image_shape"] = list(image.shape)
    case_grp.attrs["lm_shape"] = list(lm.shape)

    meta = indices.get("meta")
    if isinstance(meta, dict):
        meta = sanitize_meta_for_monai(dict(meta))
        case_grp.attrs["meta_json"] = json.dumps(meta, default=str)
        filename_or_obj = meta.get("filename_or_obj")
        if filename_or_obj is not None:
            case_grp.attrs["source_meta_filename_or_obj"] = str(filename_or_obj)
    elif meta is not None:
        case_grp.attrs["meta_json"] = json.dumps(meta, default=str)


def _build_shard_groups(case_records, cases_per_shard, max_shard_bytes):
    assert cases_per_shard is None or max_shard_bytes is None
    if max_shard_bytes is not None:
        max_shard_bytes = int(max_shard_bytes)
        if max_shard_bytes <= 0:
            raise ValueError(f"max_shard_bytes must be > 0, got {max_shard_bytes}")
        groups = []
        current = []
        current_bytes = 0
        for rec in case_records:
            rec_bytes = rec["total_bytes"]
            if current and current_bytes + rec_bytes > max_shard_bytes:
                groups.append(current)
                current = [rec]
                current_bytes = rec_bytes
            else:
                current.append(rec)
                current_bytes += rec_bytes
        if current:
            groups.append(current)
        return groups

    cases_per_shard = int(cases_per_shard)
    if cases_per_shard <= 0:
        raise ValueError(f"cases_per_shard must be > 0, got {cases_per_shard}")

    return [
        case_records[idx : idx + cases_per_shard]
        for idx in range(0, len(case_records), cases_per_shard)
    ]


def create_hdf5_shards(
    output_folder,
    src_dims,
    cases_per_shard=5,
    max_shard_bytes=None,
    overwrite=False,
    compression="gzip",
    compression_opts=1,
):
    output_folder = Path(output_folder)
    src_dims = _normalize_src_dims(src_dims)
    images_folder = output_folder / "images"
    lms_folder = output_folder / "lms"
    indices_folder = output_folder / "indices"

    for folder in (images_folder, lms_folder, indices_folder):
        if not folder.exists():
            raise FileNotFoundError(f"Required folder missing: {folder}")

    src_tag = "_".join(str(v) for v in src_dims)
    shards_folder = output_folder / "hdf5_shards" / f"src_{src_tag}"
    manifest_fn = shards_folder / "manifest.json"
    maybe_makedirs([shards_folder])

    existing_shards = sorted(shards_folder.glob("shard_*.h5"))
    if manifest_fn.exists() and not overwrite:
        print(f"HDF5 shards already present, skipping: {manifest_fn}")
        return existing_shards
    if existing_shards and not overwrite:
        raise FileExistsError(
            f"Existing shard files found in {shards_folder}. Set overwrite=True to regenerate."
        )
    if overwrite:
        for pth in existing_shards:
            pth.unlink()
        if manifest_fn.exists():
            manifest_fn.unlink()

    image_case_ids = {pth.stem for pth in images_folder.glob("*.pt")}
    lm_case_ids = {pth.stem for pth in lms_folder.glob("*.pt")}
    indices_case_ids = {pth.stem for pth in indices_folder.glob("*.pt")}
    case_ids = sorted(image_case_ids & lm_case_ids & indices_case_ids)
    if len(case_ids) == 0:
        raise ValueError(
            f"No shared case IDs found across images/lms/indices in {output_folder}"
        )

    case_records = []
    for case_id in case_ids:
        image_pt = images_folder / f"{case_id}.pt"
        lm_pt = lms_folder / f"{case_id}.pt"
        indices_pt = indices_folder / f"{case_id}.pt"
        total_bytes = image_pt.stat().st_size + lm_pt.stat().st_size + indices_pt.stat().st_size
        case_records.append(
            {
                "case_id": case_id,
                "image_pt": image_pt,
                "lm_pt": lm_pt,
                "indices_pt": indices_pt,
                "total_bytes": int(total_bytes),
            }
        )

    shard_groups = _build_shard_groups(case_records, cases_per_shard, max_shard_bytes)
    h5py = import_h5py()
    shard_paths = []
    shard_manifest = []
    for shard_idx, shard_cases in enumerate(shard_groups):
        shard_fn = shards_folder / f"shard_{shard_idx:04d}.h5"
        with h5py.File(shard_fn, "w") as h5f:
            case_ids_shard = [rec["case_id"] for rec in shard_cases]
            h5f.attrs["format"] = "fran_hdf5_shards_v1"
            h5f.attrs["src_dims"] = list(src_dims)
            h5f.attrs["cases_per_shard"] = int(cases_per_shard)
            h5f.attrs["case_ids_json"] = json.dumps(case_ids_shard)
            h5f.attrs["compression"] = "" if compression is None else str(compression)
            h5f.attrs["compression_opts"] = -1 if compression_opts is None else int(
                compression_opts
            )

            for rec in shard_cases:
                _write_case_to_hdf5_shard(
                    h5f=h5f,
                    case_id=rec["case_id"],
                    image_pt=rec["image_pt"],
                    lm_pt=rec["lm_pt"],
                    indices_pt=rec["indices_pt"],
                    src_dims=src_dims,
                    compression=compression,
                    compression_opts=compression_opts,
                )

        shard_paths.append(shard_fn)
        shard_manifest.append({"shard": shard_fn.name, "case_ids": case_ids_shard})

    manifest = {
        "format": "fran_hdf5_shards_v1",
        "src_dims": list(src_dims),
        "compression": compression,
        "compression_opts": compression_opts,
        "cases_per_shard": int(cases_per_shard),
        "max_shard_bytes": max_shard_bytes,
        "num_cases": len(case_records),
        "num_shards": len(shard_paths),
        "shards": shard_manifest,
    }
    save_json(manifest, manifest_fn)
    print(f"Wrote {len(shard_paths)} HDF5 shards in {shards_folder}")
    return shard_paths


class Preprocessor(GetAttr):
    _default = "project"
    PREPROCESS_LOG_COLUMNS = [
        "case_id",
        "status",
        "image",
        "lm",
        "error_type",
        "error_message",
        "traceback",
    ]

    def __init__(
        self,
        project,
        plan,
        data_folder=None,
        output_folder=None,
    ) -> None:
        self.project = project
        self.plan = plan
        self.data_folder = data_folder
        self.store_gifs = False
        self.store_label_stats = True
        self.set_input_output_folders(data_folder, output_folder)

    def _df_from_db(self):
        con = sqlite3.connect(str(self.project.db))
        query = "SELECT case_id, image, lm, ds , alias FROM datasources"
        df = pd.read_sql_query(query, con)
        con.close()
        dfs = []
        for ds in df["ds"].unique():
            df_ds = df[df["ds"] == ds].copy()
            alias = df_ds["alias"].iloc[0]
            if not is_excel_None(alias):
                df_ds["ds"] = alias
            dfs.append(df_ds)
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop(columns=["alias"])
        df["image"] = df["image"].apply(Path)
        df["lm"] = df["lm"].apply(Path)
        return df

    def _df_from_folder(self):
        df = create_df_from_folder(self.data_folder)
        extract_ds = lambda x: x.split("_")[0]
        df["ds"] = df["case_id"].apply(extract_ds)
        return df

    def create_data_df(self):
        if self.data_folder is not None:
            data_folder = Path(self.data_folder)
            raw_data_folder = Path(self.project.raw_data_folder)
            if data_folder == raw_data_folder and Path(self.project.db).exists():
                self.df = self._df_from_db()
            else:
                self.df = self._df_from_folder()
            assert len(self.df) > 0, "No valid case files found in {}".format(
                self.data_folder
            )
            self.case_ids = self.df["case_id"].tolist()

        else:
            self.df = self.project.df
            self.case_ids = self.project.case_ids

        self.df = self.df.map(lambda x: x.lower() if isinstance(x, str) else x)
        print("Total number of cases: ", len(self.df))

    def set_remapping_per_ds(self):
        datasources = self.plan.get("datasources")
        datasources = datasources.replace(" ", "").split(",")
        remappings = self.plan.get(self.remapping_key)
        remappings = ast_literal_eval(remappings)
        if remappings is None:
            remappings = [None] * len(datasources)
        assert len(remappings) == len(datasources), (
            f"There should be a unique remapping for each datasource.\n Got {len(datasources)} datasources and {len(remappings)} remappingss"
        )
        for ds, remapping in zip(datasources, remappings):
            dss = getattr(DS, ds)
            mask = self.df["ds"] == dss.name
            if mask.sum() == 0:
                dss_in_df = self.df.ds.unique().tolist()
                raise ValueError(
                    f"Datasource {ds} not found in df.\nDatasources present: {dss_in_df}"
                )
            self.df.loc[mask, self.remapping_key] = [remapping] * mask.sum()

    def set_input_output_folders(self, data_folder, output_folder):
        raise NotImplementedError

    def save_pt(self, tnsr, subfolder, contiguous=True, suffix: str = None):
        if contiguous == True:
            tnsr = tnsr.contiguous()
        if hasattr(tnsr, "meta") and isinstance(tnsr.meta, dict):
            tnsr.meta = sanitize_meta_for_monai(dict(tnsr.meta))
        fn = Path(tnsr.meta["filename_or_obj"])
        fn_name = strip_extension(fn.name)
        if suffix:
            fn_name = fn_name + "_" + suffix + ".pt"
        else:
            fn_name = fn_name + ".pt"

        fn = self.output_folder / subfolder / fn_name
        try:
            torch.save(tnsr, fn)
        except OSError as e:
            # get filesystem info
            try:
                usage = shutil.disk_usage(os.path.dirname(fn))
                fsinfo = f"Total={usage.total // (1024**3)}G, Used={usage.used // (1024**3)}G, Free={usage.free // (1024**3)}G"
            except Exception:
                fsinfo = "disk usage unavailable"

            print(f"[ERROR] Failed saving to {fn}")
            print(f"[ERROR] Filesystem info: {fsinfo}")

            raise RuntimeError(f"Quota exceeded at path: {fn}") from e

    def register_existing_files(self):
        existing_img = {p.name for p in (self.output_folder / "images").glob("*.pt")}
        existing_lm = {p.name for p in (self.output_folder / "lms").glob("*.pt")}
        self.existing_output_fnames = existing_img.intersection(existing_lm)
        print("Output folder: ", self.output_folder)
        print(
            "Image files fully processed in a previous session: ",
            len(self.existing_output_fnames),
        )

    def remove_completed_cases(self):
        if not getattr(self, "existing_output_fnames", None):
            return
        n_before = len(self.df)
        keep_mask = self.df["image"].apply(
            lambda x: (
                strip_extension(Path(x).name) + ".pt" not in self.existing_output_fnames
            )
        )
        self.df = self.df[keep_mask]
        print("Image files remaining to process:", len(self.df), "/", n_before)

    def save_indices(self, indices_dict, subfolder, suffix: str = None):
        fn = Path(indices_dict["meta"]["filename_or_obj"])
        fn_name = strip_extension(fn.name)
        if suffix:
            fn_name = fn_name + "_" + suffix + ".pt"
        else:
            fn_name = fn_name + ".pt"
        # fn_name = strip_extension(fn.name) + ".pt"
        fn = self.output_folder / subfolder / fn_name
        torch.save(indices_dict, fn)

    def shutdown_ray_workers(self):
        actors = getattr(self, "actors", None)
        if not actors:
            return
        shutdown_actors(actors)
        self.actors = []
        self.n_actors = 0

    def run_worker_jobs(self):
        if getattr(self, "use_ray", False):
            try:
                return ray.get(
                    [
                        actor.process.remote(mini_df)
                        for actor, mini_df in zip(self.actors, self.mini_dfs)
                    ]
                )
            finally:
                # Release actor CPU reservations before downstream Ray-based
                # postprocessing stages create their own actor pools.
                self.shutdown_ray_workers()
        return [self.local_worker.process(self.mini_dfs[0])]

    def flatten_results(self, results):
        df = pd.DataFrame(il.chain.from_iterable(results))
        audit_cols = [col for col in df.columns if str(col).startswith("_preprocess_")]
        if audit_cols:
            df = df.drop(columns=audit_cols)
        return df

    @property
    def results_csv_fn(self):
        return self.output_folder / "resampled_dataset_properties.csv"

    def _read_existing_results_df(self):
        csv_fn = self.results_csv_fn
        if not csv_fn.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(csv_fn)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _write_results_csv(self):
        existing_df = self._read_existing_results_df()
        current_df = getattr(self, "results_df", pd.DataFrame())
        if current_df is None:
            current_df = pd.DataFrame()
        results_df = pd.concat([existing_df, current_df], ignore_index=True, sort=False)
        fn_series = (
            results_df["fn_name"]
            if "fn_name" in results_df.columns
            else pd.Series([None] * len(results_df), index=results_df.index)
        )
        case_series = (
            results_df["case_id"]
            if "case_id" in results_df.columns
            else pd.Series([None] * len(results_df), index=results_df.index)
        )
        resume_key = fn_series.where(fn_series.notna(), case_series)
        if resume_key.notna().any():
            results_df = results_df.assign(_resume_key=resume_key)
            results_df = results_df.drop_duplicates(subset=["_resume_key"], keep="last")
            results_df = results_df.drop(columns=["_resume_key"])
        self.results_df = results_df
        self.results_df.to_csv(self.results_csv_fn, index=False)

    @staticmethod
    def _coerce_log_value(value):
        if value is None:
            return ""
        return str(value)

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

    def build_preprocessing_log_rows(self, results):
        if len(results) != len(self.mini_dfs):
            raise ValueError(
                "Worker output group mismatch: "
                f"got {len(results)} worker output groups for {len(self.mini_dfs)} mini_dfs"
            )
        rows = []
        for mini_df, worker_outs in zip(self.mini_dfs, results):
            if len(worker_outs) != len(mini_df):
                raise ValueError(
                    "Worker output length mismatch: "
                    f"got {len(worker_outs)} rows for mini_df of size {len(mini_df)}"
                )
            for (_, src_row), worker_out in zip(mini_df.iterrows(), worker_outs):
                case_id = self._coerce_log_value(src_row.get("case_id"))
                image = self._coerce_log_value(src_row.get("image"))
                lm = self._coerce_log_value(src_row.get("lm"))
                status = "OK"
                error_type = ""
                error_message = ""
                trace = ""
                if isinstance(worker_out, dict):
                    err_info = worker_out.get("_preprocess_error")
                    if isinstance(err_info, dict):
                        status = "ERROR"
                        error_type = self._coerce_log_value(err_info.get("error_type"))
                        error_message = self._coerce_log_value(
                            err_info.get("error_message")
                        )
                        trace = self._coerce_log_value(err_info.get("traceback"))
                    else:
                        events = self._normalize_preprocess_events(
                            worker_out.get("_preprocess_events")
                        )
                        if events:
                            status = "WARNING"
                            unique_types = []
                            messages = []
                            for event in events:
                                evt_type = event["error_type"]
                                if evt_type not in unique_types:
                                    unique_types.append(evt_type)
                                messages.append(event["error_message"])
                            error_type = "; ".join(unique_types)
                            error_message = "; ".join(messages)
                rows.append(
                    {
                        "case_id": case_id,
                        "status": status,
                        "image": image,
                        "lm": lm,
                        "error_type": error_type,
                        "error_message": error_message,
                        "traceback": trace,
                    }
                )
        return rows

    def write_preprocessing_log(self, results):
        rows = self.build_preprocessing_log_rows(results)
        df = pd.DataFrame(rows, columns=self.PREPROCESS_LOG_COLUMNS)
        log_fn = self.output_folder / "preprocessing_log.csv"
        df.to_csv(log_fn, index=False)

    def postprocess_results(self, **process_kwargs):
        self._write_results_csv()
        self._store_dataset_properties()
        store_label_count(
            self.output_folder, num_processes=getattr(self, "num_processes", 1)
        )
        create_dataset_stats_artifacts(
            lms_folder=self.output_folder / "lms",
            gif=self.store_gifs,
            label_stats=self.store_label_stats,
            gif_window=infer_dataset_stats_window(self.project),
        )
        folder_key = getattr(self, "subfolder_key", "data_folder_source")
        folder_kwargs = (
            {folder_key: self.output_folder}
            if folder_key in {
                "data_folder_source",
                "data_folder_lbd",
                "data_folder_whole",
                "data_folder_pbd",
            }
            else {"data_folder_source": self.output_folder}
        )
        add_plan_to_db(self.plan, db_path=self.project.db, **folder_kwargs)

    def initialize_process_state(self):
        self.create_output_folders()
        self.results = []
        self.shapes = []

    def postprocess_artifacts_missing(self):
        stats_folder = self.output_folder / "dataset_stats"
        required = [
            self.output_folder / "labels_all.json",
            self.output_folder / "resampled_dataset_properties.json",
        ]
        if self.store_gifs:
            required.append(stats_folder / "snapshot.gif")
        if self.store_label_stats:
            required.append(stats_folder / "lesion_stats.csv")
        missing = [pth for pth in required if not pth.exists()]
        if (
            (self.store_gifs or self.store_label_stats)
            and stats_folder.exists()
            and not any(stats_folder.iterdir())
        ):
            missing.append(stats_folder)
        if missing:
            print("Missing postprocess artifacts:")
            for pth in missing:
                print(" ", pth)
        return len(missing) > 0

    def run_postprocess_only(self, **process_kwargs):
        print("Running postprocess on existing output tensors")
        self.results_df = pd.DataFrame()
        self.postprocess_results(**process_kwargs)
        self._maybe_create_hdf5_shards(**process_kwargs)
        return self.results_df

    def process(self, **process_kwargs):
        if not hasattr(self, "df"):
            print("No data frames have been created. Run setup")
            return 0
        if len(self.df) == 0:
            if getattr(self, "run_postprocess_if_empty", False):
                return self.run_postprocess_only(**process_kwargs)
            print("No data frames have been created. Run setup")
            return 0
        self.initialize_process_state()
        self.results = self.run_worker_jobs()
        self.write_preprocessing_log(self.results)
        self.results_df = self.flatten_results(self.results)
        self.postprocess_results(**process_kwargs)
        self._maybe_create_hdf5_shards(**process_kwargs)
        return self.results_df

    def create_hdf5_shards(self, **kwargs):
        return create_hdf5_shards(output_folder=self.output_folder, **kwargs)

    def _maybe_create_hdf5_shards(self, **process_kwargs):
        if not process_kwargs.get("create_hdf5_shards", False):
            return []
        return self.create_hdf5_shards(
            src_dims=process_kwargs.get("src_dims", (192, 192, 128)),
            cases_per_shard=process_kwargs.get("cases_per_shard", 5),
            max_shard_bytes=process_kwargs.get("max_shard_bytes"),
            overwrite=process_kwargs.get("overwrite_hdf5_shards", False),
            compression=process_kwargs.get("hdf5_compression", "gzip"),
            compression_opts=process_kwargs.get("hdf5_compression_opts", 1),
        )

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
            self.save_pt(image[0], "images")
            self.save_pt(lm[0], "lms")
            self.results.append(get_tensor_stats(image))

    def get_tensor_folder_stats(self, debug=True):
        analysis = analyze_tensor_data_folder(
            self.output_folder / ("images"),
            glob_pattern="*",
            debug=debug,
            recursive=False,
            include_per_file_stats=True,
        )
        results = analysis["per_file_stats"]
        self.shapes = [a["shape"] for a in results]
        self.results_df = pd.DataFrame(results)  # .values
        self.results = self.results_df[["max", "min", "median"]]
        self._store_dataset_properties()

    def _collect_output_folder_stats(self, debug=False):
        num_processes = getattr(self, "num_processes", 1)
        image_stats = analyze_tensor_data_folder(
            self.output_folder / "images",
            glob_pattern="*",
            num_processes=num_processes,
            debug=debug,
            recursive=False,
            include_per_file_stats=False,
        )
        lm_stats = analyze_tensor_data_folder(
            self.output_folder / "lms",
            glob_pattern="*",
            num_processes=num_processes,
            debug=debug,
            recursive=False,
            include_per_file_stats=False,
        )
        return image_stats, lm_stats

    def _store_dataset_properties(self):
        self.image_folder_stats, self.lm_folder_stats = (
            self._collect_output_folder_stats()
        )
        print("Caculating data shape and intensity profile.")
        resampled_dataset_properties = self.create_properties_dict()
        resampled_dataset_properties_fname = (
            self.output_folder / "resampled_dataset_properties.json"
        )
        print(
            "Writing preprocessing output properties to {}".format(
                resampled_dataset_properties_fname
            )
        )
        save_json(resampled_dataset_properties, resampled_dataset_properties_fname)

    def create_properties_dict(self):
        resampled_dataset_properties = dict()
        lm_stats = getattr(self, "lm_folder_stats", {})
        image_stats = getattr(self, "image_folder_stats", {})
        intensity_profile = image_stats.get("intensity_profile", {})

        resampled_dataset_properties["min_shape"] = lm_stats.get("min_shape", np.nan)
        resampled_dataset_properties["median_shape"] = lm_stats.get(
            "median_shape", np.nan
        )
        resampled_dataset_properties["max_shape"] = lm_stats.get("max_shape", np.nan)

        resampled_dataset_properties["dataset_spacing"] = self.plan.get("spacing")
        resampled_dataset_properties["dataset_max"] = intensity_profile.get(
            "dataset_max", np.nan
        )
        resampled_dataset_properties["dataset_min"] = intensity_profile.get(
            "dataset_min", np.nan
        )
        resampled_dataset_properties["dataset_median"] = intensity_profile.get(
            "dataset_median", np.nan
        )
        return resampled_dataset_properties

    def create_output_folders(self):
        maybe_makedirs(
            [
                self.output_folder / ("lms"),
                self.output_folder / ("images"),
                self.indices_subfolder,
            ]
        )

    def ray_init(self):
        if not ray.is_initialized():
            from fran.utils.common import COMMON_PATHS

            ray_fldr = COMMON_PATHS["ray_folder"]
            ray_tmp = Path(ray_fldr) / ("tmp")
            try:
                ray.init(
                    ignore_reinit_error=True,
                    include_dashboard=False,
                    _temp_dir=str(ray_tmp),
                )
            except Exception as e:
                print("Ray init warning:", e)

    def split_dataframe_for_workers(self, df, num_processes: int):
        if len(df) == 0:
            return []
        n = max(1, min(len(df), int(num_processes)))
        return [
            df.iloc[idx].reset_index(drop=True)
            for idx in np.array_split(np.arange(len(df)), n)
            if len(idx) > 0
        ]

    def ray_prepare(self, actor_kwargs: dict, num_processes: int):
        self.ray_init()
        n = max(1, min(len(self.df), int(num_processes))) if len(self.df) else 0
        self.n_actors = n
        self.mini_dfs = self.split_dataframe_for_workers(self.df, n)
        self.actors = (
            [self.actor_cls.remote(**actor_kwargs) for _ in range(n)] if n else []
        )

    def extra_worker_kwargs(self, **setup_kwargs):
        return {}

    def build_worker_kwargs(self, device="cpu", **setup_kwargs):
        worker_kwargs = dict(
            project=self.project,
            plan=self.plan,
            data_folder=self.data_folder,
            output_folder=self.output_folder,
            device=device,
        )
        if hasattr(self, "debug"):
            worker_kwargs["debug"] = self.debug
        worker_kwargs.update(self.extra_worker_kwargs(**setup_kwargs))
        return worker_kwargs

    def should_use_ray(self):
        debug = getattr(self, "debug", False)
        return (self.num_processes > 1) and (not debug)

    def setup(
        self, overwrite=False, num_processes=8, device="cpu", **setup_kwargs
    ):
        self.num_processes = max(1, int(num_processes))
        if "debug" in setup_kwargs:
            self.debug = setup_kwargs["debug"]
        self.run_postprocess_if_empty = False
        self.create_data_df()
        if getattr(self, "remapping_key", None) is not None:
            self.set_remapping_per_ds()
        self.register_existing_files()
        print("Overwrite:", overwrite)
        if overwrite == False:
            self.remove_completed_cases()
        if len(self.df) == 0:
            missing_arts = postprocess_artifacts_missing(self.output_folder)
            missing_all = all(missing_arts.values())
            self.run_postprocess_if_empty = missing_all
            return

        worker_kwargs = self.build_worker_kwargs(device=device, **setup_kwargs)
        self.use_ray = self.should_use_ray()
        if hasattr(self, "debug"):
            print(
                f"use_ray={self.use_ray} (num_processes={self.num_processes}, debug={self.debug})"
            )
        if self.use_ray:
            self.ray_prepare(worker_kwargs, self.num_processes)
        else:
            self.mini_dfs = [self.df]
            self.local_worker = self.local_worker_cls(**worker_kwargs)

    # @property
    # def indices_subfolder(self):
    #     indices_subfolder = self.output_folder / ("indices")
    #     return indices_subfolder


# %%
# %%
#SECTION:-------------------- --------------------------------------------------------------------------------------
if __name__ == "__main__":
# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
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
    plan = conf["plan_train"]
# %%
#
    bboxes_fldr = Path(
        "/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_ric03e8a587_ex000"
    )
    bboxes_fldr = Path(
        "/r/datasets/preprocessed/lidc/lbd/spc_075_075_075_rlb109adb5e_rlb109adb5e_ex000"
    )
    lms = bboxes_fldr / "lms"
    # generate_bboxes_from_lms_folder(lms, debug=False)

# %%

    src_dims = conf["dataset_params"]["src_dims"] 
    src_dims = tuple(src_dims)
    cases_per_shard=4
    # max_shard_bytes=2_000_000_000
    max_shard_bytes=None
    overwrite=False
    compression="gzip"
    compression_opts=1
# %%
    output_folder = "/r/datasets/preprocessed/kits23/kbd/spc_080_080_150_54787144"
    output_folder = Path(output_folder)
    images_folder = output_folder / "images"
    lms_folder = output_folder / "lms"
    indices_folder = output_folder / "indices"

    shards_folder = R.output_folder / "hdf5_shards"
# %%
#     src_tag = "_".join(str(v) for v in src_dims)
#     shards_folder = output_folder / "hdf5_shards" / f"src_{src_tag}"
#     manifest_fn = shards_folder / "manifest.json"
#     maybe_makedirs([shards_folder])
#
#     existing_shards = sorted(shards_folder.glob("shard_*.h5"))
#     if manifest_fn.exists() and not overwrite:
#         print(f"HDF5 shards already present, skipping: {manifest_fn}")
#         return existing_shards
#     if existing_shards and not overwrite:
#         raise FileExistsError(
#             f"Existing shard files found in {shards_folder}. Set overwrite=True to regenerate."
#         )
#     if overwrite:
#         for pth in existing_shards:
#             pth.unlink()
#         if manifest_fn.exists():
#             manifest_fn.unlink()
#
# # %%
    image_case_ids = {pth.stem for pth in images_folder.glob("*.pt")}
    lm_case_ids = {pth.stem for pth in lms_folder.glob("*.pt")}
    indices_case_ids = {pth.stem for pth in indices_folder.glob("*.pt")}
    case_ids = sorted(image_case_ids & lm_case_ids & indices_case_ids)
    if len(case_ids) == 0:
        raise ValueError(
            f"No shared case IDs found across images/lms/indices in {output_folder}"
        )

    case_records = []
    for case_id in case_ids:
        image_pt = images_folder / f"{case_id}.pt"
        lm_pt = lms_folder / f"{case_id}.pt"
        indices_pt = indices_folder / f"{case_id}.pt"
        total_bytes = image_pt.stat().st_size + lm_pt.stat().st_size + indices_pt.stat().st_size
        case_records.append(
            {
                "case_id": case_id,
                "image_pt": image_pt,
                "lm_pt": lm_pt,
                "indices_pt": indices_pt,
                "total_bytes": int(total_bytes),
            }
        )
# %%

    shard_groups = _build_shard_groups(case_records, cases_per_shard, max_shard_bytes)
    h5py = import_h5py()
    shard_paths = []
    shard_manifest = []
    for shard_idx, shard_cases in tqdm(enumerate(shard_groups)):
        shard_fn = shards_folder / f"shard_{shard_idx:04d}.h5"
        with h5py.File(shard_fn, "w") as h5f:

            case_ids_shard = [rec["case_id"] for rec in shard_cases]
            h5f.attrs["format"] = "fran_hdf5_shards_v1"
            h5f.attrs["src_dims"] = list(src_dims)
            h5f.attrs["cases_per_shard"] = int(cases_per_shard)
            h5f.attrs["case_ids_json"] = json.dumps(case_ids_shard)
            h5f.attrs["compression"] = "" if compression is None else str(compression)
            h5f.attrs["compression_opts"] = -1 if compression_opts is None else int(
                compression_opts
            )
            for rec in shard_cases:
                _write_case_to_hdf5_shard(
                    h5f=h5f,
                    case_id=rec["case_id"],
                    image_pt=rec["image_pt"],
                    lm_pt=rec["lm_pt"],
                    indices_pt=rec["indices_pt"],
                    src_dims=src_dims,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            cprint(f"Writing file {shard_fn}", "blue")

        shard_paths.append(shard_fn)
        shard_manifest.append({"shard": shard_fn.name, "case_ids": case_ids_shard})
# %%

    manifest = {
        "format": "fran_hdf5_shards_v1",
        "src_dims": list(src_dims),
        "compression": compression,
        "compression_opts": compression_opts,
        "cases_per_shard": int(cases_per_shard),
        "max_shard_bytes": max_shard_bytes,
        "num_cases": len(case_records),
        "num_shards": len(shard_paths),
        "shards": shard_manifest,
    }
    save_json(manifest, manifest_fn)
    print(f"Wrote {len(shard_paths)} HDF5 shards in {shards_folder}")
    # return shard_paths



    masks_folder = bboxes_fldr
    label_files = list(masks_folder.glob("*pt"))
# %%
    bg_label = 0
    arguments = [
        [x, bg_label] for x in label_files
    ]  # 0.2 factor for thresholding as kidneys are small on low-res imaging and will be wiped out by default threshold 3000
# %%
    debug = False
    num_processes = 12
    bboxes = multiprocess_multiarg(
        func=bboxes_function_version,
        arguments=arguments,
        num_processes=num_processes,
        debug=debug,
    )

    df = pd.DataFrame(bboxes)
    df = df.explode("bbox_stats").reset_index(drop=True)


# %%
    output_folder = "/r/datasets/preprocessed/kits23/kbd/spc_080_080_150_54787144/lms"
    create_dataset_stats_artifacts(
            lms_folder=output_folder,
            gif=True,
            label_stats=True,
            gif_window="abdomen",
        )

# %
