# %%
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
from fran.data.dataregistry import DS
from fran.preprocessing import bboxes_function_version
from fran.preprocessing.helpers import (
    create_dataset_stats_artifacts,
    infer_dataset_stats_window,
    postprocess_artifacts_missing,
    sanitize_meta_for_monai,
)
from fran.utils.dataset_properties import analyze_tensor_data_folder
from fran.utils.string_works import is_excel_None
from utilz.fileio import maybe_makedirs, save_dict, save_json
from utilz.helpers import create_df_from_folder, multiprocess_multiarg
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


class Preprocessor(GetAttr):
    _default = "project"

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

    def run_worker_jobs(self):
        if getattr(self, "use_ray", False):
            return ray.get(
                [
                    actor.process.remote(mini_df)
                    for actor, mini_df in zip(self.actors, self.mini_dfs)
                ]
            )
        return [self.local_worker.process(self.mini_dfs[0])]

    def flatten_results(self, results):
        return pd.DataFrame(il.chain.from_iterable(results))

    def postprocess_results(self, **process_kwargs):
        ts = self.results_df.shape
        if ts[-1] == 4:
            self._store_dataset_properties()
            generate_bboxes_from_lms_folder(self.output_folder / ("lms"))
        else:
            print(
                "self.results  shape is {0}. Last element should be 4 , is {1}. therefore".format(
                    ts, ts[-1]
                )
            )
            print(
                "since some files skipped, dataset stats are not being stored. run self.get_tensor_folder_stats and generate_bboxes_from_lms_folder separately"
            )
        add_plan_to_db(self.plan, self.output_folder, db_path=self.project.db)

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
        derive_bboxes = process_kwargs.get("derive_bboxes", True)
        self._store_dataset_properties()
        if derive_bboxes:
            generate_bboxes_from_lms_folder(
                self.output_folder / "lms",
                num_processes=getattr(self, "num_processes", 1),
            )
        store_label_count(
            self.output_folder, num_processes=getattr(self, "num_processes", 1)
        )
        create_dataset_stats_artifacts(
            lms_folder=self.output_folder/"lms",
            gif=self.store_gifs,
            label_stats=self.store_label_stats,
            gif_window=infer_dataset_stats_window(self.project),
        )
        return 1

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
        self.results_df = self.flatten_results(self.results)
        self.postprocess_results(**process_kwargs)
        return self.results_df

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

    def setup_workers(
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

if __name__ == "__main__":
    bboxes_fldr = Path(
        "/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_ric03e8a587_ex000"
    )
    bboxes_fldr = Path(
        "/r/datasets/preprocessed/lidc/lbd/spc_075_075_075_rlb109adb5e_rlb109adb5e_ex000"
    )
    lms = bboxes_fldr / "lms"
    generate_bboxes_from_lms_folder(lms, debug=False)
# %%
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
