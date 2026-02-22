# %%
from pathlib import Path
import sqlite3

import ipdb


tr = ipdb.set_trace

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from fastcore.all import store_attr
from fastcore.foundation import GetAttr
from utilz.fileio import maybe_makedirs, save_dict, save_json
from utilz.helpers import create_df_from_folder, multiprocess_multiarg
from utilz.stringz import info_from_filename, strip_extension

from tqdm.auto import tqdm
from fran.preprocessing import bboxes_function_version
from fran.utils.dataset_properties import analyze_tensor_data_folder


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


def _itk_binary_stats_like_fran(mask_arr: np.ndarray, separate_islands: bool = True) -> dict:
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
    centroids = np.vstack([bg_centroid] + [np.array(e[2], dtype=np.float64) for e in entries])

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
    label_files = sorted(list(masks_folder.glob("*.nii.gz")) + list(masks_folder.glob("*.nii")))
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


def get_tensor_stats(tnsr)->dict:
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
        store_attr("project,plan,data_folder")
        self.data_folder = data_folder
        self.set_input_output_folders(data_folder, output_folder)

    def create_data_df(self):
        if self.data_folder is not None:
            data_folder = Path(self.data_folder)
            raw_data_folder = Path(self.project.raw_data_folder)
            if data_folder == raw_data_folder and Path(self.project.db).exists():
                con = sqlite3.connect(str(self.project.db))
                try:
                    query = "SELECT case_id, image, lm, ds FROM datasources"
                    self.df = pd.read_sql_query(query, con)
                finally:
                    con.close()
                self.df["image"] = self.df["image"].apply(Path)
                self.df["lm"] = self.df["lm"].apply(Path)
            else:
                self.df = create_df_from_folder(self.data_folder)
                extract_ds = lambda x: x.split("_")[0]
                self.df["ds"] = self.df["case_id"].apply(extract_ds)
            assert len(self.df) >0 , "No valid case files found in {}".format(self.data_folder)
            self.case_ids = self.df["case_id"].tolist()

        else:
            self.df = self.project.df
            self.case_ids = self.project.case_ids
        print("Total number of cases: ", len(self.df))

    def set_input_output_folders(self, data_folder, output_folder):
        raise NotImplementedError

    def save_pt(self, tnsr, subfolder, contiguous=True, suffix: str = None):
        if contiguous == True:
            tnsr = tnsr.contiguous()
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
                fsinfo = f"Total={usage.total//(1024**3)}G, Used={usage.used//(1024**3)}G, Free={usage.free//(1024**3)}G"
            except Exception:
                fsinfo = "disk usage unavailable"

            print(f"[ERROR] Failed saving to {fn}")
            print(f"[ERROR] Filesystem info: {fsinfo}")

            raise RuntimeError(f"Quota exceeded at path: {fn}") from e


    def register_existing_files(self):
        existimg_lm_ids = self._get_existing_ids(self.output_folder / ("lms"))
        existing_img_ids = self._get_existing_ids(self.output_folder / ("images"))
        self.existing_case_ids = existing_img_ids.intersection(existimg_lm_ids)
        print("Output folder: ", self.output_folder)
        print("Case ids processed in a previous session: ", len(self.existing_case_ids))

    def _get_existing_ids(self,subfolder):
        existing_files = list(subfolder.glob("*pt"))
        existing_case_ids = [
            info_from_filename(f.name, full_caseid=True)["case_id"]
            for f in existing_files
        ]
        existing_case_ids = set(existing_case_ids)
        return existing_case_ids

    def remove_completed_cases(self):
        self.df = self.df[~self.df.case_id.isin(self.existing_case_ids)]

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

    #CODE: rename below to process_files  (see #9)
    def process(
        self,
    ):
        if not hasattr(self, "dl"):
            print("No data loader created. No data to be processed")
            return 0
        self.create_output_folders()
        self.results = []
        self.shapes = []
#CODE:  move away from dataloader and use multiprocessing  (see #7)
        for batch in pbar(self.dl): 
            self.process_batch(batch)
        self.results_df = pd.DataFrame(self.results)
        # self.results= pd.DataFrame(self.results).values
        ts = self.results_df.shape
        if ts[-1] == 4:  # only store if entire dset is processed
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

    def process_batch(self, batch):
        # U = ToCPUd(keys=["image", "lm", "lm_fg_indices", "lm_bg_indices"])
        # batch = U(batch)
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
            self.extract_image_props(image)

    def extract_image_props(self, image):
        self.results.append(get_tensor_stats(image))
        # self.shapes.append(image.shape[1:])

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

    def _store_dataset_properties(self):
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
        shapes_for_median = None
        if hasattr(self, "shapes") and len(self.shapes) > 0:
            shapes_for_median = np.array(self.shapes)
        elif (
            hasattr(self, "results_df")
            and "shape" in self.results_df.columns
            and self.results_df["shape"].notna().any()
        ):
            shapes_for_median = np.array(self.results_df["shape"].dropna().tolist())

        if shapes_for_median is not None and len(shapes_for_median) > 0:
            shapes_for_median = np.array(shapes_for_median)
            resampled_dataset_properties["min_shape"] = np.min(
                shapes_for_median, 0
            ).tolist()
            resampled_dataset_properties["median_shape"] = np.median(
                shapes_for_median, 0
            ).tolist()
            resampled_dataset_properties["max_shape"] = np.max(
                shapes_for_median, 0
            ).tolist()
        else:
            resampled_dataset_properties["min_shape"] = np.nan
            resampled_dataset_properties["median_shape"] = np.nan
            resampled_dataset_properties["max_shape"] = np.nan

        resampled_dataset_properties["dataset_spacing"] = self.plan.get('spacing')
        resampled_dataset_properties["dataset_max"] = (
            self.results_df["max"].max().item()
        )
        resampled_dataset_properties["dataset_min"] = (
            self.results_df["min"].min().item()
        )
        resampled_dataset_properties["dataset_median"] = np.median(
            self.results_df["median"]
        ).item()
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
            try:
                ray.init(ignore_reinit_error=True)
            except Exception as e:
                print("Ray init warning:", e)

    def ray_prepare(self, actor_cls, actor_kwargs: dict, num_processes: int):
        self.ray_init()
        n = max(1, min(len(self.df), int(num_processes))) if len(self.df) else 0
        self.n_actors = n
        self.mini_dfs = np.array_split(self.df, n) if n else []
        self.actors = [actor_cls.remote(**actor_kwargs) for _ in range(n)] if n else []

    def ray_run(self, actor_method: str = "process"):
        if not getattr(self, "actors", None):
            print("No actors created. Did you run ray_prepare()?")
            self.results_df = pd.DataFrame([])
            return self.results_df
        futs = [getattr(a, actor_method).remote(mdf) for a, mdf in zip(self.actors, self.mini_dfs)]
        results_lists = ray.get(futs)
        flat = list(il.chain.from_iterable(results_lists))
        self.results_df = pd.DataFrame(flat) if flat else pd.DataFrame([])
        return self.results_df
    # @property
    # def indices_subfolder(self):
    #     indices_subfolder = self.output_folder / ("indices")
    #     return indices_subfolder


# %%

if __name__ == "__main__":
    bboxes_fldr = Path(
        "/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_ric03e8a587_ex000"
    )
    bboxes_fldr = Path("/r/datasets/preprocessed/bones/fixed_spacing/spc_100_100_100")
    lms = bboxes_fldr / "lms"
    generate_bboxes_from_lms_folder(lms, debug=False)

# %
