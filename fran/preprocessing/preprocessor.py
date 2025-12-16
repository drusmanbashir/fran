# %%
from pathlib import Path

import ipdb


tr = ipdb.set_trace

import numpy as np
import pandas as pd
import torch
from fastcore.all import store_attr
from fastcore.foundation import GetAttr
from utilz.fileio import maybe_makedirs, save_dict, save_json
from utilz.helpers import create_df_from_folder, multiprocess_multiarg
from utilz.string import info_from_filename, strip_extension

from tqdm.auto import tqdm
from fran.preprocessing import bboxes_function_version


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
            self.df = create_df_from_folder(self.data_folder)
            assert len(self.df) >0 , "No valid case files found in {}".format(self.data_folder)
            extract_ds = lambda x: x.split("_")[0]

            # self.df = pd.merge(self.df,self.project.df[['case_id','fold','ds']],how="left",on="case_id")
            self.df["ds"] = self.df["case_id"].apply(extract_ds)
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
        img_filenames = (self.output_folder / ("images")).glob("*")
        args = [[img_fn] for img_fn in img_filenames]
        results = multiprocess_multiarg(get_tensorfile_stats, args, debug=debug,io=True)
        self.shapes = [a["shape"] for a in results]
        self.results = pd.DataFrame(results)  # .values
        self.results = self.results[["max", "min", "median"]]
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
        self.shapes = np.array(self.shapes)
        resampled_dataset_properties = dict()
        resampled_dataset_properties["median_shape"] = np.median(
            self.shapes, 0
        ).tolist()
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
        "/s/fran_storage/datasets/preprocessed/fixed_spacing/totalseg/spc_150_150_150"
    )
    lms = bboxes_fldr / "lms"
    generate_bboxes_from_lms_folder(lms, debug=False)

# %
