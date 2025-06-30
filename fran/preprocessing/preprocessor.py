# %%
from pathlib import Path
import ipdb

tr = ipdb.set_trace

from fran.managers.project import DS
from fran.transforms.intensitytransforms import NormaliseClipd

import numpy as np
import pandas as pd
import torch
from fastcore.all import store_attr
from fastcore.foundation import GetAttr
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.spatial.dictionary import Spacingd
from monai.transforms.utility.dictionary import (
    EnsureChannelFirstd,
    FgBgToIndicesd,
    ToDeviced,
)

from fran.data.collate import dict_list_collated
from fran.preprocessing.dataset import ResamplerDataset
from fran.preprocessing import bboxes_function_version
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import ToCPUd
from fran.transforms.misc_transforms import (
    ChangeDtyped,
    DictToMeta,
    FgBgToIndicesd2,
    LabelRemapd,
    Recastd,
)
from fran.transforms.spatialtransforms import ResizeToTensord
from fran.utils.common import *
from utilz.fileio import load_dict, maybe_makedirs, save_dict, save_json
from utilz.helpers import create_df_from_folder, folder_name_from_list, multiprocess_multiarg, pbar
from utilz.string import info_from_filename, strip_extension


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
    tnsr = torch.load(filename,weights_only=False)
    return get_tensor_stats(tnsr)


def get_tensor_stats(tnsr):
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
        spacing,
        data_folder=None,
        output_folder=None,
    ) -> None:
        store_attr("project,spacing,data_folder")
        self.data_folder=data_folder
        self.set_input_output_folders(data_folder,output_folder)
        self.create_data_df()

    def create_data_df(self):
        if self.data_folder is not None:
            self.df = create_df_from_folder(self.data_folder)
            extract_ds = lambda x: DS.resolve_ds_name(x.split('_')[0])
            # self.df = pd.merge(self.df,self.project.df[['case_id','fold','ds']],how="left",on="case_id")
            self.df["ds"] = self.df["case_id"].apply(extract_ds)
            self.case_ids  = self.df['case_id'].tolist()

        else:
            self.df = self.project.df
            self.case_ids =self.project.case_ids
        print("Total number of cases: ", len(self.df))


    def set_input_output_folders(self,data_folder,output_folder): raise NotImplementedError
            
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
        torch.save(tnsr, fn)

    def register_existing_files(self):
        self.existing_files = list((self.output_folder / ("lms")).glob("*pt"))
        self.existing_case_ids = [
            info_from_filename(f.name, full_caseid=True)["case_id"]
            for f in self.existing_files
        ]
        self.existing_case_ids = set(self.existing_case_ids)
        print("Case ids processed in a previous session: ", len(self.existing_case_ids))
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

    def _store_dataset_properties(self):
        resampled_dataset_properties = self.create_info_dict()
        resampled_dataset_properties_fname = (
            self.output_folder / "resampled_dataset_properties.json"
        )
        print(
            "Writing preprocessing output properties to {}".format(
                resampled_dataset_properties_fname
            )
        )
        save_dict(resampled_dataset_properties, resampled_dataset_properties_fname)

    def create_info_dict(self):
        resampled_dataset_properties = dict()
        resampled_dataset_properties["dataset_spacing"] = self.spacing
        resampled_dataset_properties["dataset_max"] = self.results_df['max'].max().item()
        resampled_dataset_properties["dataset_min"] = self.results_df['min'].min().item()
        resampled_dataset_properties["dataset_median"] = np.median(self.results_df['median'])
        return resampled_dataset_properties

    def process(
        self,
    ):
        if not hasattr(self, "dl"):
            print("No data loader created. No data to be processed")
            return 0
        self.create_output_folders()
        self.results = []
        self.shapes = []

        for batch in pbar(self.dl):
            self.process_batch(batch)
        self.results_df = pd.DataFrame(self.results)
        # self.results= pd.DataFrame(self.results).values
        ts = self.results_df.shape
        if ts[-1] == 4:  # only store if entire dset is processed
            self._store_dataset_properties()
            generate_bboxes_from_lms_folder(self.output_folder / ("lms"))
        else:
            print("self.results  shape is {0}. Last element should be 4 , is {1}. therefore".format(ts,ts[-1]))
            print(
                "since some files skipped, dataset stats are not being stored. run self.get_tensor_folder_stats and generate_bboxes_from_lms_folder separately"
            )

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
        self.shapes.append(image.shape[1:])

    def get_tensor_folder_stats(self, debug=True):
        img_filenames = (self.output_folder / ("images")).glob("*")
        args = [[img_fn] for img_fn in img_filenames]
        results = multiprocess_multiarg(get_tensorfile_stats, args, debug=debug)
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
        resampled_dataset_properties["dataset_spacing"] = self.spacing
        resampled_dataset_properties["dataset_max"] = self.results_df["max"].max().item()
        resampled_dataset_properties["dataset_min"] = self.results_df["min"].min().item()
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

    @property
    def indices_subfolder(self):
        indices_subfolder = self.output_folder / ("indices")
        return indices_subfolder

# %%

if __name__ == '__main__':
    bboxes_fldr = Path('/s/fran_storage/datasets/preprocessed/fixed_spacing/totalseg/spc_150_150_150')
    lms = bboxes_fldr / 'lms'
    generate_bboxes_from_lms_folder(lms,debug=False)

# %%
