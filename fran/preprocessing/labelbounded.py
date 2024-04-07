# %%
import cc3d
import torchio as tio
from label_analysis.totalseg import TotalSegmenterLabels
from monai.data import Dataset
from monai.transforms import Compose
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from torch.utils.data import DataLoader

from fran.data.dataloader import dict_list_collated
from fran.preprocessing.fixed_spacing import ResampleDatasetniftiToTorch, generate_bboxes_from_lms_folder, get_tensor_stats
from fran.preprocessing.patch import PatchDataGenerator
from fran.transforms.imageio import LoadSITKd, LoadTorchd
from fran.transforms.inferencetransforms import BBoxFromPTd
from fran.transforms.misc_transforms import (ApplyBBox, MergeLabelmapsd,
                                             Recast, RemapSITK)
from fran.transforms.spatialtransforms import ResizeToTensord

if "get_ipython" in globals():
    print("setting autoreload")
    from IPython import get_ipython

    ipython = get_ipython()
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
from pathlib import Path

import ipdb
import numpy as np
import SimpleITK as sitk
from fastcore.basics import GetAttr, store_attr

from fran.preprocessing.datasetanalyzers import bboxes_function_version
from fran.preprocessing.fixed_spacing import _Preprocessor
from fran.utils.fileio import *
from fran.utils.helpers import *
from fran.utils.imageviewers import *


class LabelBoundedDataGenerator(PatchDataGenerator,_Preprocessor, GetAttr):

    _default = "project"

    def __init__(
        self,
        project,
        expand_by,
        spacing,
        lm_group,
        imported_folder,
        imported_labelsets,
        remapping,
        keep_imported_labels=False,
    ) -> None:
        store_attr()
        self.case_ids = self.get_case_ids_lm_group(lm_group)
        configs = self.get_resampling_config(spacing)
        self.fixed_spacing_subfolder= Path(configs['resampling_output_folder'])
        self.output_folder = Path(configs['lbd_output_folder'])

    def get_resampling_config(self, spacing):
        resamping_config_fn = self.project.fixed_spacing_folder / ("resampling_configs.json")
        resampling_configs = load_dict(resamping_config_fn)
        for config in resampling_configs:
            if spacing == config["spacing"]:
               return config
        raise ValueError("No resampling config found for this spacing: {0}. \nAll configs are:\n{1}".format(spacing, resampling_configs))

    def create_dl(self, debug=False):
        self.register_existing_files()
        self.ds = ImporterDataset(
            case_ids=self.case_ids,
            expand_by=self.expand_by,
            spacing = self.spacing,
            data_folder=self.fixed_spacing_subfolder,
            imported_folder=self.imported_folder,
            remapping=self.remapping,
        )
        self.ds.setup()
        self.dl = DataLoader(
            dataset=self.ds,
            num_workers=4,
            collate_fn=dict_list_collated(
                keys=["image", "lm", "lm_imported", "bounding_box"]
            ),
            batch_size=4 if debug == False else 1,
        )

    def get_case_ids_lm_group(self, lm_group):
        dsrcs = self.global_properties[lm_group]["ds"]
        cids = []
        for dsrc in dsrcs:
            cds = self.df["case_id"][self.df["ds"] == dsrc].to_list()
            cids.extend(cds)
        return cids

 
    def create_properties_dict(self):
        resampled_dataset_properties = super().create_properties_dict()
        labels ={k[0]:k[1] for k in self.remapping.items() if self.remapping[k[0]] != 0}
        additional_props = {
            "imported_folder": str(self.imported_folder),
            "imported_labels":labels,
            "keep_imported_labels": self.keep_imported_labels,
        }
        return resampled_dataset_properties|additional_props



class ImporterDataset(Dataset):

    def __init__(
        self,
        expand_by,
        case_ids,
        data_folder,
        spacing,
        imported_folder,
        remapping,
    ):
        """
        imported_folder: Folder containing sitk Labelmaps
        keep_imported_labels: bool If True, imported labels are incorporated into the generated images and may be used in training.
        """
        store_attr("expand_by,spacing,case_ids,data_folder,imported_folder,remapping")

    def setup(self):
        self.create_transforms()
        data = self.create_data_dicts(self.imported_folder, self.remapping)
        Dataset.__init__(self, data=data, transform=self.transform)

    def create_data_dicts(self, imported_folder, remapping):
        masks_folder = self.data_folder / "lms"
        images_folder = self.data_folder / "images"
        lm_fns = list(masks_folder.glob("*.pt"))
        img_fns = list(images_folder.glob("*.pt"))
        imported_files = list(imported_folder.glob("*"))
        data = []
        for cid in self.case_ids:
            lm_fn = self.case_id_file_match(cid, lm_fns)
            img_fn = self.case_id_file_match(cid, img_fns)
            imported_fn = self.case_id_file_match(cid, imported_files)
            dici = {
                "lm": lm_fn,
                "image": img_fn,
                "lm_imported": imported_fn,
                "remapping": remapping,
            }
            data.append(dici)
        return data

    def case_id_file_match(self, case_id, fileslist):
        fns = [fn for fn in fileslist if case_id in fn.name]
        if len(fns) != 1:
            tr()
        return fns[0]

    def create_transforms(self):

        R = RemapSITK(keys=["lm_imported"], remapping_key="remapping")
        L1 = LoadSITKd(keys=["lm_imported"], image_only=True)
        L2 = LoadTorchd(keys=["lm", "image"])
        Re = Recast(keys=["lm_imported"])

        E = EnsureChannelFirstd(
            keys=["lm_imported", "image", "lm"], channel_dim="no_channel")
        
        Rz = ResizeToTensord(
            keys=["lm_imported"], key_template_tensor="lm", mode="nearest"
        )
        M = MergeLabelmapsd(keys=["lm_imported", "lm"], key_output="lm_out")
        B = BBoxFromPTd(
            keys=["lm_imported"], spacing=self.spacing, expand_by=self.expand_by
        )
        A = ApplyBBox(keys=["lm", "image", "lm_out"], bbox_key="bounding_box")
        tfms = [R, L1, L2, Re, E, Rz, M, B, A]
        C = Compose(tfms)
        self.transform = C


# %%
if __name__ == "__main__":

    from fran.utils.common import *

    P = Project(project_title="litsmc")
    P.maybe_store_projectwide_properties()
# %%
    lmg = "lm_group1"
    P.global_properties[lmg]

    imported_folder = Path("/s/fran_storage/predictions/totalseg/LITS-827/")
    TSL = TotalSegmenterLabels()
    imported_labelsets = TSL.labels("lung", "right"), TSL.labels("lung", "left")
    remapping = TSL.create_remapping(imported_labelsets, [8, 9])
    P.imported_labels(lmg, imported_folder, imported_labelsets)

# %%
    L = LabelBoundedDataGenerator(
        project=P,
        expand_by=20,
        spacing=[0.8, 0.8, 1.5],
        lm_group="lm_group1",
        imported_folder=imported_folder,
        imported_labelsets=imported_labelsets,
        keep_imported_labels=False,
        remapping=remapping,
    )

    L.create_dl()
    L.process()

# %%
    bbfn = "/home/ub/datasets/preprocessed/tmp/lbd/spc_080_080_150/bboxes_info.pkl"
    dic = load_dict(bbfn)
    generate_bboxes_from_lms_folder(Path("/r/datasets/preprocessed/lidc2/lbd/spc_080_080_150/lms/"))
# %%
