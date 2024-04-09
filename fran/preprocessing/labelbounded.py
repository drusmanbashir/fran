# %%
import cc3d
import torchio as tio
from label_analysis.totalseg import TotalSegmenterLabels
from monai.data import Dataset
from monai.transforms import Compose
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToDeviceD, ToDeviced
from torch.utils.data import DataLoader

from fran.data.dataloader import dict_list_collated
from fran.preprocessing.dataset import CropToLabelDataset, ImporterDataset
from fran.preprocessing.fixed_spacing import (ResampleDatasetniftiToTorch,
                                              generate_bboxes_from_lms_folder,
                                              get_tensor_stats)
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


class LabelBoundedDataGenerator(PatchDataGenerator, _Preprocessor, GetAttr):
    _default = "project"

    def __init__(self, project, expand_by, spacing, lm_group, mask_label, fg_labels) -> None:
        store_attr()
        self.case_ids = self.get_case_ids_lm_group(lm_group)
        self.set_folders_from_spacing(spacing)
        print("Total case ids:", len(self.case_ids))

    def set_folders_from_spacing(self, spacing):
        self.fixed_spacing_subfolder = folder_name_from_list(
            prefix="spc",
            parent_folder=self.fixed_spacing_folder,
            values_list=spacing,
        )
        self.output_folder = folder_name_from_list(
            prefix="spc",
            parent_folder=self.lbd_folder,
            values_list=spacing,
        )

    def create_dl(self, device='cpu', batch_size = 4,debug=False,overwrite=True): # optimised defaults. Do not change. GPU wont work (multiprocessing issues)
        # 'gpu' wont work on multiprocessing
        device =resolve_device(device)
        num_workers = 1
        print("Processing on ",device)
        self.register_existing_files()
        print("Overwrite:",overwrite)
        if overwrite==False:
            self.case_ids = self.remove_completed_cases()

        self.ds = CropToLabelDataset(
            case_ids=self.case_ids,
            expand_by=self.expand_by,
            spacing=self.spacing,
            data_folder=self.fixed_spacing_subfolder,
            mask_label=self.mask_label,
            fg_labels =self.fg_labels,
            device=device
        )
        self.ds.setup()

        self.dl = DataLoader(
            dataset=self.ds,
            num_workers=num_workers,
            collate_fn=dict_list_collated(keys=["image", "lm"]),
            batch_size=batch_size,
        )

    def remove_completed_cases(self):
        case_ids = set(self.case_ids).difference(self.existing_case_ids)
        print("Remaining case ids to process:", len(case_ids))

        return case_ids


    def get_case_ids_lm_group(self, lm_group):
        dsrcs = self.global_properties[lm_group]["ds"]
        cids = []
        for dsrc in dsrcs:
            cds = self.df["case_id"][self.df["ds"] == dsrc].to_list()
            cids.extend(cds)
        return cids


#
#
#    def create_properties_dict(self):
#        resampled_dataset_properties = super().create_properties_dict()
#        labels ={k[0]:k[1] for k in self.remapping.items() if self.remapping[k[0]] != 0}
#        additional_props = {
#            "imported_folder": str(self.imported_folder),
#            "imported_labels":labels,
#            "keep_imported_labels": self.keep_imported_labels,
#        }
#        return resampled_dataset_properties|additional_props
#


class LabelBoundedDataGeneratorImported(PatchDataGenerator, _Preprocessor, GetAttr):

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
        self.fixed_spacing_subfolder = Path(configs["resampling_output_folder"])
        self.output_folder = Path(configs["lbd_output_folder"])

    def get_resampling_config(self, spacing):
        resamping_config_fn = self.project.fixed_spacing_folder / (
            "resampling_configs.json"
        )
        resampling_configs = load_dict(resamping_config_fn)
        for config in resampling_configs:
            if spacing == config["spacing"]:
                return config
        raise ValueError(
            "No resampling config found for this spacing: {0}. \nAll configs are:\n{1}".format(
                spacing, resampling_configs
            )
        )

    def create_dl(self, debug=False):
        self.register_existing_files()
        self.ds = ImporterDataset(
            case_ids=self.case_ids,
            expand_by=self.expand_by,
            spacing=self.spacing,
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
        labels = {
            k[0]: k[1] for k in self.remapping.items() if self.remapping[k[0]] != 0
        }
        additional_props = {
            "imported_folder": str(self.imported_folder),
            "imported_labels": labels,
            "keep_imported_labels": self.keep_imported_labels,
        }
        return resampled_dataset_properties | additional_props


# %%
if __name__ == "__main__":

    from fran.utils.common import *

    P = Project(project_title="litsmc")
    P.maybe_store_projectwide_properties()
# %%
    L = LabelBoundedDataGenerator(
        project=P,
        expand_by=40,
        spacing=[0.8, 0.8, 1.5],
        lm_group="lm_group1",
        mask_label=1,
    )

# %%
    L.create_dl(overwrite=False, device = 'cpu',batch_size=4)
    L.process()

# %%
# %%

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
        expand_by=40,
        spacing=[0.8, 0.8, 1.5],
        lm_group="lm_group1",
        mask_label=1,
    )

    L.create_dl()
    L.process()

# %%

    for  batch in pbar(L.dl):
        images, lms = batch["image"], batch["lm"]
        print(images.shape)
    ImageMaskViewer([images[0][0], lms[0][0]])
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
    generate_bboxes_from_lms_folder(
        Path("/r/datasets/preprocessed/lidc2/lbd/spc_080_080_150/lms/")
    )
# %%
    spacing = [0.8, 0.8, 1.5]
# %%
    L.expand_by = 50
    L.device='cpu'

    L2 = LoadTorchd(keys=["lm", "image"])
    # En = EnsureTyped(keys = ["lm","image"])
    D = ToDeviced(device =L.device,keys=["lm","image"])

    E = EnsureChannelFirstd(
        keys=[ "image", "lm"], channel_dim="no_channel")
    
    margin= [int(L.expand_by / sp) for sp in L.spacing]
    margin2= [int(L.expand_by / sp)*2 for sp in L.spacing]
# %%
    Cr1 = CropForegroundd(keys = ["image","lm"], source_key = "lm", select_fn = lambda lm: lm==L.mask_label,margin = margin)
    Cr2 = CropForegroundd(keys = ["image","lm"], source_key = "lm", select_fn = lambda lm: lm==L.mask_label,margin = margin2)
    tfms = [L2,D,E,Cr1]
    C1 = Compose(tfms)

    tfms2= [L2,D,E,Cr2]
    C2 = Compose(tfms)
# %%
    dici = L.ds.data[1].copy()
    dici1 = C1(dici)
    img1 = dici1['image'][0]
    lm1 = dici1['lm'][0]
    ImageMaskViewer([img1,lm1])
# %%

    dici2 = L.ds.data[1].copy()
    dici2 = C2(dici2)
    img2 = dici2['image'][0]
    lm2 = dici2['lm'][0]
    ImageMaskViewer([img2,lm2])
# %%

