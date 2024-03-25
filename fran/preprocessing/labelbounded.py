# %%
import cc3d
import torchio as tio
from label_analysis.totalseg import TotalSegmenterLabels
from monai.data import Dataset
from monai.transforms import Compose
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from torch.utils.data import DataLoader

from fran.data.dataloader import dict_list_collated
from fran.preprocessing.fixed_spacing import generate_bboxes_from_masks_folder
from fran.preprocessing.patch import PatchGenerator
from fran.transforms.imageio import LoadSITKd, LoadTorchd
from fran.transforms.inferencetransforms import BBoxFromPTd
from fran.transforms.misc_transforms import (ApplyBBox, MergeLabelmapsd,
                                             Recast, RemapSITK)
from fran.transforms.spatialtransforms import ResizeDynamicd

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
from fran.utils.fileio import *
from fran.utils.helpers import *
from fran.utils.imageviewers import *


class LabelBoundedDataGenerator(PatchGenerator, GetAttr):

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
        self.fixed_spacing_folder = self.get_resampled_folder(spacing)
        self.output_folder = self.fixed_spacing_folder

    def get_resampled_folder(self, spacing):
        resamping_config_fn = self.fixed_spacing_folder / ("resampling_configs.json")
        resampling_configs = load_dict(resamping_config_fn)
        for config in resampling_configs:
            if spacing == config["spacing"]:
                fldr = config["resampling_output_folder"]
        spacings = [a["spacing"] for a in resampling_configs]
        assert (
            self.spacing in spacings
        ), f"{spacings} are all available spacings. Choose one of these, not {self.spacing}"
        return Path(fldr)

    @property
    def output_folder(self):
        return self._output_folder

    @output_folder.setter
    def output_folder(self, fixed_spacing_folder):
        self._output_folder = self.lbd_folder / fixed_spacing_folder.name

    def create_dl(self, debug=False):
        self.register_existing_files()
        self.ds = ImporterDataset(
            case_ids=self.case_ids,
            expand_by=self.expand_by,
            fixed_spacing_folder=self.fixed_spacing_folder,
            imported_folder=self.imported_folder,
            remapping=self.remapping,
        )
        self.ds.setup()
        self.dl = DataLoader(
            dataset=self.ds,
            num_workers=4,
            collate_fn=dict_list_collated(
                keys=["image", "mask", "mask_imported", "bounding_box"]
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

    def process(self):
        self.create_output_folders()
        for i, item in enumerate(self.dl):
            self.shapes =[]
            images = item["image"]
            if self.keep_imported_labels == False:
                masks = item["mask"]
            else:
                masks = item["mask_out"]
            for image, mask in zip(images, masks):
                self.save_pt(image[0], "images")
                self.save_pt(mask[0], "masks")
                self.shapes.append(image.shape[1:])
        self.store_info()

    def save_pt(self, tnsr, subfolder):
        fn = Path(tnsr.meta["filename"])
        fn = self.output_folder / subfolder / fn.name
        torch.save(tnsr, fn)

    def store_info(self):
        generate_bboxes_from_masks_folder(self.output_folder/("masks"))
        self.shapes = np.array(self.shapes)
        fn_dict = self.output_folder / "info.json"
        labels ={k[0]:k[1] for k in self.remapping.items() if self.remapping[k[0]] != 0}

        dici = {
            "median_shape":np.median(self.shapes,0).tolist(),
            "imported_folder": str(self.imported_folder),
            "imported_labels":labels,
            "keep_imported_labels": self.keep_imported_labels,
            "spacing": self.spacing,
        }
        save_dict(dici,fn_dict)



class ImporterDataset(Dataset):

    def __init__(
        self,
        expand_by,
        case_ids,
        fixed_spacing_folder,
        imported_folder,
        remapping,
    ):
        """
        imported_folder: Folder containing sitk Labelmaps
        keep_imported_labels: bool If True, imported labels are incorporated into the generated images and may be used in training.
        """
        self.spacing = spacing_from_folder_name(
            prefix="spc", folder_name=fixed_spacing_folder
        )
        store_attr("expand_by,case_ids,fixed_spacing_folder,imported_folder,remapping")

    def setup(self):
        self.create_transforms()
        data = self.create_data_dicts(self.imported_folder, self.remapping)
        Dataset.__init__(self, data=data, transform=self.transform)

    def create_data_dicts(self, imported_folder, remapping):
        masks_folder = self.fixed_spacing_folder / "masks"
        images_folder = self.fixed_spacing_folder / "images"
        lm_fns = list(masks_folder.glob("*.pt"))
        img_fns = list(images_folder.glob("*.pt"))
        imported_files = list(imported_folder.glob("*"))
        data = []
        for cid in self.case_ids:
            lm_fn = self.case_id_file_match(cid, lm_fns)
            img_fn = self.case_id_file_match(cid, img_fns)
            imported_fn = self.case_id_file_match(cid, imported_files)
            dici = {
                "mask": lm_fn,
                "image": img_fn,
                "mask_imported": imported_fn,
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

        R = RemapSITK(keys=["mask_imported"], remapping_key="remapping")
        L1 = LoadSITKd(keys=["mask_imported"], image_only=True)
        L2 = LoadTorchd(keys=["mask", "image"])
        Re = Recast(keys=["mask_imported"])

        E = EnsureChannelFirstd(
            keys=["mask_imported", "image", "mask"], channel_dim="no_channel"
        )  # funny shape output mismatch
        Rz = ResizeDynamicd(
            keys=["mask_imported"], key_spatial_size="mask", mode="nearest"
        )
        M = MergeLabelmapsd(keys=["mask_imported", "mask"], key_output="mask_out")
        B = BBoxFromPTd(
            keys=["mask_imported"], spacings=self.spacing, expand_by=self.expand_by
        )
        A = ApplyBBox(keys=["mask", "image", "mask_out"], bbox_key="bounding_box")
        tfms = [R, L1, L2, Re, E, Rz, M, B, A]
        C = Compose(tfms)
        self.transform = C


# %%
if __name__ == "__main__":

    from fran.utils.common import *

    P = Project(project_title="lidc2")
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
# %%
