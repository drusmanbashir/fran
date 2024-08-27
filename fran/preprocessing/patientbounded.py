# %%
import time
import SimpleITK as sitk
import itertools as il
from SimpleITK.SimpleITK import LabelShapeStatisticsImageFilter
from label_analysis.helpers import get_labels, relabel, to_binary
from label_analysis.totalseg import TotalSegmenterLabels
from label_analysis.utils import ast, compress_img
from label_analysis.merge import LabelMapGeometry
from monai.data.meta_tensor import MetaTensor
from monai.transforms.utils import generate_spatial_bounding_box
import matplotlib.patches as patches
import torch.nn.functional as F
from pathlib import Path
from monai.transforms.intensity.array import NormalizeIntensity, ScaleIntensity
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.spatial.dictionary import Resized, Resize
from torchvision.datasets.folder import is_image_file

from fran.preprocessing.labelbounded import (
    LabelBoundedDataGenerator,
    LabelBoundedDataGeneratorImported,
)
from fran.transforms.imageio import LoadSITKd
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from fran.utils.config_parsers import is_excel_None
from fran.utils.fileio import is_sitk_file, load_dict
from fran.utils.helpers import find_matching_fn, folder_name_from_list
import ipdb

tr = ipdb.set_trace

from fran.utils.imageviewers import ImageMaskViewer, view_sitk
from fran.utils.string import info_from_filename
from monai.visualize import *
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


class PatientBoundedDataGenerator(LabelBoundedDataGeneratorImported):
    def __init__(
        self,
        project,
        expand_by,
        spacing,
        lm_group,
        imported_folder,
        imported_labelsets,
        remapping,
        merge_imported=False,
        output_suffix: str = None,
        fg_indices_exclude: list = None,
    ) -> None:
        super().__init__(
            project,
            expand_by,
            spacing,
            lm_group,
            imported_folder,
            imported_labelsets,
            remapping,
            merge_imported,
            output_suffix,
            fg_indices_exclude,
        )

    
    @property
    def output_folder(self):
        self._output_folder = folder_name_from_list(
            prefix="spc",
            parent_folder=self.pbd_folder,
            values_list=self.spacing,
        )
        if self.output_suffix:
            output_name = "_".join([self._output_folder.name , self.output_suffix])
            self._output_folder= Path(self._output_folder.parent / output_name)#.name = self.output_folder.name + self.output_suffix
        return self._output_folder




if __name__ == "__main__":
# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------

    from fran.utils.common import *

    P = Project(project_title="litsmc")
    spacing = [0.8, 0.8, 1.5]
    P.maybe_store_projectwide_properties()


# %%
    conf = ConfigMaker(
        P, raytune=False, configuration_filename=None
    ).config
# %%

    plan = conf['plan6']
    plan['spacing'] = ast.literal_eval(plan['spacing'])
    fg_indices_exclude=plan['fg_indices_exclude']
    plan_suffix = "plan6"


    imported_folder = Path("/s/fran_storage/predictions/totalseg/LITS-860")

    

    TSL = TotalSegmenterLabels()
    imported_labelsets = [TSL.labels("all")]
    TSL.labels("lung","right")
    remapping = TSL.create_remapping(imported_labelsets, [9,]*len(imported_labelsets))
    lm_group = plan['lm_groups']

# %%
    B = PatientBoundedDataGenerator(
        project=P,
        expand_by=plan['expand_by'],
        spacing=plan['spacing'],
        lm_group=plan['lm_groups'],
        imported_folder=imported_folder,
        imported_labelsets=imported_labelsets,
        merge_imported=False,
        remapping=remapping,
        output_suffix=plan_suffix,
        fg_indices_exclude=plan['fg_indices_exclude']
    )
# %%

    B.setup(overwrite=False)
    B.process()
# %% #SECTION:-------------------- ROUGH--------------------------------------------------------------------------------------
    conf = ConfigMaker(P, raytune=False, configuration_filename=None).config
    lm_fn = Path("/s/fran_storage/predictions/totalseg/LITS-860/lidc2_0009.nii.gz")
    img_fn = Path("/s/xnat_shadow/lidc2/images/lidc2_0009.nii.gz")

    dici = {"img": img_fn, "lm": lm_fn}
    L = LoadSITKd(keys=["img", "lm"])
    dici = L(dici)
    img = dici["img"]

    lm = dici["lm"]
    lm2 = lm.sum(0)
    lmv = torch.permute(lm2, (1, 0))
    lm3 = lm2.unsqueeze(0)

    add_to_bbox = 0
    shp = lm3.shape[1:]
    bb = generate_spatial_bounding_box(
        lm3, channel_indices=0, margin=add_to_bbox, allow_smaller=True
    )
    centres = [(b - a) / 2 for a, b in zip(*bb)]
    wh = [(b - a) for a, b in zip(*bb)]
    centres_normd = [a / b for a, b in zip(centres, shp)]
    wh_normd = [a / b for a, b in zip(wh, shp)]
    dat = [*centres_normd, *wh_normd]
    im2 = img.float().mean(0)

    fig, ax = plt.subplots()
    ax.imshow(lmv)
    rect = patches.Rectangle(
        bb[0],
        bb[1][0] - bb[0][0],
        bb[1][1] - bb[0][1],
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)

# %%
