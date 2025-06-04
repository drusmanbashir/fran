# %%
import shutil
from matplotlib import pyplot as plt
from torch import nn

from gudhi.cubical_complex import CubicalComplex
import cudf
import cugraph
from send2trash import send2trash

from utilz.helpers import info_from_filename
import torch
import SimpleITK as sitk
import re
from pathlib import Path

from label_analysis.helpers import get_labels
from utilz.fileio import maybe_makedirs
from utilz.imageviewers import ImageMaskViewer

    
if __name__ == '__main__':
# %%
    imgfn = "/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_plan2/images/nodes_56_41T410_CAP1p5SoftTissue.pt"    
    # lmfn = "/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_plan2/lms/nodes_24_410813_ChestAbdoC1p5SoftTissue.pt"    


    img = torch.load(imgfn,weights_only=False)
    img = img.permute(2,0,1)

    ImageMaskViewer([img,img],apply_transpose=True)
# %%
