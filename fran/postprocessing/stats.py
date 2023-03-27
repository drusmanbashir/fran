
# %%
import SimpleITK as sitk
import operator
import numpy as np
import cc3d
from fastai.callback.tracker import Transform, store_attr
from fran.transforms.totensor import ToTensorT
from fran.transforms.spatialtransforms import MaskLabelRemap
from fran.transforms.inferencetransforms import MaskToBinary

from fran.utils.imageviewers import ImageMaskViewer


# %%
if __name__ == "__main__":
    from fran.utils.common import *
    img_fn = "/media/ub/datasets_bkp/litq/complete_cases/images/litq_0014389_20190925.nii"
    mask_fn = "/media/ub/datasets_bkp/litq/complete_cases/masks/litq_0014389_20190925.nii"
    pred_fn = "/s/fran_storage/predictions/lits/ensemble_LITS-265_LITS-255_LITS-270_LITS-271_LITS-272/litq_0014389_20190925.nii.gz"
    img_sitk = sitk.ReadImage(img_fn)
    mask_sitk = sitk.ReadImage(mask_fn)
    pred_sitk = sitk.ReadImage(pred_fn)

    P = Project(project_title="lits"); proj_defaults= P.proj_summary
# %%
    label_defaults = proj_defaults.mask_labels
    label=2

# %%
    overl = sitk.LabelOverlapMeasuresImageFilter()

