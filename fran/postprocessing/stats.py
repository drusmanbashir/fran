
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
    img_fn = "/media/ub/datasets_bkp/litq/complete_cases/images/litq_0014389_20190627.nii"
    mask_fn = "/s/fran_storage/predictions/lits/segmentations_LITS-265/litq_0014389_20190627.nii.gz"
    img_sitk = sitk.ReadImage(img_fn)
    img = ToTensorT()(img_fn)
    mask = ToTensorT()(mask_fn)
    ImageMaskViewer([img,mask])

    P = Project(project_title="lits"); proj_defaults= P.proj_summary
# %%
    label_defaults = proj_defaults.mask_labels
    label=2
    info = [l for l in label_defaults if l['label']==label][0]
    M = MaskToBinary(label=label,n_classes=3,merge_labels=[])
    mask_bin = M.encodes(mask)
# %%

    multilab , n= cc3d.largest_k(mask_bin,k=info['k_largest'],return_N=True)
    stats = cc3d.statistics(multilab)

    spacings = img_sitk.GetSpacing()
    voxvol = fl.reduce(operator.mul,spacings)  # mm3
    voxvol
    to_cc = 1e-3
    volumes =[voxvol*voxels*to_cc for voxels in stats['voxel_counts'][1:]] 
    ImageMaskViewer([img,mask_bin])


