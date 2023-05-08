
# %%
import SimpleITK as sitk



# %%
if __name__ == "__main__":
    from fran.utils.common import *
    img_fn = "/media/ub/datasets_bkp/litq/complete_cases/images/litq_0014389_20190925.nii"
    mask_fn = "/media/ub/datasets_bkp/litq/complete_cases/masks/litq_0014389_20190925.nii"
    pred_fn = "/s/fran_storage/predictions/lits/ensemble_LITS-265_LITS-255_LITS-270_LITS-271_LITS-272/litq_0014389_20190925.nii.gz"
    img_sitk = sitk.ReadImage(img_fn)
    mask_sitk = sitk.ReadImage(mask_fn)
    pred_sitk = sitk.ReadImage(pred_fn)

    P = Project(project_title="lits"); proj_defaults= P
# %%
    label_defaults = load_dict(proj_defaults.label_dict_filename)
    label=2

# %%
    overl = sitk.LabelOverlapMeasuresImageFilter()

