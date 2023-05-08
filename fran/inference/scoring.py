# %matplotlib inline
# %matplotlib widget
from monai.utils.enums import LossReduction
from fastai.callback.tracker import torch

from monai.losses import DiceLoss
# %%

from fran.utils.common import *
from fran.utils.helpers import get_case_id_from_filename, get_fold_case_ids
from fran.utils.imageviewers import ImageMaskViewer
from fastai.vision.augment import typedispatch
from fran.utils.common import *
from fran.transforms.totensor import ToTensorT
from fran.transforms.spatialtransforms import one_hot
import SimpleITK as sitk
from monai.metrics import *

# %%



@typedispatch
def img_shape(x:sitk.Image):
   return x.GetSize()

@typedispatch
def img_shape(x:torch.Tensor):
   return x.shape


def compute_dice_fran(pred,mask,n_classes):
    mask_pt = ToTensorT.encodes(mask)
    pred_pt = ToTensorT.encodes(pred)
    pred_onehot,mask_onehot = [one_hot(x,classes=n_classes,axis=0).unsqueeze(0) for x in [pred_pt,mask_pt]]
    aa = compute_dice(pred_onehot,mask_onehot,include_background=False)
    return aa


# %%
if __name__ == "__main__":
    P = Project(project_title="lits"); proj_defaults= P

    # %%
    configs_excel = ConfigMaker(proj_defaults.configuration_filename,raytune=False).config
    train_list, valid_list, test_list = get_fold_case_ids(
            fold=configs_excel['metadata']["fold"],
            json_fname=proj_defaults.validation_folds_filename,
        )



    # %%
    mask_files = list((proj_defaults.raw_data_folder/("masks")).glob("*nii*"))
    img_files= list((proj_defaults.raw_data_folder/("images")).glob("*nii*"))
    masks_valid = [filename for filename in mask_files if  get_case_id_from_filename(None, filename) in valid_list]
    masks_train = [filename for filename in mask_files if  get_case_id_from_filename(None, filename) in train_list]
    imgs_valid =  [proj_defaults.raw_data_folder/"images"/mask_file.name for mask_file in masks_valid]
    imgs_test =  [filename for filename in img_files if  get_case_id_from_filename(None, filename) in test_list]
    imgs_train =  [filename for filename in img_files if  get_case_id_from_filename(None, filename) in train_list]
    # %%
    run_name = "LITS-122"
    preds_folder = list(proj_defaults.predictions_folder.glob(f"*{run_name}"))[0]
    pred_fns = list(preds_folder.glob("*"))
    pred_fn = pred_fns[0]
    case_id = get_case_id_from_filename(None,pred_fn)
# %%
    
    mask_fn = [fn for fn in masks_train if 'lits-128' in str(fn)][0]
    img_fn = [fn for fn in imgs_train if 'lits-128' in str(fn)][0]
    pred_fn = [fn for fn in pred_fns if 'lits-128' in str(fn)][0]
    img = sitk.ReadImage(img_fn)
    img_np= sitk.GetArrayFromImage(img)
    img_pt = torch.tensor(img_np)
    # %%
    pred= sitk.ReadImage(pred_fn)
    pred_np = sitk.GetArrayFromImage(pred)
    # %%
    mask = sitk.ReadImage(mask_fn)

# %% [markdown]
## Dice score for 3 classes, BG, liver, tumour
# %%
    n_classes = 3

    mask_pt = ToTensorT().encodes(mask).unsqueeze(0)
    pred_pt = ToTensorT().encodes(pred).unsqueeze(0)
    pred_pt_ub = one_hot(pred_pt,classes=2,axis=1)

# %%
    D = DiceLoss()
# %%
    import time
    st = time.time()
    for i in range(10):
        l = D(pred_pt_ub,mask_pt)
    eno = time.time()
    lapse= eno-st
    print(lapse)
# %%
    pred_onehot,mask_onehot = [one_hot(x,classes=n_classes,axis=1) for x in [pred_pt,mask_pt]]
    D2 = DiceLoss(include_background=False,to_onehot_y=True,batch=True,reduction= LossReduction.NONE)
    l2 = D2(input=pred_onehot,target=mask_pt)
    pp(l2)
# %%
    st = time.time()
    for i in range(10):
        mask_onehot = one_hot(mask_pt,classes=n_classes,axis=1)
        l2 = D2(pred_onehot,mask_onehot)
    eno = time.time()
    lapse2= eno-st
    print(lapse2)

# %%
    aa = compute_dice(pred_onehot,mask_onehot)


# %%
    ImageMaskViewer([pred_pt,mask_pt],data_types=['mask','mask'])

# %%
