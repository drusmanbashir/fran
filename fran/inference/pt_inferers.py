# %%
from pathlib import Path

from monai.transforms.io.dictionary import LoadImaged, SaveImaged
from monai.utils.enums import LossReduction
from torch.nn.functional import one_hot
from utilz.cprint import cprint
from utilz.fileio import maybe_makedirs
from utilz.helpers import info_from_filename
from utilz.imageviewers import ImageMaskViewer

from fran.inference.base import BaseInferer, load_images_pt
from fran.inference.helpers import list_to_chunks
from fran.managers.wandb import WandbManager, download_path_no_wandb, download_wandb_checkpoint
from fran.transforms.imageio import TorchReader, TorchWriter


class BaseInfererPT(BaseInferer):
    def check_plan_compatibility(self):
        pass
    def load_images(self,images):
        return load_images_pt(images)

    def set_preprocess_tfms_keys(self):
        self.preprocess_tfms_keys = "E,N"  # No spacing done , put a channel dim then normalise

    def create_postprocess_transforms(self, preprocess_transform):
        super().create_postprocess_transforms(preprocess_transform)
        Sav = SaveImaged(
            keys=["pred"],
            output_ext="pt",
            writer=TorchWriter,
            output_dir=self.output_folder,
            output_postfix="",
            output_dtype="float32",
            separate_folder=False,
        )
        self.postprocess_transforms_dict["Sav"]     =Sav

    def set_postprocess_tfms_keys(self):
        if self.safe_mode == False:
            self.postprocess_tfms_keys = "Sq,A,Int"

        else:
            self.postprocess_tfms_keys = "Sq"
        self.postprocess_tfms_keys += ",Sq,CPU"  # additional key for this version
        if self.save_channels == True:
            self.postprocess_tfms_keys += ",SaM"
        if self.k_largest is not None:
            self.postprocess_tfms_keys += ",K"
        if self.save == True:
            self.postprocess_tfms_keys += ",Sav"

# %%
if __name__ == '__main__':

    save_channels = False
    safe_mode = False
    bs = 1
    overwrite = True
    devices = [0]
    save_channels = False
    run  = "KITS-n7"
    run = "KITS-TW"
    run  = "KITS2-bk"

    from fran.managers.project import Project
    p = Project("kits2")
    # download_wandb_checkpoint(p, run)
    # download_path_no_wandb(remote_dir_parent="/data/EECS-LITQ/fran_storage/checkpoints/kits2/kits2/KITS-TW/checkpoints",local_dir_parent="/s/fran_storage/checkpoints/kits2/kits2/KITS-TW/checkpoints")


    debug_ = False
#
    _, val = p.get_train_val_case_ids(1) 
    img_fldr = Path('/r/datasets/preprocessed/kits2/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/images')
    img_fns = list(img_fldr.glob("*.pt"))
    val_fns =[fn for fn in img_fns if info_from_filename(fn.name, full_caseid=True)["case_id"] in val]
    print(len(val_fns))



# %%
    T = BaseInfererPT(
        patch_overlap=0,
        run_name=run,
        save_channels=save_channels,
        safe_mode=safe_mode,
        devices=devices,
        debug=debug_,
    )

    # print(T.postprocess_tfms_keys)
# %%
#SECTION:-------------------- Loss func--------------------------------------------------------------------------------------
    from monai.losses import DiceLoss

    dice_reduction = (
        LossReduction.NONE
    )  # unreduced loss is outputted for logging and will be reduced manually
    loss_fn = DiceLoss(
        include_background=False,
        to_onehot_y=False,   # already one-hot
        softmax=True  ,       # applies softmax to logits,
        reduction=dice_reduction,
        
    )

    import torch
# %%
#SECTION:-------------------- RUN--------------------------------------------------------------------------------------

    imgs = val_fns[:4]
    chunksize=2
    devices = [0]
    T.setup()
    imgs = T.maybe_filter_images(imgs, overwrite)
    imgs = list_to_chunks(imgs, chunksize)
    imgs_sublist = imgs[0]

# %%
#SECTION:-------------------- process_imgs_sublist--------------------------------------------------------------------------------------
    data = T.load_images(imgs_sublist)
    T.prepare_data(data, collate_fn=None)
    T.create_and_set_postprocess_transforms()

    outputs = []
# %%
#SECTION:--------------------  predict--------------------------------------------------------------------------------------
    iteri = iter(T.predict())
# %%
    batch = next(iteri)
    batch_fn = batch["image"].meta["filename_or_obj"]
    print(batch_fn)
    batch.keys()
    batch2 = T.postprocess(batch)
        outputs.append(batch)
    n_classes = batch["pred"].shape[1]
# %%

    batch_fn = batch_fn.replace("images","lms")
    print(batch["pred"].shape)
    print(batch2["pred"].shape)


    pred = batch2["pred"]

# %%
    targ = torch.load(batch_fn, weights_only=False)
    targ = targ.long()
    print(targ.shape)
    ImageMaskViewer([pred,targ],'mm')
# %%
    targ_oh = one_hot(targ,n_classes)
    targ_oh = targ_oh.permute(3,0,1,2)
    targ_oh = targ_oh.unsqueeze(0)
    targ_oh.shape
# %%
    pred_raw = batch["pred"]
    pred_raw = pred_raw.to("cpu")
    
    loss = loss_fn(pred_raw, targ_oh)
    loss_unred= loss.flatten()
    print(loss_unred)
# %%
# %%
    output = T.process_imgs_sublist(imgs_sublist)
# %%
        overwrite=False
    preds = T.run(val_fns, chunksize=2, overwrite=overwrite)

# %%
    lbdkits_fldr = Path("")
    imf_fn = Path('/r/datasets/preprocessed/kits2/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex100/images/kits23_00018.pt')
    imf_fn = Path('/r/datasets/preprocessed/kits2/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex100/images/kits23_00053.pt')
    imf_fn = list(imf_fldr.glob("*.pt"))

# %%
    fldr = Path('/r/datasets/preprocessed/kits2/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex100/')
    output = fldr/("dataset_stats")/("output_gif.gif")
    maybe_makedirs(output.parent)
    from utilz.overlay_grid_gif import create_nifti_overlay_grid_gif

    create_nifti_overlay_grid_gif(dataset_root=fldr,output_gif=output, grid_shape =(3,3),num_frames=30,stride=4, window="abdomen",fps=5)
# %%
    img, pred = preds[0]['image'], preds[0]['pred']
   
    img.shape
    pred.shape

    ImageMaskViewer([img[0],pred[0]])
# %%
    pred = batch['pred']
    print(pred.shape)
    print(pred.max())
# %%
