# %%
from monai.losses import DiceLoss
from pathlib import Path

from monai.data.dataset import Dataset
from monai.transforms.io.dictionary import LoadImaged, SaveImaged
from monai.utils.enums import LossReduction
from torch.utils.data import DataLoader
from utilz.cprint import cprint
from utilz.helpers import find_matching_fn

from fran.inference.base import BaseInferer
from fran.inference.helpers import list_to_chunks, load_images_pt
from fran.managers.wandb import WandbManager, download_path_no_wandb, download_wandb_checkpoint
from fran.transforms.imageio import LoadImage, TorchReader, TorchWriter


class BaseInfererPT(BaseInferer):
    def __init__(self, run_name, patch_overlap: float, project_title=None, ckpt=None, state_dict=None, params=None, bs=8, mode="constant", devices=..., safe_mode=False, save_channels=False, save=True, k_largest=None, debug=False):
        super().__init__(run_name, patch_overlap, project_title, ckpt, state_dict, params, bs, mode, devices, safe_mode, save_channels, save, k_largest, debug)
        self.create_loss_func()

    def create_loss_func(self):
        dice_reduction = (
            LossReduction.NONE
        )  # unreduced loss is outputted for logging and will be reduced manually
        self.loss_func = DiceLoss(
            include_background=False,
            to_onehot_y=False,   # already one-hot
            softmax=True  ,       # applies softmax to logits,
            reduction=dice_reduction,
            
        )

    def check_plan_compatibility(self):
        pass

    def load_images(self,images):
        return load_images_pt(images)

    def load_images_and_gts(self,imgs_gt_sublist):
        loader = LoadImage( reader=TorchReader)
        img_gt_outimages = []
        for img,gt in imgs_gt_sublist:
            img = loader(img)
            if gt is not None:
                gt = loader(gt)
            else:
                gt = None
            img_gt_outimages.append((img,gt))
        return img_gt_outimages

    def set_preprocess_tfms_keys(self):
        self.preprocess_tfms_keys = "E,N"  # No spacing done , put a channel dim then normalise
 

    def run(self, imgs: list, gt_fldr: str|Path|None, chunksize=12, overwrite=False):
        """
        imgs can be a list comprising any of filenames, folder, or images (sitk or itk)
        chunksize is necessary in large lists to manage system ram
        """
        self.losses=[]
        self.setup()
        imgs = self.maybe_filter_images(imgs, overwrite)
        imgs = list_to_chunks(imgs, chunksize)
        losses_all = []
        for imgs_sublist in imgs:
            output = self.process_imgs_sublist(imgs_sublist, gt_fldr)
            losses_all.append(output)
        return output, losses_all

    def append_loss(self,batch):
        case_id = batch["case_id"][0]
        loss_dici = {"case_id": case_id}

        loss = batch["loss_dice_perlabel"]
        for ind in range(len(loss)):
            key = "loss_dice_label_"+str(ind+1)
            loss_val = loss[ind].item()
            loss_dici[key] = loss_val
        self.losses.append(loss_dici)

    def process_imgs_sublist(self, imgs_sublist, gt_fldr=None):
        imgs_gt_sublist=[]
        if gt_fldr is not None:
            for img_fn in imgs_sublist:
                gt_fn = find_matching_fn(img_fn, gt_fldr, ["all"])[0]
                imgs_gt_sublist.append((img_fn,gt_fn))
        else:
            imgs_gt_sublist = [(img_fn,None) for img_fn in imgs_sublist]
        data = self.load_images_and_gts(imgs_gt_sublist)
        self.prepare_data(data)
        self.create_and_set_postprocess_transforms()

        outputs = []
        for batch in self.predict():
            batch = self.compute_loss(batch)
            batch = self.postprocess(batch)
            outputs.append(batch)

        if self.safe_mode:
            self.reset()
            outputs.append(None)

        return outputs

    def compute_loss(self, batch):
            targ = batch.get("lm")
            if targ is not None:
                pred_raw = batch["pred"]
                pred_raw = pred_raw.to("cpu")
                n_classes = pred_raw.shape[1]
                targ = targ.long()
                targ_oh = one_hot(targ,n_classes)
                targ_oh2 = targ_oh.permute(0,4,1,2,3)
                # targ_oh2 = targ_oh2.unsqueeze(0)
                
                loss = self.loss_func(pred_raw, targ_oh2)
                loss_unred= loss.flatten()
                batch["loss_dice_perlabel"] = loss_unred
            else:
                batch["loss_dice_perlabel"] = None
            self.append_loss(batch)
            return batch



    def prepare_data(self, data, collate_fn=None):
        """
        data: list
        """

        image_gt_dicts = []
        for img_gt in data:
            img= img_gt[0]
            lm= img_gt[1]
            img_fn = img.meta["filename_or_obj"]
            img_fn_name = img_fn.split("/")[-1]
            case_id = info_from_filename(img_fn_name, full_caseid=True)["case_id"]
            image_gt_dicts.append({"image": img, "lm":lm, "case_id":case_id})
    
        nw, bs = 0, 1  # Slicer bugs out
        self.ds = Dataset(data=image_gt_dicts, transform=self.preprocess_compose)
        self.pred_dl = DataLoader(
            self.ds, num_workers=nw, batch_size=bs, collate_fn=collate_fn
        )
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
    from pathlib import Path
    from torch.nn.functional import one_hot
    from utilz.fileio import maybe_makedirs
    from utilz.helpers import info_from_filename
    from utilz.imageviewers import ImageMaskViewer
    import torch

    save_channels = False
    safe_mode = False
    bs = 1
    overwrite = True
    devices = [0]
    save_channels = False
    run  = "KITS-n7"
    run = "KITS-TW"
    run  = "KITS2-bah"

    from fran.managers.project import Project
    p = Project("kits2")
    # download_wandb_checkpoint(p, run)
    # download_path_no_wandb(remote_dir_parent="/data/EECS-LITQ/fran_storage/checkpoints/kits2/kits2/KITS-TW/checkpoints",local_dir_parent="/s/fran_storage/checkpoints/kits2/kits2/KITS-TW/checkpoints")


    debug_ = False
#
    _, val = p.get_train_val_case_ids(1) 
    img_fldr = Path('/r/datasets/preprocessed/kits2/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/images')
    gt_fldr = Path('/r/datasets/preprocessed/kits2/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/lms')
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
#SECTION:-------------------- RUN--------------------------------------------------------------------------------------

    imgs = val_fns
    chunksize=2
    devices = [0]
    T.setup()
    imgs = T.maybe_filter_images(imgs, overwrite)
    # imgs = list_to_chunks(imgs, chunksize)
    imgs_sublist = val_fns

    outs = T.run(imgs_sublist, gt_fldr,overwrite=True)
# %%
    losses_all = T.losses
    losses_df = pd.DataFrame(losses_all)

    fldr =  T.output_folder/("results")
    maybe_makedirs(fldr)
    losses_df.to_csv(fldr/"val_losses.csv")
# %%
#SECTION:-------------------- process_imgs_sublist--------------------------------------------------------------------------------------
    # data = T.load_images(imgs_sublist)
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


    pred = batch2["pred"]
# %%
    batch_fn = batch_fn.replace("images", "lms")
    targ = torch.load(batch_fn, weights_only=False)
    print(batch_fn)
    targ = targ.long()
# %%
    targ_oh = one_hot(targ,n_classes)
    targ_oh = targ_oh.permute(3,0,1,2)
    targ_oh = targ_oh.unsqueeze(0)
# %%
    pred_raw = batch["pred"]
    pred_raw = pred_raw.to("cpu")
    
    loss = loss_fn(pred_raw, targ_oh)
    loss_unred= loss.flatten()
    print(loss_unred)
# %%
    ImageMaskViewer([pred,targ],'mm')
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
