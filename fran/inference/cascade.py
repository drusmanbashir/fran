# %%
from os.path import join
import time
import ipdb
from label_analysis.totalseg import TotalSegmenterLabels
from monai.transforms.utility.dictionary import SqueezeDimd
import itertools as il

from fran.trainers.base import checkpoint_from_model_id
from fran.transforms.misc_transforms import SelectLabels
from utilz.fileio import maybe_makedirs

tr = ipdb.set_trace

import sys
from pathlib import Path
import torch
import numpy as np
import SimpleITK as sitk
from monai.transforms.compose import Compose

# from monai.apps.detection.transforms.array import *
# from monai.data.box_utils import *
# from monai.inferers.merger import *
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.post.dictionary import (
    Activationsd,
    AsDiscreted,
    Invertd,
    MeanEnsembled,
)
from monai.transforms.spatial.dictionary import Resized

# from monai.transforms.utility.dictionary import AddChanneld, EnsureTyped

from fran.data.dataset import FillBBoxPatchesd
from fran.inference.base import (
    BaseInferer,
    get_patch_spacing,
    list_to_chunks,
    load_params,
)
from fran.transforms.inferencetransforms import (
    BBoxFromPTd,
    KeepLargestConnectedComponentWithMetad,
    RenameDictKeys,
    SaveMultiChanneld,
    ToCPUd,
)

# from utilz.itk_sitk import *

sys.path += ["/home/ub/code"]

# These are the usual ipython objects, including this one you are creating
ipython_vars = ["In", "Out", "exit", "quit", "get_ipython", "ipython_vars"]
import sys

from fastcore.foundation import listify


sys.path += ["/home/ub/Dropbox/code/fran/"]

from fastcore.basics import store_attr

from utilz.imageviewers import ImageMaskViewer


def img_bbox_collated(batch):
    imgs = []
    imgs_c = []
    bboxes = []
    for i, item in enumerate(batch):
        imgs.append(item["image"])
        # imgs_c.append(item["image_cropped"])
        bboxes.append(item["bounding_box"])
    output = {
        "image": torch.stack(imgs, 0),
        # "image_cropped": torch.stack(imgs_c, 0),
        "bounding_box": bboxes,
    }
    return output


class WholeImageInferer(BaseInferer):
    def __init__(self, run_name, devices=[1], save_channels=True, save=True, **kwargs):
        """
        Resizes image directly to patch_size and applies inference, one model run per image.
        """

        super().__init__(
            run_name=run_name,
            devices=devices,
            save_channels=save_channels,
            save=save,
            **kwargs
        )

        self.tfms = "ESN"

    def check_plan_compatibility(self):
        pass

    def create_transforms(self):

        super().create_transforms()
        self.S = Resized(
            keys=["image"], spatial_size=self.plan["patch_size"]
        )  # KEEP NAME AS S TO AVOID BUGS

    #
    #
    # def predict_inner(self,batch):
    #                 img_input = batch["image"]
    #                 img_input = img_input.cuda()
    #                 if "filename_or_obj" in img_input.meta.keys():
    #                     print("Processing: ", img_input.meta["filename_or_obj"])
    #                 output_tensor = self.model(img_input)
    #                 output_tensor = output_tensor[0]
    #                 batch["pred"] = output_tensor
    #                 batch["pred"].meta = batch["image"].meta.copy()
    #                 return batch
    #


class PatchInferer(BaseInferer):
    def __init__(
        self,
        run_name,
        patch_overlap=0.25,
        bs=1,
        grid_mode="gaussian",
        devices=[1],
        save_channels=True,
        **kwargs
    ):
        super().__init__(
            run_name=run_name,
            devices=devices,
            save_channels=save_channels,
            save=False,
            **kwargs
        )

    def check_plan_compatibility(self):
        pass

    def create_postprocess_transforms(self, preprocess_transform):
        Sq = SqueezeDimd(keys=["image", "pred"], dim=0)

        I = Invertd(
            keys=["pred"], transform=preprocess_transform, orig_keys=["image"]
        )  # watchout: use detach beforeharnd. make sure spacing are correct in preds
        U = ToCPUd(keys=["image", "pred"])
        Sa = SaveMultiChanneld(
            keys=["pred"],
            output_dir=self.output_folder,
            output_postfix="",
            separate_folder=False,
        )

        tfms = [Sq, I, U]
        if self.save_channels == True:
            tfms = [Sq, I, Sa, U]
        C = Compose(tfms)
        self.postprocess_transforms = C


class CascadeInferer(BaseInferer):  # SPACING HAS TO BE SAME IN PATCHES
    def __init__(
        self,
        run_name_w,
        runs_p,
        localiser_labels: list[str],  # these labels will be used to create bbox
        devices=[0],
        overwrite=True,
        safe_mode=False,
        profile=None,
        save_channels=False,
        save=True,
        save_localiser=True,
        k_largest=None,  # assign a number if there are organs involved
    ):
        """
        Creates a single dataset (cascade dataset) which normalises images once for both patches and whole images. Hence, the model should be developed from the same dataset std, mean values.
        """
        assert profile in [
            None,
            "dataloading",
            "prediction",
            "all",
        ], "Choose one of None , 'dataloading', 'prediction', 'all'"

        self.params = load_params(runs_p[0])
        # CODE:  change params to a different name more aligned and found else where in library
        self.Ps = [
            PatchInferer(
                run_name=run,
                devices=devices,
                save_channels=save_channels,
                safe_mode=safe_mode,
            )
            for run in runs_p
        ]
        self.predictions_folder = self.Ps[0].project.predictions_folder
        self.localiser_tfms = "ESN"
        WSInf = self.inferer_from_params(run_name_w)
        self.W = WSInf(
            run_name=run_name_w,
            save_channels=save_channels,
            devices=devices,
            safe_mode=safe_mode,
        )
        store_attr()

    def setup(self):
        pass

    def inferer_from_params(self, run_name_w):
        self.ckpt = checkpoint_from_model_id(run_name_w)
        dic1 = torch.load(self.ckpt, weights_only=False)
        mode = dic1["datamodule_hyper_parameters"]["config"]["plan"][
            "mode"
        ]  # ["dataset_params"]["mode"]
        if mode == "source":
            return BaseInferer
        elif mode == "whole":
            return WholeImageInferer

    def run(self, imgs: list, chunksize=12):
        """
        imgs can be a list comprising any of filenames, folder, or images (sitk or itk)
        chunksize is necessary in large lists to manage system ram
        """
        chunksize = np.minimum(len(imgs), chunksize)
        imgs = listify(imgs)
        if self.overwrite == False and (
            isinstance(imgs[0], str) or isinstance(imgs[0], Path)
        ):
            imgs = self.filter_existing_preds(imgs)
        else:
            pass
            # self.save = False  # don't save if input is pure images. Just output those.
        if len(imgs) > 0:
            imgs = list_to_chunks(imgs, chunksize)
            for imgs_sublist in imgs:
                output = self.process_imgs_sublist(imgs_sublist)
            return output
        else:
            return 1

    def process_imgs_sublist(self, imgs_sublist):
        data = self.load_images(imgs_sublist)
        self.bboxes = self.extract_fg_bboxes(data)
        data = self.apply_bboxes(data, self.bboxes)
        pred_patches = self.patch_prediction(data)
        pred_patches = self.decollate_patches(pred_patches, self.bboxes)
        output = self.postprocess(pred_patches)
        if self.save == True:
            self.save_pred(output)
        self.cuda_clear()
        return output

    def apply_bboxes(self, data, bboxes):
        data2 = []
        for i, dat in enumerate(data):
            dat["image"] = dat["image"][self.bboxes[i][1:]]
            dat["bounding_box"] = self.bboxes[i]
            data2.append(dat)
        return data2

    def filter_existing_preds(self, imgs):

        print(
            "Filtering existing predictions\nNumber of images provided: {}".format(
                len(imgs)
            )
        )
        out_fns = [self.output_folder / img.name for img in imgs]
        to_do = [not fn.exists() for fn in out_fns]
        imgs = list(il.compress(imgs, to_do))
        print("Number of images remaining to be predicted: {}".format(len(imgs)))
        return imgs

    def filter_existing_localisers(self, imgs):
        print(
            "Filtering existing localisers\nNumber of images provided: {}".format(
                len(imgs)
            )
        )
        new_W = []
        for P in self.Ps:
            out_fns = [P.output_folder / img.name for img in imgs]
            new_P = np.array([not fn.exists() for fn in out_fns])
            new_Ps.append(new_P)
        if len(P) > 1:
            new_Ps = np.logical_or(*new_Ps)
        else:
            new_Ps = new_Ps[0]
        imgs = list(il.compress(imgs, new_Ps))
        print("Number of images remaining to be predicted: {}".format(len(imgs)))
        return imgs

    def save_pred(self, preds):
        S = SaveImaged(
            keys=["pred"],
            output_dir=self.output_folder,
            output_postfix="",
            separate_folder=False,
        )
        for pp in preds:
            S(pp)

    def get_mini_bundle(self, patch_bundles, indx):
        patch_bundle = {}
        for key, val in patch_bundles.items():
            pred_patch = {key: val[indx]}
            patch_bundle.update(pred_patch)
        return patch_bundle

    def decollate_patches(self, pa, bboxes):
        num_cases = len(bboxes)
        keys = self.runs_p
        output = []
        for case_idx in range(num_cases):
            img_bbox_preds = {}
            for i, run_name in enumerate(keys):
                pred = pa[run_name][case_idx]["pred"]
                img_bbox_preds[run_name] = pred
            # img_bbox_preds.update(self.ds[case_idx])
            img_bbox_preds["bounding_box"] = bboxes[case_idx]
            output.append(img_bbox_preds)

        return output

    def cuda_clear(self):

        for p in self.Ps:
            del p.model
        torch.cuda.empty_cache()

    def extract_fg_bboxes(self, data):
        spacing = get_patch_spacing(self.run_name_w)
        Sel = SelectLabels(keys=["pred"], labels=self.localiser_labels)
        B = BBoxFromPTd(keys=["pred"], spacing=spacing, expand_by=10)
        if self.overwrite == False:
            print("Bbox overwrite not implemented yet")
        print("Starting localiser data prep and prediction")
        self.W.setup()
        self.W.prepare_data(data, tfms="ESN")
        p = self.W.predict()
        preds = self.W.postprocess(p)
        if self.save_localiser == True:
            self.W.save_pred(preds)
        bboxes = []
        for pred in preds:
            pred = Sel(pred)
            pred = B(pred)
            bb = pred["bounding_box"]
            # Check if bounding box is empty
            if bb is None or (isinstance(bb, (list, tuple)) and len(bb) == 0):
                raise ValueError(
                    "No bounding box found - localizer failed to detect region of interest"
                )
            bboxes.append(bb)
        return bboxes

    def patch_prediction(self, data):
        del self.W.model
        torch.cuda.empty_cache()
        preds_all_runs = {}
        print("Starting patch data prep and prediction")
        for P in self.Ps:
            P.setup()
            P.prepare_data(data=data, tfms="ESN", collate_fn=img_bbox_collated)
            preds = P.predict()
            preds = P.postprocess(preds)
            preds_all_runs[P.run_name] = preds
        return preds_all_runs

    @property
    def output_folder(self):
        fldr = "_".join(self.runs_p)
        fldr = self.predictions_folder / fldr
        return fldr

    def postprocess(self, preds):
        keys = self.runs_p
        A = Activationsd(keys="pred", softmax=True)
        D = AsDiscreted(keys=["pred"], argmax=True)
        F = FillBBoxPatchesd()
        if len(keys) == 1:
            MR = RenameDictKeys(new_keys=["pred"], keys=keys)
        else:
            MR = MeanEnsembled(output_key="pred", keys=keys)
        if self.k_largest:
            K = KeepLargestConnectedComponentWithMetad(
                keys=["pred"], independent=False, num_components=self.k_largest
            )  # label=1 is the organ
            tfms = [MR, A, D, K, F]
        else:
            tfms = [MR, A, D, F]
        # S = SaveListd(keys = ['pred'],output_dir=self.output_folder,output_postfix='',separate_folder=False)
        C = Compose(tfms)
        output = C(preds)
        return output


# %%

if __name__ == "__main__":
# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR>

    # ... run your application ...
    from fran.utils.common import *
    from fran.managers import Project

    run_w2 = "LIT-145"
    run_w = "LITS-1088"  # this run has localiser_labels not full TSL.

    run_lidc2 = ["LITS-902"]
    run_nodes = ["LITS-1110"]
    run_lidc2 = ["LITS-842"]
    run_lidc2 = ["LITS-913"]
    run_lidc2 = ["LITS-911"]
    run_litsmc = ["LITS-933"]
    run_litsmc2 = ["LITS-1018"]
    run_ts = ["LITS-827"]

    img_fna = "/s/xnat_shadow/litq/test/images_ub/"
    fns = "/s/datasets_bkp/drli_short/images/"
    img_fldr = Path("/s/xnat_shadow/lidc2/images/")
    img_fn2 = "/s/xnat_shadow/crc/wxh/images/crc_CRC198_20170718_CAP1p51.nii.gz"
    img_fn3 = "/s/xnat_shadow/crc/srn/images/crc_CRC002_20190415_CAP1p5.nii.gz"

    # fldr_crc = Path("/s/xnat_shadow/crc/images_train_rad/images/")
    fldr_crc = Path("/s/xnat_shadow/crc/images")
    # srn_fldr = "/s/xnat_shadow/crc/srn/cases_with_findings/images/"
    litq_fldr = "/s/xnat_shadow/litq/test/images_ub/"
    litq_imgs = list(Path(litq_fldr).glob("*"))
    t6_fldr = Path("/s/datasets_bkp/Task06Lung/images")
    imgs_t6 = list(t6_fldr.glob("*"))
    react_fldr = Path("/s/insync/react/sitk/images")
    imgs_react = list(react_fldr.glob("*"))
    imgs_crc = list(fldr_crc.glob("*"))
    nodesthick_fldr = Path("/s/xnat_shadow/nodesthick/images")
    nodes_fldr = Path("/s/xnat_shadow/nodes/images_pending")
    nodes = list(nodes_fldr.glob("*"))

    img_fns = [imgs_t6][:20]
    localiser_labels = [45, 46, 47, 48, 49]
    localiser_labels_litsmc = [1]
    TSL = TotalSegmenterLabels()
# %%
# %%
# SECTION:-------------------- LIDC-------------------------------------------------------------------------------------- <CR>

    loc_lidc = [7]  # lung in localiser_label

    safe_mode = False
    devices = [1]
    overwrite = True
    save_channels = False
    En = CascadeInferer(
        run_w,
        run_lidc2,
        save_channels=save_channels,
        devices=devices,
        overwrite=overwrite,
        localiser_labels=loc_lidc,
        safe_mode=safe_mode,
        k_largest=None,
    )

# %%
    imgs_tmp = ["/s/xnat_shadow/litq/test/images/litq_10.nii.gz"]
    preds = En.run(imgs_tmp, chunksize=1)

# %%
    model = En.Ps[0].model

# %%
# SECTION:-------------------- NODES -------------------------------------------------------------------------------------- <CR>
    localiser_labels = set(TSL.label_localiser)

    safe_mode = False
    devices = [1]
    overwrite = True
    save_channels = False
    save_localiser=True
    En = CascadeInferer(
        run_w,
        run_nodes,
        save_channels=save_channels,
        devices=devices,
        overwrite=overwrite,
        localiser_labels=localiser_labels,
        save_localiser=save_localiser,
        safe_mode=safe_mode,
        k_largest=None,
    )

# %%

    preds = En.run(nodes, chunksize=2)

# %%

    preds = En.W.postprocess(p)
    bboxes = []
# %%

# SECTION:-------------------- TOTALSEG WholeImageinferer-------------------------------------------------------------------------------------- <CR>

    safe_mode = False
    run_tot = ["LITS-1088"]
    W = WholeImageInferer(
        run_tot[0], safe_mode=safe_mode, k_largest=None, save_channels=False
    )
# %%

    preds = W.run(imgs_crc, chunksize=6)
    nodes_imgs = list(nodes_fldr.glob("*"))
    nodesthick_imgs = list(nodesthick_fldr.glob("*"))
    preds = W.run(nodesthick_imgs, chunksize=1)
    p = preds[0]["pred"][0]

# %%
# %%

# %%
# SECTION:---------------------------------------- LITSMC predictions-------------------------------------------------------------------- <CR>

    run = run_litsmc
    localiser_labels_litsmc = [3]
    run_w = "LITS-1088"
    devices = [1]
    overwrite = True
    safe_mode = True
    save_localiser = True
    save_channels = False
    project = Project(project_title="litsmc")
    if project.project_title == "litsmc":
        k_largest = 1
    else:
        k_largest = None
    En = CascadeInferer(
        run_w,
        run,
        save_channels=save_channels,
        devices=devices,
        overwrite=overwrite,
        localiser_labels=localiser_labels_litsmc,
        safe_mode=safe_mode,
        save_localiser=save_localiser,
        k_largest=k_largest,
    )

# %%

    img_fns = list(img_fldr.glob("*"))[20:50]
    case_id = "crc_CRC089"
    # imgs_crc = [fn for fn in imgs_crc if case_id in fn.name]
    tn = time.time()
    preds = En.run(imgs_crc[:30], chunksize=4)
    t2 = time.time()
    lapse = t2 - tn
# %%
    imgs_tmp = ["/s/xnat_shadow/litq/test/images/litq_10.nii.gz"]
    preds = En.run(imgs_tmp, chunksize=4)

# %%
    preds = En.W.postprocess(p)
    bboxes = []
# %%

# SECTION:-------------------- TROUBLESHOOTING En.run-------------------------------------------------------------------------------------- <CR>
# SECTION:-------------------- extract_fg_bboxes-------------------------------------------------------------------------------------- <CR>
# %%
    spacing = get_patch_spacing(En.run_name_w)
    Sel = SelectLabels(keys=["pred"], labels=list(En.localiser_labels))
    B = BBoxFromPTd(keys=["pred"], spacing=spacing, expand_by=10)
    if En.overwrite == False:
        print("Bbox overwrite not implemented yet")
    print("Starting localiser data prep and prediction")
# %%

    imgs_sublist = nodes[:2]
    data = En.load_images(imgs_sublist)
    En.W.setup()
    En.W.prepare_data(data, tfms="ERN")
    p = En.W.predict()
    preds = En.W.postprocess(p)
# %%
# %%
    preds[0]["pred"].shape
    ImageMaskViewer([preds[0]["pred"][0].detach(), preds[0]["image"][0].detach()])
    ImageMaskViewer([preds[1]["pred"][0].detach(), preds[1]["image"][0].detach()])
    bboxes = []
# %%
    for pred in preds:
        pred = Sel(pred)
        pred = B(pred)
        bb = pred["bounding_box"]
        bboxes.append(bb)

    print(bboxes)
# %%

    pred_patches = En.patch_prediction(data)
    pred_patches = En.decollate_patches(pred_patches, En.bboxes)
    output = En.postprocess(pred_patches)
    chunksize = 12
    imgs_sublist = imgs_crc
    imgs_sublist = listify(imgs)
    if En.overwrite == False and (
        isinstance(imgs[0], str) or isinstance(imgs[0], Path)
    ):
        imgs = En.filter_existing_preds(imgs)
    else:
        pass
        # En.save = False  # don't save if input is pure images. Just output those.
    if len(imgs) > 0:
        imgs = list_to_chunks(imgs, chunksize)
        for imgs_sublist in imgs:
            output = En.process_imgs_sublist(imgs_sublist)

# %%
# %%
# SECTION:-------------------- process_imgs_sublist-------------------------------------------------------------------------------------- <CR>

        imgs_sublist = imgs_tmp
        data = En.load_images(imgs_sublist)
        En.bboxes = En.extract_fg_bboxes(data)
        data = En.apply_bboxes(data, En.bboxes)
        data[0].keys()
        ImageMaskViewer([data[0]["image"], data[0]["image"]])
# %%
# SECTION:--------------------Patch predictor -------------------------------------------------------------------------------------- <CR>

        P = En.Ps[0]

        P.setup()
        P.prepare_data(data=data, tfms="ESN", collate_fn=img_bbox_collated)
        batch = next(iter(P.pred_dl))

# %%
        with torch.inference_mode():
            for i, batch in enumerate(P.pred_dl):
                with torch.no_grad():
                    img_input = batch["image"]
                    img_input = img_input.cuda()
                    if "filename_or_obj" in img_input.meta.keys():
                        print("Processing: ", img_input.meta["filename_or_obj"])
                    output_tensor = P.inferer(inputs=img_input, network=P.model)

                    #
# %%
        fn = img_input.meta["filename_or_obj"]
        fn_name = Path(fn).name
        fn_out = Path(fldr) / fn_name
        img_input.shape
        img_input2 = img_input[0, 0].cpu()
        img_input2 = torch.permute(img_input2, (2, 1, 0))
        img = sitk.GetImageFromArray(img_input2)
        sitk.WriteImage(img, str(fn_out))
        fldr = "preds"
        maybe_makedirs(fldr)
        ot = output_tensor
        imgs = [im.cpu()[0] for im in ot]

        [i.shape for i in imgs]
# %%
        for ind in range(len(imgs)):
            for ch, im in enumerate(imgs[ind]):
                fn_name_pred = fn_name + "_pred_{0}_ch{1}.nii.gz".format(ind, ch)
                im2 = torch.permute(im, (2, 1, 0))
                print(im.shape)
                pred_out = sitk.GetImageFromArray(im2)
                print(fn_name_pred)
                sitk.WriteImage(pred_out, Path(fldr) / fn_name_pred)
                print("\\n")

# %%
            img = imgs[ind]

        ImageMaskViewer([ot[0][0][2].cpu(), ot[2][0][2].cpu()])
# %%
        # output_tensor = output_tensor[0]
        # batch["pred"] = output_tensor
        # batch["pred"].meta = batch["image"].meta.copy()

        pred_patches = En.patch_prediction(data)
        pred_patches = En.decollate_patches(pred_patches, En.bboxes)
        output = En.postprocess(pred_patches)
        if En.save == True:
            En.save_pred(output)
        En.cuda_clear()


# %%
