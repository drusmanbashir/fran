# %%
import itertools as il
import time

import ipdb
import SimpleITK as sitk
from utilz.cprint import cprint
from label_analysis.totalseg import TotalSegmenterLabels
from monai.transforms.utility.dictionary import CastToTyped

from fran.trainers.base import checkpoint_from_model_id
from fran.transforms.misc_transforms import SelectLabels
from fran.utils.misc import parse_devices

tr = ipdb.set_trace

import sys
from pathlib import Path

import numpy as np
import torch
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.post.dictionary import AsDiscreted, Invertd
from monai.transforms.spatial.dictionary import Resized

from fran.data.dataset import FillBBoxPatchesd
from fran.inference.base import (BaseInferer, get_patch_spacing,
                                 list_to_chunks, load_images, load_params)
from fran.transforms.inferencetransforms import (
    BBoxFromPTd, KeepLargestConnectedComponentWithMetad, MakeWritabled,
    RenameDictKeys)

# from monai.transforms.utility.dictionary import AddChanneld, EnsureTyped


# from utilz.itk_sitk import *

sys.path += ["/home/ub/code"]

# These are the usual ipython objects, including this one you are creating
ipython_vars = ["In", "Out", "exit", "quit", "get_ipython", "ipython_vars"]
import sys

from fastcore.foundation import listify

sys.path += ["/home/ub/Dropbox/code/fran/"]

from fastcore.basics import store_attr
from utilz.imageviewers import ImageMaskViewer


def apply_bboxes(data, bboxes):
    data2 = []
    for i, dat in enumerate(data):
        dat["image"] = dat["image"][bboxes[i][1:]]
        dat["bounding_box"] = bboxes[i]
        data2.append(dat)
    return data2


def img_bbox_collated(batch):
    imgs = []
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


def validate_bbox(box):
    for j, s in enumerate(box):
        if isinstance(s, slice) and s.start == 0 and s.stop == 0:
            raise ValueError(f"Invalid bbox at index {i}, slice {j}: {s}")
    return box  # usage


class WholeImageInferer(BaseInferer):
    def __init__(
        self,
        run_name,
        project_title=None,
        devices=[1],
        save_channels=True,
        save=True,
        patch_overlap=0.0,
        **kwargs,
    ):
        """
        Resizes image directly to patch_size and applies inference, one model run per image.
        """
        cprint("Setting up whole image inference", color="red", bold=True)

        super().__init__(
            run_name=run_name,
            project_title=project_title,
            devices=devices,
            save_channels=save_channels,
            save=save,
            patch_overlap=patch_overlap,
            **kwargs,
        )


    def set_preprocess_tfms_keys(self):
        self.preprocess_tfms_keys = "E,ResW,N"

    def check_plan_compatibility(self):
        pass

    def create_preprocess_transforms(self):
        super().create_preprocess_transforms()
        self.preprocess_transforms_dict["ResW"] = Resized(
            keys=["image"], spatial_size=self.plan["patch_size"]
        )  # KEEP NAME AS S TO AVOID BUGS

    def create_postprocess_transforms(self, preprocess_transform):
        super().create_postprocess_transforms(preprocess_transform)
        tfms = [
            self.postprocess_transforms_dict["Sq"],
            self.postprocess_transforms_dict["Re"],
        ]
        C = Compose(tfms)
        self.postprocess_compose = C

    def set_postprocess_tfms_keys(self):
        self.postprocess_tfms_keys = "Sq,Re"
        if self.save == True:
            self.postprocess_tfms_keys += ",Sav"


class PatchInferer(BaseInferer):
    def __init__(
        self,
        run_name,
        project_title=None,
        patch_overlap=0.25,
        bs=1,
        grid_mode="gaussian",
        devices=[1],
        save_channels=True,
        safe_mode=False,
        save=False,
        params=None,
        debug=False,
        **kwargs,
    ):

        cprint("Setting up Patch inference", color="red", bold=True)
        super().__init__(
            run_name=run_name,
            project_title=project_title,
            devices=devices,
            save_channels=save_channels,
            save=save,
            safe_mode=safe_mode,
            patch_overlap=patch_overlap,
            params=params,
            debug=debug,
            **kwargs,
        )

    def check_plan_compatibility(self):
        pass

    def create_postprocess_transforms(self, preprocess_transform):
        super().create_postprocess_transforms(preprocess_transform)
        InvP = Invertd(
            keys=["pred"], transform=preprocess_transform, orig_keys=["image"]
        )  # watchout: use detach beforeharnd. make sure spacing are correct in pred
        self.postprocess_transforms_dict["InvP"] = InvP

    def set_postprocess_tfms_keys(self):
        self.postprocess_tfms_keys = "Sq,SqL,InvP"
        if self.safe_mode == True:
            self.postprocess_tfms_keys += ",CPU"
        if self.save_channels == True:
            self.postprocess_tfms_keys += ",Sa"
        if self.k_largest is not None:
            self.postprocess_tfms_keys += ",K"
        if self.save == True:
            self.postprocess_tfms_keys += ",Sav"


class CascadeInferer(BaseInferer):  # SPACING HAS TO BE SAME IN PATCHES
    def __init__(
        self,
        run_w,
        run_p,
        localiser_labels: list[str],  # these labels will be used to create bbox
        project_title=None,
        devices=[0],
        safe_mode=False,
        patch_overlap=0.2,
        profile=None,
        save_channels=False,
        save=True,
        save_localiser=True,
        k_largest=None,  # assign a number if there are organs involved
        debug=False,
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

        self.device = parse_devices(devices)
        self.params = load_params(run_p)
        self.P = PatchInferer(
            run_name=run_p,
            project_title=project_title,
            devices=devices,
            patch_overlap=patch_overlap,
            save_channels=save_channels,
            safe_mode=safe_mode,
            params=self.params,
            debug=debug,
        )

        self.predictions_folder = self.P.project.predictions_folder
        WSInf = self.inferer_from_params(run_w)
        self.W = WSInf(
            run_name=run_w,
            save_channels=save_channels,
            devices=devices,
            safe_mode=True,  # no need to get single channels. Do NOT CHANGE THIS
            save=save_localiser,
            debug=debug,
        )

        store_attr()

    def setup(self):
        pass

    def inferer_from_params(self, run_w):
        self.ckpt = checkpoint_from_model_id(run_w)
        dic1 = torch.load(self.ckpt, weights_only=False)
        mode = dic1["datamodule_hyper_parameters"]["configs"]["plan_train"][
            "mode"
        ]  # ["dataset_params"]["mode"]
        if mode == "source":
            return BaseInferer
        elif mode == "whole":
            return WholeImageInferer

    def create_and_set_postprocess_transforms(self):
        self.create_postprocess_transforms()
        self.set_postprocess_tfms_keys()
        self.set_postprocess_transforms()

    def process_imgs_sublist(self, imgs_sublist):
        self.create_and_set_postprocess_transforms()
        data = load_images(imgs_sublist)

        self.bboxes = self.extract_fg_bboxes(data)

        data = apply_bboxes(data, self.bboxes)
        pred_patches = self.patch_prediction(data)
        pred_patches = self.decollate_patches(pred_patches, self.bboxes)
        output = self.postprocess(pred_patches)
        # if self.save == True:
        #     self.save_pred(output)
        self.cuda_clear()
        return output

    def filter_existing_localisers(self, imgs):
        print(
            "Filtering existing localisers\nNumber of images provided: {}".format(
                len(imgs)
            )
        )
        for P in self.P:
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

    def get_mini_bundle(self, patch_bundles, indx):
        patch_bundle = {}
        for key, val in patch_bundles.items():
            pred_patch = {key: val[indx]}
            patch_bundle.update(pred_patch)
        return patch_bundle

    def decollate_patches(self, pa, bboxes):
        num_cases = len(bboxes)
        keys = listify(self.run_p)
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

        for p in self.P:
            del p.model
        torch.cuda.empty_cache()

    def extract_fg_bboxes(self, data):
        spacing = get_patch_spacing(self.run_w)
        Sel = SelectLabels(keys=["pred"], labels=self.localiser_labels)
        B = BBoxFromPTd(keys=["pred"], spacing=spacing, expand_by=10)
        print("Starting localiser data prep and prediction")
        self.W.setup()
        self.W.prepare_data(data)
        self.W.create_and_set_postprocess_transforms()
        bboxes = []
        for batch in self.W.predict():
            if self.debug == True:
                tr()
                print(batch["pred"].shape)
            # p = self.W.predict()
            pred = self.W.postprocess(batch)
            pred = Sel(pred)
            pred = B(pred)
            bb = pred["bounding_box"]
            # Check if bounding box is empty
            if bb is None or (isinstance(bb, (list, tuple)) and len(bb) == 0):
                raise ValueError(
                    "No bounding box found - localizer failed to detect region of interest"
                )
            bb = validate_bbox(bb)
            bboxes.append(bb)
        return bboxes

    def patch_prediction(self, data):
        if hasattr(self.W, "model"):
            del self.W.model
        torch.cuda.empty_cache()
        print("Starting patch data prep and prediction")
        preds_all_runs = {}
        preds_all_runs[self.P.run_name] = []
        self.P.setup()
        self.P.prepare_data(data=data, collate_fn=img_bbox_collated)
        self.P.create_and_set_postprocess_transforms()
        for batch in self.P.predict():
            batch = self.P.postprocess(batch)
            preds_all_runs[self.P.run_name].append(batch)
        return preds_all_runs

    @property
    def output_folder(self):
        # fldr = "_".join(self.run_p)
        fldr = self.predictions_folder / self.run_p
        return fldr

    def create_postprocess_transforms(self):
        keys = listify(self.run_p)
        self.postprocess_transforms_dict = {
            # "U": ToDeviceD(keys=keys, device=self.device),
            "MR": RenameDictKeys(new_keys=["pred"], keys=keys),
            "A": AsDiscreted(
                keys=["pred"],
                argmax=True,
            ),
            "Int": CastToTyped(keys=["pred"], dtype=np.uint8),
            "W": MakeWritabled(keys=["pred"]),
            "K": KeepLargestConnectedComponentWithMetad(
                keys=["pred"], independent=False, num_components=self.k_largest
            ),
            "F": FillBBoxPatchesd(),
            "S": SaveImaged(
                keys=["pred"],
                output_dir=self.output_folder,
                output_postfix="",
                separate_folder=False,
                output_dtype=np.uint8,
            ),
        }

    def set_postprocess_tfms_keys(self):
        if self.safe_mode == False:
            self.postprocess_tfms_keys = "MR,A,Int,W"
        else:
            self.postprocess_tfms_keys = "MR,W"
        if self.k_largest is not None:
            self.postprocess_tfms_keys += ",K"
        self.postprocess_tfms_keys += ",F"
        if self.save == True:
            self.postprocess_tfms_keys += ",S"


# %%
if __name__ == "__main__":
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>

    # ... run your application ...
    from fran.managers import Project
    from fran.utils.common import *

    conf_fldr = os.environ["FRAN_CONF"]
    from utilz.fileio import load_yaml
    best_runs = load_yaml(conf_fldr+"/best_runs.yaml")
    run_w = best_runs["run_w"]

    print(best_runs)

# %%

    from fran.data.dataregistry import DS
    fldr_lidc = DS["lidc"].folder / ("images")
    imgs_lidc = list(fldr_lidc.glob("*"))
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
    nodesthick_imgs = list(nodesthick_fldr.glob("*"))

    bones_fldr = Path("/s/xnat_shadow/bones/images")
    bones_imgs = list(bones_fldr.glob("*"))

    nodes_fldr = Path("/s/xnat_shadow/nodes/images_pending/thin_slice/images")
    nodes_fldr_training = Path("/s/xnat_shadow/nodes/images")
    nodes_imgs = list(nodes_fldr.glob("*"))
    nodes_imgs_training = list(nodes_fldr_training.glob("*"))
    capestart_fldr = Path("/s/insync/datasets/capestart/nodes_2025/images")
    capestart = list(capestart_fldr.glob("*"))

    fldr_misc =  Path("/s/xnat_shadow/misc/images")
    imgs_misc = list(fldr_misc.glob("*"))
    img_fns = [imgs_t6][:20]
    localiser_labels = [45, 46, 47, 48, 49]
    localiser_labels_litsmc = [1]
    TSL = TotalSegmenterLabels()
# %%
# SECTION:-------------------- LIDC-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>

    loc_lidc = [7]  # lung in localiser_label

    safe_mode = False
    devices = [0]
    overwrite = True
    debug_ = True
    debug_ = False
    save_channels = False
    save_localiser = True
# %%
    En = CascadeInferer(
        run_w,
        run_lidc2,
        save_channels=save_channels,
        save_localiser=save_localiser,
        devices=devices,
        localiser_labels=loc_lidc,
        safe_mode=safe_mode,
        k_largest=None,
        debug=debug_,
    )

# %%
    imgs_tmp = ["/s/insync/datasets/today/mets/201 Axial  iDose (6).nii.gz"]
    imgs_tmp = ["/s/xnat_shadow/litq/test/images/litq_10.nii.gz"]
    preds = En.run(imgs_tmp, chunksize=1, overwrite=overwrite)

# %%
    model = En.Ps[0].model

# %%
# SECTION:-------------------- NODES -------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>
    localiser_labels = set(TSL.label_localiser)
    safe_mode = True
    patch_overlap=0.0
    devices = [1]
    overwrite = True
    save_channels = False
    save_localiser = True
    En = CascadeInferer(
        run_w,
        run_nodes,
        patch_overlap=patch_overlap,
        save_channels=save_channels,
        devices=devices,
        localiser_labels=localiser_labels,
        save_localiser=save_localiser,
        safe_mode=safe_mode,
        k_largest=None,
    )

# %%
    imgs = imgs_misc
    imgs = nodes_imgs

    preds = En.run(imgs, chunksize=1, overwrite=overwrite)
    # preds = En.run(img_fns, chunksize=2)
# %%
#SECTION:-------------------- BONES--------------------------------------------------------------------------------------

    run = best_runs["projects"]["bones"]
    localiser_labels= run['localiser_labels']
    if localiser_labels=="TSL.label_localiser":
        localiser_labels = set(TSL.label_localiser)
    run_name = run['run_ids'][0]
# %%
    safe_mode = True
    patch_overlap=0.0
    devices = [1]
    overwrite = True
    save_channels = False
    save_localiser = True
    En = CascadeInferer(
        run_w,
        run_name,
        patch_overlap=patch_overlap,
        save_channels=save_channels,
        devices=devices,
        localiser_labels=localiser_labels,
        save_localiser=save_localiser,
        safe_mode=safe_mode,
        k_largest=None,
    )

# %%
    imgs = bones_imgs

    preds = En.run(imgs, chunksize=1, overwrite=overwrite)
    # preds = En.run(img_fns, chunksize=2)
# %%

# SECTION:-------------------- TOTALSEG WholeImageinferer-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>

    devices = [1]
    debug_ = False
    safe_mode = True

    W = WholeImageInferer(
        run_w,
        project_title="totalseg",
        safe_mode=safe_mode,
        k_largest=None,
        save_channels=False,
        debug=debug_,
        devices=devices,
    )
# %%

    imgs = nodes_imgs_training
    imgs = nodes_imgs[:2]
    imgs = imgs_lidc
    # preds = W.run(imgs_crc, chunksize=6)
    preds = W.run(imgs, chunksize=2, overwrite=False)
# %%

    dl = W.pred_dl
    iteri = iter(dl)
    batch = next(iteri)
    img = batch["image"]
    img = img.to("cuda:1")
    pred = W.model(img)

    pred = pred[0]
    pred2 = torch.argmax(pred, dim=1)
    ImageMaskViewer([img[0, 0].detach().cpu(), pred2[0].detach().cpu()])
# %%
# SECTION:-------------------- TOTALSEG LBD (TOTALSEG WB followed by TOTALSEG LGD)-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>

    localiser_labels = set(TSL.label_localiser)
    safe_mode = True
    devices = [0]
    overwrite = False
    save_channels = False
    save_localiser = False
    En = CascadeInferer(
        run_w,
        run_totalseg,
        project_title="totalseg",
        save_channels=save_channels,
        devices=devices,
        localiser_labels=localiser_labels,
        save_localiser=save_localiser,
        safe_mode=safe_mode,
        k_largest=None,
    )

# %%

    # preds = En.run(capestart, chunksize=2)
    preds = En.run(nodes, chunksize=2)
    # preds = En.run(img_fns, chunksize=2)
# %%
# SECTION:---------------------------------------- LITSMC predictions-------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>
    run = run_litsmc2
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
        localiser_labels=localiser_labels_litsmc,
        safe_mode=safe_mode,
        save_localiser=save_localiser,
        k_largest=k_largest,
    )

# %%

    img_fns = ["/s/insync/datasets/today/mets/201 Axial  iDose (6).nii.gz"]
    img_fns = list(img_fldr.glob("*"))[:20]
    img_fns = imgs_crc[:20]
    case_id = "crc_CRC089"
    # imgs_crc = [fn for fn in imgs_crc if case_id in fn.name]
    tn = time.time()
    # preds = En.run(imgs_crc[:30], chunksize=4)
# %%
    preds = En.run(img_fns, chunksize=4, overwrite=overwrite)
    t2 = time.time()
    lapse = t2 - tn
# %%
    imgs_tmp = ["/s/xnat_shadow/litq/test/images/litq_10.nii.gz"]
    preds = En.run(imgs_tmp, chunksize=4, overwrite=overwrite)

# %%
    preds = En.W.postprocess(p)
    bboxes = []
# %%

# SECTION:-------------------- TROUBLESHOOTING En.run-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>
# SECTION:-------------------- extract_fg_bboxes-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>
    imgs_sublist = imgs_tmp

    En.create_postprocess_transforms()
    data = load_images(imgs_sublist)

    En.bboxes = En.extract_fg_bboxes(data)

    data = En.apply_bboxes(data, En.bboxes)
    En.debug = True
    pred_patches = En.patch_prediction(data)
    pred_patches = En.decollate_patches(pred_patches, En.bboxes)
    output = En.postprocess(pred_patches)
    print(output[0]["pred"].shape)
    # if En.save == True:
    #     En.save_pred(output)
# %%
    img = data[0]["image"].cpu().detach()
    prd = pred_patches[0]["LITS-911"].cpu().detach()
    ImageMaskViewer([img, prd[2]])
# %%
    spacing = get_patch_spacing(En.run_w)
    Sel = SelectLabels(keys=["pred"], labels=list(En.localiser_labels))
    B = BBoxFromPTd(keys=["pred"], spacing=spacing, expand_by=10)
    if overwrite == False:
        print("Bbox overwrite not implemented yet")
    print("Starting localiser data prep and prediction")
# %%
    imgs_sublist = capestart
    data = load_images(imgs_sublist[:3])
    En.bboxes = En.extract_fg_bboxes(data)
    data = En.apply_bboxes(data, En.bboxes)

# %%

    spacing = get_patch_spacing(En.run_w)
    Sel = SelectLabels(keys=["pred"], labels=En.localiser_labels)
    B = BBoxFromPTd(keys=["pred"], spacing=spacing, expand_by=10)
    print("Starting localiser data prep and prediction")
    En.W.setup()
    En.W.prepare_data(data)
    En.W.create_and_set_postprocess_transforms()
    bboxes = []
    for batch in En.W.predict():
        if En.debug == True:
            tr()
        # p = En.W.predict()
        pred = En.W.postprocess(batch)
        pred = Sel(pred)
        pred = B(pred)
        bb = pred["bounding_box"]
        # Check if bounding box is empty
        if bb is None or (isinstance(bb, (list, tuple)) and len(bb) == 0):
            raise ValueError(
                "No bounding box found - localizer failed to detect region of interest"
            )
        bb = validate_bbox(bb)
# %%
    lm = batch["pred"][0][0].cpu().detach().numpy()
    ImageMaskViewer([lm, lm])
# %%
    pred_patches = En.patch_prediction(data)
    pred_patches = En.decollate_patches(pred_patches, En.bboxes)
    output = En.postprocess(pred_patches)
    if En.save == True:
        En.save_pred(output)
    En.cuda_clear()

# %%
    P = En.Ps[0]
    dici = P.L(data)
# %%
    pred = pred_patches[0]["LITS-1238"]
    preds[0]["pred"].shape
    preds[0]["image"].shape
    ImageMaskViewer([preds[0]["pred"][0].detach(), preds[0]["image"][0].detach()])
    ImageMaskViewer([image, pred[0]])
# %%
    bboxes = []
    for pred in preds:
        pred = Sel(pred)
        pred = B(pred)
        bb = pred["bounding_box"]
        bboxes.append(bb)

    print(bboxes)
# %%
    dat = data[0]
    image = dat["image"]
    pred = dat["pred"]
    ImageMaskViewer([image.detach(), image.detach()])
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
# SECTION:-------------------- process_imgs_sublist-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>
        n = 4
        data = En.load_images(nodes[:5])
        img0 = data[n]["image"]
        print(img0.shape)
        En.bboxes = En.extract_fg_bboxes(data)
        data2 = En.apply_bboxes(data, En.bboxes)
# %%
        img1 = data2[n]["image"]
        print(img0.shape)
        print(img1.shape)
# %%

        img0 = img0.permute(2, 1, 0)
        img1 = img1.permute(2, 1, 0)
        ImageMaskViewer([img0, img1])
# %%
# SECTION:--------------------Patch predictor -------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>

        imgs_sublist = nodes[:3]
        data = En.load_images(imgs_sublist)
        En.bboxes = En.extract_fg_bboxes(data)
        data = apply_bboxes(data, En.bboxes)
# %%
        # pred_patches = En.patch_prediction(data)

        print("Starting patch data prep and prediction")
        preds_all_runs = {}
        preds_all_runs[En.P.run_name] = []
        En.P.setup()
        En.P.prepare_data(data=data, collate_fn=img_bbox_collated)
        En.P.create_postprocess_transforms(En.P.ds.transform)
        En.P.pp_transforms  # ['Sq'](batch)
# %%
        for batch in En.P.predict():
            batch = En.P.postprocess_compose(batch)
            # preds = En.P.predict()
            # preds = En.P.postprocess(preds)
            preds_all_runs[En.P.run_name].append(batch)
# %%

        print(batch.keys())
        print(batch["pred"].shape)
# %%
    fn = "/s/fran_storage/predictions/lidc2/LITS-911/litq_10.nii.gz"
    lm = sitk.ReadImage(fn)

    lm.GetSize()
# %%
