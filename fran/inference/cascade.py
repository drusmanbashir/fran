# %%
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm
import gc
import ipdb
from monai.transforms.post.array import KeepLargestConnectedComponent
from monai.transforms.utility.array import ToTensor
from monai.transforms.utility.dictionary import SqueezeDimd
from fran.transforms.misc_transforms import SelectLabels
from fran.transforms.spatialtransforms import ResizeToMetaSpatialShaped

from fran.transforms.totensor import ToTensorI, ToTensorT
tr = ipdb.set_trace

from monai.utils.misc import ensure_tuple
import sys
from collections.abc import Callable
from pathlib import Path
from monai.data.image_reader import ITKReader

import numpy as np
import SimpleITK as sitk
from lightning.fabric import Fabric
from monai.apps.detection.transforms.array import *
from monai.data.box_utils import *
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset, PersistentDataset
from monai.data.utils import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.inferers.merger import *
from monai.transforms import (AsDiscreted, Compose, EnsureChannelFirstd,
                              Invertd, Spacingd)
from monai.transforms.io.dictionary import LoadImaged, SaveImaged
from monai.transforms.post.dictionary import (Activationsd,
                                              KeepLargestConnectedComponentd,
                                              MeanEnsembled)
from monai.transforms.spatial.dictionary import Orientationd, Resized
# from monai.transforms.utility.dictionary import AddChanneld, EnsureTyped
from torchvision.transforms.functional import resize

from fran.data.dataset import (FillBBoxPatchesd,  NormaliseClipd,
                               SavePatchd)
from fran.inference.base import (BaseInferer, InferenceDatasetNii,
                                 list_to_chunks, load_dataset_params)
from fran.managers.training import (DataManager, UNetTrainer,
                                    checkpoint_from_model_id)
from fran.transforms.imageio import LoadSITKd, SITKReader
from fran.transforms.inferencetransforms import (
    BBoxFromPTd, KeepLargestConnectedComponentWithMetad, RenameDictKeys, SaveMultiChanneld,
    ToCPUd, TransposeSITKd)
from fran.utils.common import *
from fran.utils.dictopts import DictToAttr
from fran.utils.fileio import load_dict, load_yaml, maybe_makedirs
from fran.utils.helpers import get_available_device, timing
from fran.utils.itk_sitk import *
from fran.utils.string import drop_digit_suffix, find_file

sys.path += ["/home/ub/code"]
from label_analysis.helpers import to_cc, to_int, to_label

# These are the usual ipython objects, including this one you are creating
ipython_vars = ["In", "Out", "exit", "quit", "get_ipython", "ipython_vars"]
import sys

from fastcore.all import GetAttr, ItemTransform, Pipeline, Sequence
from fastcore.foundation import L, Union, listify, operator

from fran.transforms.intensitytransforms import ClipCenterI

sys.path += ["/home/ub/Dropbox/code/fran/"]

import torch.nn.functional as F
from fastcore.basics import store_attr
from fastcore.transform import Transform as TFC

from fran.utils.imageviewers import ImageMaskViewer


    


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
    def __init__(
        self,
        project,
        run_name,
        devices=[1],
        save_channels=True,
        save=True,
        overwrite=False,
        **kwargs
    ):

        super().__init__(project=project, run_name=run_name, devices=devices, save_channels=save_channels,overwrite= overwrite,save=save,**kwargs)

    def create_transforms(self):
        super().create_transforms()
        self.S = Resized(keys=["image"], spatial_size=self.dataset_params["patch_size"])

class PatchInferer(BaseInferer):
    def __init__(
        self,
        project,
        run_name,
        patch_overlap=0.25,
        bs=1,
        grid_mode="gaussian",
        devices=[1],
        save_channels=True,
        overwrite=False,
        **kwargs
    ):
        super().__init__(project=project, run_name=run_name, devices= devices, save_channels=save_channels,overwrite= overwrite,save=False,**kwargs)


    def create_postprocess_transforms(self,preprocess_transform):
        Sq = SqueezeDimd(keys=["image","pred"], dim=0)

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

        tfms = [Sq,I,  U]
        if self.save_channels == True:
            tfms = [Sq,I,Sa,U]
        C = Compose(tfms)
        self.postprocess_transforms = C

class CascadeInferer(BaseInferer):  # SPACING HAS TO BE SAME IN PATCHES
    def __init__(
        self,
        project,
        run_name_w,
        runs_p,
        localiser_labels:list ,  #these labels will be used to create bbox
        devices=[0],
        overwrite=True,
        safe_mode=False,
        profile=None,
        save_channels=False,
        save=True,
        k_largest=None, # assign a number if there are organs involved
    ):
        """
        Creates a single dataset (cascade dataset) which normalises images once for both patches and whole images. Hence, the model should be developed from the same dataset std, mean values.
        """
        assert profile in [None, "dataloading", "prediction", "all"], "Choose one of None , 'dataloading', 'prediction', 'all'"

        self.predictions_folder = project.predictions_folder
        self.dataset_params = load_dataset_params(runs_p[0])
        self.Ps = [PatchInferer(project=project,run_name=run, devices=devices, save_channels=save_channels, overwrite=overwrite,safe_mode=safe_mode) for run in runs_p]
        self.localiser_tfms= "ESN"
        WSInf = self.inferer_from_params(run_name_w)
        self.W = WSInf( project=project, run_name=run_name_w,  save_channels=save_channels, devices=devices, overwrite=overwrite,safe_mode=safe_mode)
        store_attr()
    def setup(self): pass

    def inferer_from_params(self,run_name_w):
          self.ckpt = checkpoint_from_model_id(run_name_w)
          dic1 = torch.load(self.ckpt)
          mode = dic1['datamodule_hyper_parameters']['dataset_params']['mode']
          if mode == "source":
            return BaseInferer
          elif mode == "whole":
            return WholeImageInferer
    def get_patch_spacing(self, run_name):
        ckpt = checkpoint_from_model_id(run_name)
        dic1 = torch.load(ckpt)
        spacing = dic1["datamodule_hyper_parameters"]["dataset_params"]["spacing"]
        return spacing


    def run(self, imgs:list, chunksize=12):
        """
        imgs can be a list comprising any of filenames, folder, or images (sitk or itk)
        chunksize is necessary in large lists to manage system ram
        """
        imgs=listify(imgs)
        if self.overwrite==False and (isinstance(imgs[0],str) or isinstance(imgs[0], Path)):
            imgs = self.filter_existing_preds(imgs)
        else:
            pass
            # self.save = False  # don't save if input is pure images. Just output those.
        if len(imgs)>0:
            imgs = list_to_chunks(imgs, chunksize)
            for imgs_sublist in imgs:
                output = self.process_imgs_sublist(imgs_sublist)
            return output
        else: return 1

    def process_imgs_sublist(self,imgs_sublist):
                data = self.load_images(imgs_sublist)
                self.bboxes = self.extract_fg_bboxes(data)
                data = self.apply_bboxes(data,self.bboxes)
                pred_patches = self.patch_prediction(data )
                pred_patches = self.decollate_patches(pred_patches, self.bboxes)
                output = self.postprocess(pred_patches)
                if self.save == True:
                    self.save_pred(output)
                self.cuda_clear()
                return output

    def apply_bboxes(self,data,bboxes):
        data2=[]
        for i, dat in enumerate(data):
            dat['image'] = dat['image'][self.bboxes[i][1:]]
            dat['bounding_box'] = self.bboxes[i]
            data2.append(dat)
        return data2



    def filter_existing_preds(self, imgs):

        print("Filtering existing predictions\nNumber of images provided: {}".format(len(imgs)))
        out_fns = [self.output_folder/img.name for img in imgs]
        to_do= [not fn.exists() for fn in out_fns]
        imgs = list(il.compress(imgs, to_do))
        print("Number of images remaining to be predicted: {}".format(len(imgs)))
        return imgs


    def filter_existing_localisers(self, imgs):
        print("Filtering existing localisers\nNumber of images provided: {}".format(len(imgs)))
        new_W =[]
        for P in self.Ps:
            out_fns = [P.output_folder/img.name for img in imgs]
            new_P = np.array([not fn.exists() for fn in out_fns])
            new_Ps.append(new_P)
        if len(P)>1:
            new_Ps= np.logical_or(*new_Ps)
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

    def extract_fg_bboxes(self,data):
        Sel = SelectLabels(keys = ['pred'],labels = self.localiser_labels)
        B = BBoxFromPTd(keys = ['pred'],spacing = self.W.dataset_params['spacing'], expand_by =10)
        if self.overwrite==False:
            print("Bbox overwrite not implemented yet")
        print("Starting localiser data prep and prediction")
        self.W.setup()
        self.W.prepare_data(data,tfms="ESN")
        p = self.W.predict()
        preds = self.W.postprocess(p)
        bboxes = []
        for pred in preds:
            pred = Sel(pred)
            pred = B(pred)
            bb = pred['bounding_box']
            bboxes.append(bb)
        return bboxes

    def patch_prediction(self, data):
        del self.W.model
        torch.cuda.empty_cache()
        preds_all_runs = {}
        print("Starting patch data prep and prediction")
        for P in self.Ps:
            P.setup()
            P.prepare_data(data=data,tfms='ESN', collate_fn=img_bbox_collated)
            preds = P.predict()
            preds = P.postprocess(preds)
            preds_all_runs[P.run_name] = preds
        return preds_all_runs

    @property
    def output_folder(self):
        fldr = "_".join(self.runs_p)
        fldr = self.predictions_folder / fldr
        return fldr

    def postprocess(self, patch_bundle):
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
            tfms = [MR, A, D,  F]
        # S = SaveListd(keys = ['pred'],output_dir=self.output_folder,output_postfix='',separate_folder=False)


        C = Compose(tfms)
        output = C(patch_bundle)
        return output

class CascadeFew(CascadeInferer):
    pass

# %%


if __name__ == "__main__":
    # ... run your application ...
    project = Project(project_title="litsmc")

    run_w = "LIT-145"
    run_ps = ["LIT-143", "LIT-150", "LIT-149", "LIT-153", "LIT-161"]
    run_ps = ["LITS-630", "LITS-633", "LITS-632", "LITS-647", "LITS-650"]
    run_ps = ["LITS-720"]

    run_ps = ["LITS-787", "LITS-810", "LITS-811"]
    run_lidc2 = ["LITS-902"]
    run_lidc2 = ["LITS-842"]
    run_lidc2= ["LITS-913"]
    run_lidc2= ["LITS-911"]
    run_ts= ["LITS-827"]
# %%
    img_fna = "/s/xnat_shadow/litq/test/images_ub/"
    fns = "/s/datasets_bkp/drli_short/images/"
    img_fldr= Path("/s/xnat_shadow/lidc2/images/")
    img_fn2= "/s/xnat_shadow/crc/wxh/images/crc_CRC198_20170718_CAP1p51.nii.gz"
    img_fn3= "/s/xnat_shadow/crc/srn/images/crc_CRC002_20190415_CAP1p5.nii.gz"


    imgs_fldr = Path("/s/xnat_shadow/crc/images")
    srn_fldr = "/s/xnat_shadow/crc/srn/cases_with_findings/images/"
    srn_imgs = list(Path(srn_fldr).glob("*"))
    wxh_fldr = "/s/xnat_shadow/crc/wxh/completed/"
    wxh_imgs = list(Path(wxh_fldr).glob("*"))
    litq_fldr = "/s/xnat_shadow/litq/test/images_ub/"
    litq_imgs = list(Path(litq_fldr).glob("*"))
    t6_fldr = Path("/s/datasets_bkp/Task06Lung/images")
    imgs_t6 = list(t6_fldr.glob("*"))
# %%
    react_fldr = Path("/s/insync/react/sitk/images")
    imgs_react = list(react_fldr.glob("*"))

    img_fns = [imgs_t6][:20]
    localiser_labels =[1]
    localiser_labels =[45,46,47,48,49]
    runs_p = run_ps
# %%
    run_w = run_ts[0]
    runs_p = run_lidc2
# %%
    project = Project(project_title="lidc2")
    safe_mode=True
    En = CascadeInferer(project, run_w, runs_p, save_channels=False, devices=[0],overwrite=True,localiser_labels=localiser_labels,safe_mode=safe_mode)

# %%
    # img_fns = list(img_fldr.glob("*"))[20:50]
    preds = En.run(imgs_t6)

# %%
    imgs_sublist = imgs_react
    data = En.load_images(imgs_sublist)
    En.bboxes = En.extract_fg_bboxes(data)
    data = En.apply_bboxes(data,En.bboxes)

# %%

