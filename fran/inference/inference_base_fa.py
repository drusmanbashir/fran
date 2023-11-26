#sjupyter: ---
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from fastai.data.core import get_model
from fran.architectures.create_network import create_model_from_conf
from fran.managers.base import load_checkpoint, reconcile_keys
from fran.utils.common import *
import sys
import gc
from fran.callback.neptune import NeptuneManager
from fran.transforms.inferencetransforms import *
from monai.inferers.utils import sliding_window_inference
from monai.inferers import SlidingWindowInferer
from torch.functional import Tensor
import numpy as np
from fran.inference.helpers import get_scale_factor_from_spacings
from fran.utils.fileio import load_dict, maybe_makedirs
import SimpleITK as sitk
from pathlib import Path
from fran.utils.helpers import (
    get_available_device,
)
from fran.utils.string import drop_digit_suffix
sys.path+=["/home/ub/code"]
from mask_analysis.helpers import to_int, to_label, to_cc

# These are the usual ipython objects, including this one you are creating
ipython_vars = ["In", "Out", "exit", "quit", "get_ipython", "ipython_vars"]
from fastcore.foundation import L, Union, listify, operator
from fastcore.all import GetAttr, ItemTransform, Pipeline
from fran.inference.scoring import compute_dice_fran
from fran.transforms.intensitytransforms import ClipCenter, ClipCenterI
from monai.transforms.post.array import VoteEnsemble
import os

from fran.transforms.totensor import ToTensorI, ToTensorT

import sys

sys.path += ["/home/ub/Dropbox/code/fran/"]
from fastcore.transform import Transform
from fran.utils.imageviewers import ImageMaskViewer, view_sitk
import functools as fl

from fastcore.basics import store_attr

from fran.utils.imageviewers import ImageMaskViewer
import torch.nn.functional as F


def get_epoch(fn:Path):
    pat = r"model_(\d*)"
    name = fn.name
    m = re.match(pat,name)
    epoch = int(m.groups()[0])
    return epoch

def sitk_bbox_readable(bboxes:list):# input list of bboxes in sitk format as [starts*3, sizes*3 ]. Outputs lists of [starts*3,stops*3]
     bboxes_out = []
     for bb in bboxes:
         starts = bb[:3]
         sizes = bb[3:]
         stops = [a+b for a,b in zip(starts, sizes)]
         bb_final= [*bb[:3],*stops]
         bboxes_out.append(bb_final)
     return bboxes_out

def sitk_to_slices(bboxes:list)-> list  :#of slices of bboxes

    a = 0
    b = 3
    sls =[]
    for bb in bboxes:
        sl = [slice(bb[a+x],bb[b+x]) for x in range(3)]
        sl = tuple([sl[2],sl[1],sl[0]])
        sls.append(sl)
    return sls

# from experiments.kits21.kits21_repo.evaluation.metrics import *
# HEC_SD_TOLERANCES_MM = {
#     'kidney_and_mass': 1.0330772532390826,
#     'mass': 1.1328796488598762,
#     'tumor': 1.1498198361434828,
# }
#

#
# def get_bbox_from_tnsr(pred_int):
#         pred_int_np = np.array(pred_int)
#         stats = cc3d.statistics(pred_int_np)
#         bboxes = stats["bounding_boxes"][1:]  # bbox 0 is the whole image
#         bboxes
#         if len(bboxes) < 1:
#             tr()
#         return bboxes

def pred_mean(preds):
    """
    preds are supplied as raw model output
    """

    pred_avg = torch.stack(preds)
    pred_avg = torch.mean(pred_avg, dim=0)
    return pred_avg


def pred_voted(preds_int: list):
    V = VoteEnsemble(num_classes=3)
    preds_int_vote = [pred.unsqueeze(0) for pred in preds_int]
    out = V(preds_int_vote)
    out.squeeze_(0)
    return out


class PredFlToInt(Transform):
    def encodes(self, pred_fl: Tensor):
        pred_int = torch.argmax(pred_fl, 0, keepdim=False)
        pred_int = pred_int.to(torch.uint8)
        return pred_int


class ToNumpy(Transform):
    def __init__(self, encode_dtype=np.uint8):
        if encode_dtype == np.uint8:
            self.decode_dtype = torch.uint8
        self.encode_dtype = encode_dtype

    def encodes(self, tnsrs):
        return [np.array(tnsr, dtype=self.encode_dtype) for tnsr in tnsrs]


class FillBBoxPatches(ItemTransform):
    """
    Based on size of original image and n_channels output by model, it creates a zerofilled tensor. Then it fills locations of input-bbox with data provided
    """

    def __init__(self, img_size, out_channels,device='cuda'):
        self.output_img = torch.zeros(out_channels, *img_size,device=device)

    def decodes(self, x):
        patches, bboxes = map(listify,x)
        
        for bbox, pred_patch in zip(bboxes, patches):
            for n in range(self.output_img.shape[0]):
                self.output_img[n][bbox] = pred_patch[n]
        return self.output_img


class _Predictor():
    def __init__(self,
        dataset_params,
        proj_defaults,
        out_channels,
        resample_spacings,
        patch_size: list ,
        patch_overlap:  float = 0.25,
        grid_mode="constant",  # constant or gaussian
        bs=8,
        device='cuda',
        half=False,
        merge_labels:list=[[]], #list of lists used by stencil transform. 
        postprocess_label:int=1,
        cc3d=True,
        debug=False,
        overwrite=False):

        """
        params:
        cc3d: If True, dusting and k-largest components are extracted based on mask-labels.json (corrected from mm^3 to voxels)
        patch_overlap: float : [0,1] percent overlap of patch_size

        """


        assert grid_mode in [
            "constant",
            "gaussian",
        ], "grid_mode should be either 'constant' or 'gaussian' "
        self.grid_mode = grid_mode
        self.inferer = SlidingWindowInferer(roi_size = patch_size,sw_batch_size =bs,overlap=patch_overlap,device=device,
                                                     mode=grid_mode,progress=True)

        mask_label  = load_dict(Path("/s/fran_storage/projects/mask_labels.json"))[str(postprocess_label)]
        self.k_largest = mask_label['k_largest']
        self.dusting_threshold=mask_label['dusting_threshold']
        store_attr(but='bs,patch_overlap,grid_mode')

    def load_model(self, model, model_id):
        self.model = model
        self.model.eval()
        self.model.to(self.device)
        if self.half==True:
            self.model.half()
        self.model_id = model_id
        print(
            "Default output folder, based on model name is {}.".format(
                self.output_image_folder
            )
        )

    def img_sized_bbox(self):
        shape = self.img_np_orgres.shape
        slc = []
        for s in shape:
            slc.append(slice(0, s))
        return [tuple(slc)]

    def sitk_process(self):
            self.set_sitk_props()
            self.img_np_orgres = sitk.GetArrayFromImage(self.img_sitk)


    def load_case(
        self, img_sitk, bboxes=None
    ):  # tip put this inside a transform which saves these attrs to parent like callbacks do in learner
            self.img_sitk = img_sitk
            self.sitk_process()  # flips image if needed
            self.bboxes = bboxes if bboxes else self.img_sized_bbox()
            # self.sitk_process()
            self.already_processed = False


    def create_postprocess_pipeline(self):
        self.postprocess_tfms = L(
            PredFlToInt,
            ArrayToSITKI(sitk_props=self.sitk_props),
        )

        self.postprocess_tfms.map(self.add_tfm)
        self.postprocess_pipeline = Pipeline(self.postprocess_tfms)

    def add_tfm(self, tfm):
        if isinstance(tfm, type):
            tfm = tfm()
        tfm.predictor = self
        setattr(self, tfm.name, tfm)
        return self


    def img_sized_bbox(self):
        shape = self.img_np_orgres.shape
        slc = []
        for s in shape:
            slc.append(slice(0, s))
        return [tuple(slc)]



    def set_sitk_props(self):
        origin = self.img_sitk.GetOrigin()
        spacing = self.img_sitk.GetSpacing()
        direction = self.img_sitk.GetDirection()
        direction_std = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        if direction != direction_std:
            self.img_sitk = sitk.DICOMOrient(self.img_sitk, "LPS")
        self.sz_dest, self.scale_factor = get_scale_factor_from_spacings(
            self.img_sitk.GetSize(), spacing, self.resample_spacings
        )
        self.sitk_props = origin, spacing, direction


    def run(self, img_sitk, bboxes=None, save=True):
        """
        Runs predictions. Then backsamples predictions to img size (DxHxW). Keeps num_channels
        """

        self.load_case(img_sitk, bboxes)
        if self.already_processed == False:
            self.create_encode_pipeline()
            self.create_decode_pipeline()
            self.create_postprocess_pipeline()

            self.img_transformed, self.bboxes_transformed = self.encode_pipeline(
                        [self.img_np_orgres, self.bboxes]
                    )
            self.make_prediction()

            self.backsample()
            self.postprocess(self.cc3d)
            if save == True:
                self.save_prediction()

    @property
    def n_classes(self):
        return self.out_channels

    def set_pred_fns(self, img_fn, ext=None):
        if not ext:
            ext = "." + get_extension(img_fn)
        fn_no_ext = img_fn.name.split(".")[0]
        counts = ["_" + str(x + 1) for x in range(self.n_classes)]
        self.pred_fn_i = self.output_image_folder / (fn_no_ext + ext)
        self.pred_fns_f = [
            self.output_image_folder / (fn_no_ext + c + ext) for c in counts
        ]

    def files_exist(self):
        files_exist = [self.pred_fn_i.exists()]
        if self.debug == True:
            files_exist.append([a.exists() for a in self.pred_fns_f])
        return all(files_exist)

    def dust(self):
        pred_binary_cc = sitk.ConnectedComponent(self.pred_sitk_i)
        fil_cc = sitk.RelabelComponentImageFilter()
        fil_cc.SetSortByObjectSize(True)
        pred_cc_sorted = fil_cc.Execute(pred_binary_cc)
        n_labels_pred = fil_cc.GetNumberOfObjects()
        if n_labels_pred>self.k_largest:
            dust_labels = np.arange(self.k_largest,n_labels_pred)+1
            dici = {int(label):0 for label in dust_labels}
            pred_cc_sorted = sitk.ChangeLabelLabelMap(to_label(pred_cc_sorted), dici)
        self.pred_sitk_i  = sitk.Cast(pred_cc_sorted,sitk.sitkUInt32)


    def save_sitk(self, img, fn):
        print("Saving prediction. File name : {}".format(fn))
        if img.GetPixelID() == 22:
                img = to_int(img)
        sitk.WriteImage(img, fn)

    def save_prediction(self, ext=None):
        maybe_makedirs(self.output_image_folder)
        imgs = [self.pred_sitk_i]
        fns = [self.pred_fn_i]
        if self.debug == True:
            imgs.extend(self.pred_sitk_f)
            fns.extend(self.pred_fns_f)
        for img, fn in zip(imgs, fns):
            self.save_sitk(img, fn)

    def make_prediction(self):
        self.pred_patches = []
        img_input= self.img_transformed[self.bboxes_transformed[0]]
        img_input = img_input.unsqueeze(0).unsqueeze(0).to(self.device)
        if self.half==True:
            img_input = img_input.half()
        with torch.no_grad():
            output_tensor = self.inferer(inputs = img_input,network=self.model)        
        output_tensor= output_tensor.squeeze(0)

        output_tensor = torch.nan_to_num(output_tensor, 0,)
        # output_tensor = output_tensor.float().cpu()
        output_tensor = F.softmax(output_tensor, dim=0)
        self.pred_patches.append(output_tensor)

   
 

    def postprocess(self, cc3d: bool):  # starts : pred->dust->k-largest->pred_int
        if cc3d == True:
            self.pred_sitk_i= self.postprocess_pipeline(self.pred)
            self.dust()
        else:
            pred_int = torch.argmax(self.pred, 0, keepdim=False)
            pred_int = pred_int.to(torch.uint8)
            ArrayToSITKI = [Tf for Tf in self.postprocess_pipeline if Tf.name=='ArrayToSITKI'][0]
            self.pred_sitk_i = ArrayToSITKI.encodes(pred_int)
        self.pred = self.pred.float().cpu()


    def unload_case(self):
        to_delete = ["pred_int", "_pred_sitk_f", "pred_sitk_i"]
        for item in to_delete:
            if hasattr(self, item):
                delattr(self, item)
        gc.collect()
        torch.cuda.empty_cache()

    def score_prediction(self, mask_filename, n_classes):
        mask_sitk = sitk.ReadImage(mask_filename)
        self.scores = compute_dice_fran(self.pred_int, mask_sitk, n_classes)
        return self.scores


    @classmethod
    def from_neptune(cls,proj_defaults, run_name, device="cuda"):
        NepMan = NeptuneManager(proj_defaults)
        NepMan.load_run(
            run_name=run_name, param_names="default", nep_mode="read-only"
        )
        model,        patch_size,        resample_spacings,        out_channels,        = NepMan.load_model_neptune(run_name, device="cpu")
        cls = cls(proj_defaults,out_channels,resample_spacings,patch_size)
        cls.load_model(model,run_name)
        return cls


    # @property
    # def pred_fns(self):
    #     if self.debug==True:
    #         counts = ["","_1","_2","_3","_4"][:len([self.pred_sitk_i,*self.pred_sitk_f])]
    #         self.pred_fns= [self._output_image_folder/(self.img_filename.name.split(".")[0]+c+ext) for c in counts]
    #         print("Saving prediction. File name : {}".format(self.pred_fns))
    #         for pred_sitk,fn in zip([self.pred_sitk_i]+self.pred_sitk_f,self.pred_fns):
    #             sitk.WriteImage(pred_sitk,fn)
    #     else:
    #         fn  = self._output_image_folder/(self.img_filename.name.split(".")[0]+ext)
    #

    @property
    def pred_sitk_f(self):  # list of sitk images len =  out_channels
        self._pred_sitk_f = ArrayToSITKF(sitk_props=self.sitk_props).encodes(self.pred)
        return self._pred_sitk_f[1:]  #' first channel is only bg'
    #
    # @property
    # def Pred_sitk_i(self):  # list of sitk images len =  out_channels
    #     self._pred_sitk_i = ArrayToSITKI(sitk_props=self.sitk_props).encodes(
    #         self.pred_int
    #     )
    #     return self._pred_sitk_i
    #
    @property
    def output_image_folder(self):
        self._output_image_folder = (
            Path(self.proj_defaults.predictions_folder) / self.model_id
        )
        return self._output_image_folder

    @property
    def case_id(self):
        """The case_id property."""
        return self.predictor_p.case_id

    @case_id.setter
    def case_id(self, value):
        self._case_id = value


class PatchPredictor(_Predictor):
    def __init__(

        self,
        dataset_params,
        proj_defaults,
        out_channels,
        resample_spacings,
        patch_size: list ,
        patch_overlap=0.2,
        bs = 8,
        device=None,
        half = False,
        merge_labels=[[2],[]],
        postprocess_label=2,
        cc3d=True,
        debug=False,
        overwrite=False,
        expand_bbox=0.1
    ):
        store_attr('expand_bbox')
        super().__init__(
            dataset_params=dataset_params,
            proj_defaults=proj_defaults,
            out_channels=out_channels,
            resample_spacings=resample_spacings,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            bs=bs,
            device=device,
            half = half,
            merge_labels=merge_labels,
            postprocess_label=postprocess_label,
            cc3d=cc3d,
            debug=debug,
            overwrite=overwrite
        )


 
    def create_encode_pipeline(self):
        intensity_clip_range,mean_fg,std_fg = ast.literal_eval(self.dataset_params['clip_range']), self.dataset_params['mean_fg'], self.dataset_params['std_fg']
        self.encode_tfms = L(
            ToTensorI(),
            ChangeDType(torch.float32),
            TransposeSITK(),
            ResampleToStage0(self.img_sitk, self.resample_spacings),
            BBoxesToPatchSize(self.patch_size, self.sz_dest, self.expand_bbox),
            ClipCenterI(
                clip_range=intensity_clip_range,
                mean=mean_fg,
                std=std_fg
            ),
        )
        self.encode_tfms.map(self.add_tfm)
        self.encode_pipeline = Pipeline(self.encode_tfms)

    def create_decode_pipeline(self):
        F = FillBBoxPatches(self.sz_dest, self.out_channels)
        self.decode_pipeline = Pipeline(
            [*self.encode_pipeline[2:4], F]
        )  # i.e., TransposeSITK, ResampleToStage0

    def backsample(self):
        self.pred = self.decode_pipeline.decode(
            [self.pred_patches, self.bboxes_transformed]
        )
        # self.pred = self.pred.float().cpu()



class WholeImageBBoxes(ApplyBBox):
    def __init__(self, patch_size):
        bboxes = [tuple([slice(0, p) for p in patch_size])]
        super().__init__(bboxes)

    def encodes(self, x):
        img, _ = x
        return img, self.bboxes

    def decodes(self, x):
        return x  # no processing to do


class Unlist(Transform):
    def decodes(self, x: list):
        assert len(x) == 1, "Only for lists len=1"
        return x[0]
    def encodes(self,x): return self.decodes(x)


class WholeImagePredictor(_Predictor):
    def __init__(
        self,
        dataset_params,
        out_channels,
        resample_spacings,
        patch_size: list = [128, 128, 128],
        bs = 8,
        device=None,
        half = False,
        merge_labels=[[]],
        postprocess_label=1,
        **kwargs
    ):
        super().__init__(
            dataset_params=dataset_params,
            out_channels=out_channels,
            resample_spacings=resample_spacings,
            patch_size=patch_size,
            patch_overlap=0.2,
            bs=bs,
            device=device,
            half = half,
            merge_labels=merge_labels,
            postprocess_label=postprocess_label,
            **kwargs
        )

    def create_encode_pipeline(self):
        To = ToTensorI()
        Ch = ChangeDType(torch.float32)
        T = TransposeSITK()
        Rz = ResizeFran(self.patch_size)
        # P  = PadDeficitImgMask(patch_size=self.patch_size,input_dims=3)
        W = WholeImageBBoxes(self.patch_size)
        C = ClipCenterI(
            clip_range=ast.literal_eval(self.dataset_params["clip_range"]),
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )
        self.encode_pipeline = Pipeline([To, Ch, T, Rz, W, C])

    def create_decode_pipeline(self):
        U = Unlist()
        self.decode_pipeline = Pipeline(
            [*self.encode_pipeline[2:4], U]
        )  # Transpose, Resize

    def backsample(self):
        self.pred = self.decode_pipeline.decode(self.pred_patches)
        # self.pred = self.pred.float().cpu()


class EndToEndPredictor(_Predictor):
    def __init__(
        self,
        proj_defaults,
        run_name_w,
        run_name_p,
        use_neptune=False,
        patch_overlap=0.5,
        device: int = None,
        save_localiser=False,
        overwrite=False,
    ):
        super().__init__(overwrite)
        print("Loading model checkpoints for whole image predictor")
        if not device:
            device = get_available_device()
        self.NepMan = NeptuneManager(proj_defaults)

        (
            model_w,
            patch_size_w,
            resample_spacings_w,
            out_channels_w,
        ) = self.load_model_neptune(run_name_w, device=device)
        (
            model_p,
            patch_size_p,
            resample_spacings_p,
            out_channels_p,
        ) = self.load_model_neptune(run_name_p, device=device)

        self.w = WholeImagePredictor(
            proj_defaults=proj_defaults,
            out_channels=out_channels_w,
            resample_spacings=resample_spacings_w,
            patch_size=patch_size_w,
            device=device,
        )
        self.w.load_model(model_w, model_id=run_name_w)
        self.save_localiser = save_localiser
        print("\nLoading model checkpoints for patch-based predictor")

        patch_overlap = [int(x * patch_overlap) for x in patch_size_p]
        self.p = PatchPredictor(
            proj_defaults=proj_defaults,
            out_channels=out_channels_p,
            resample_spacings=resample_spacings_p,
            patch_size=patch_size_p,
            patch_overlap=patch_overlap,
            bs=self.bs,
            device=device,
        )
        self.p.load_model(model_p, model_id=run_name_p)
        self.n_classes = out_channels_p

        print(
            "---- You can set alternative save folders by setting properties: output_localiser_folder and output_image_folder for localiser and final predictions respectively.----"
        )

    def load_model_neptune(self, run_name, device="cuda"):
        self.NepMan.load_run(
            run_name=run_name, param_names="default", nep_mode="read-only"
        )
        return self.NepMan.load_model(device)

    # def predict(self, img_fn, save_localiser=None):
    #     if save_localiser:
    #         self.save_localiser = save_localiser
    #     self.localiser_bbox(img_fn)
    #     self.run_patch_prediction(img_fn)
    #

    def localizer_pred_fn(self,img_fn):
        prefix = img_fn.name.split(".")[0]
        pred_w_fns =list((self.proj_defaults.predictions_folder/run_name_w).glob("*"))
        fn = [fn for fn in pred_w_fns if prefix in fn.name ]
        if len(fn) >0: return fn[0]
    def localiser_bbox(self, img_sitk):
           self.load_localiser_model()
           print(
                "Running predictions. Whole image predictor is on device {0}".format(
                    self.w.device
                )
            )
           self.w.run(img_sitk=img_sitk, bboxes=None, save=False)
           pred_localiser = self.w.pred_sitk_i
           pred_localiser =  sitk.DICOMOrient(pred_localiser, "LPS") # patch to erect the localiser before bbox derivation.
           self.create_bboxes_from_localiser(pred_localiser)
           self.unload_localizer_model()

    def create_bboxes_from_localiser(self,pred_localiser: sitk.Image):
        fil = sitk.LabelShapeStatisticsImageFilter()
        fil.Execute(pred_localiser)
        bboxes =[]
        for label in np.arange(self.w.k_largest)+1:
            bbox = fil.GetBoundingBox(int(label))
            bboxes.append(bbox)
        bboxes = sitk_bbox_readable(bboxes)
        self.bboxes = sitk_to_slices(bboxes) # only implmeneted for a single organ

    def unload_localizer_model(self):
        if hasattr(self,"w"):
            delattr(self, "w")
            gc.collect()
            print("BBoxes obtained. Deleting localiser and freeing ram")
            torch.cuda.empty_cache()

    def run_patch_prediction(self, img_fn):
        self.p.run(img_filename=img_fn, bboxes=self.bboxes, save=True)

    def unload_cases(self):
        self.p.unload_case()

    def score_prediction(self, mask_fn):
        self.scores = self.p.score_prediction(mask_fn, self.n_classes)

    @property
    def output_localiser_folder(self):
        return self.w.output_image_folder

    @output_localiser_folder.setter
    def output_localiser_folder(self, folder_name: Path):
        self.w.output_image_folder = folder_name

    @property
    def save_localiser(self):
        """The save_localiser property."""
        return self._save_localiser

    @save_localiser.setter
    def save_localiser(self, value=None):
        assert not value or type(value) == bool, "Illegal value for bool parameter"
        if not value or value == False:
            print("Localizer image will not be saved")
            value = False
        else:
            print(
                "Localizer image will be saved to {}".format(
                    self.output_localiser_folder
                )
            )
        self._save_localiser = value


class EnsemblePredictor(EndToEndPredictor):
    def __init__(
        self,
        proj_defaults,
        out_channels,
        run_name_w,
        runs_p,
        half=False,
        bs=6,
        device='cuda',
        debug=False,
        cc3d=False,
        overwrite=False,
    ):
        """
        param  debug: When true, prediction heatmaps are stored as numbered sitk files, each number representing the prob of that label versus all others
        """
        runs_p = listify(runs_p)
        store_attr()
        self.NepMan = NeptuneManager(proj_defaults)
        self.patch_overlap = 0.25
        self.model_id = "ensemble_" + "_".join(self.runs_p)

    def load_localiser_model(self):
        (
            model_w,
            patch_size_w,
            resample_spacings_w,
            out_channels_w,
        ) = self.load_model_neptune(self.run_name_w, device="cpu")
        dataset_params = self.NepMan.run_dict['dataset_params']
        self.w = WholeImagePredictor(
            dataset_params = dataset_params,
            proj_defaults = self.proj_defaults,
            out_channels=out_channels_w,
            resample_spacings=resample_spacings_w,
            patch_size=patch_size_w,
            bs = self.bs,
            half = self.half,
            device=self.device,
            overwrite=self.overwrite,
        )
        self.w.load_model(model_w, model_id=self.run_name_w)


    
    def load_patch_model(self, n):
        run_name_p = self.runs_p[n]
        (
            model_p,
            patch_size_p,
            resample_spacings_p,
            out_channels_p,
        ) = self.load_model_neptune(run_name_p, device="cpu")

        dataset_params = self.NepMan.run_dict['dataset_params']
        if n == 0:
            self.p = PatchPredictor(
                dataset_params = dataset_params,
                proj_defaults = self.proj_defaults,
                out_channels=out_channels_p,
                resample_spacings=resample_spacings_p,
                patch_size=patch_size_p,
                patch_overlap=self.patch_overlap,
                bs=self.bs,
                half = self.half,
                device=self.device,
                debug=self.debug,
                overwrite=self.overwrite,
            )
            # self.n_classes=out_channels_p

        self.p.load_model(model_p, model_id=run_name_p)

    def patch_prediction(self, img_sitk, n):
        if n == 0:
            self.p.load_case(img_sitk, self.bboxes)
            self.p.create_encode_pipeline()
            self.p.create_decode_pipeline()
            self.p.create_postprocess_pipeline()

            self.p.img_transformed, self.p.bboxes_transformed = self.p.encode_pipeline(
                        [self.p.img_np_orgres, self.p.bboxes]
                    )
        self.p.make_prediction()

        # self.p.backsample()
        # self.p.postprocess(self.cc3d)
        # self.p.save_prediction()
        print(
            "Patch predictions done. Deleting current model in the ensemble and loading next"
        )  # move this to start
        self.p.unload_case()
        del self.p.model
        torch.cuda.empty_cache()

    @property
    def postprocess_pipeline(self):
        return self.p.postprocess_pipeline

    def backsample(self):
        self.pred = self.p.decode_pipeline.decode(
            [self.pred_patches, self.p.bboxes_transformed]
        )

    @property
    def sitk_props(self): return self.p.sitk_props
    def save_prediction(self):
        super().save_prediction()

    def set_filenames(self,img_fn):
        self.set_pred_fns(img_fn)
        self.localizer_pred_fn(self.run_name_w,img_fn)

    def run(self, img_sitk):
            self.localiser_bbox(img_sitk)
            self.pred_patches = []
            for n in range(len(self.runs_p)):
                self.load_patch_model(n)
                self.patch_prediction(img_sitk, n)
                self.p.pred_patches= Unlist().encodes(self.p.pred_patches)
                self.pred_patches.append(self.p.pred_patches)
            self.pred_patches = pred_mean(self.pred_patches)
            self.backsample()
            del self.pred_patches
            gc.collect()
            self.postprocess(self.cc3d)
            # self.save_prediction()
            # self.unload_case()


# %%
if __name__ == "__main__":
    # ... run your application ...



    common_vars_filename = os.environ["FRAN_COMMON_PATHS"]
    P = Project(project_title="lits")
    proj_defaults = P

    import pandas as pd

    mo_df = pd.read_csv(Path("/s/datasets_bkp/litq/complete_cases/cases_metadata.csv"))
    # patch_size = [160, 160, 160]
    # resample_spacings = [1, 1, 2]
    run_name_w = "LITS-627"  # best trial

    # runs_ensemble=["LITS-265","LITS-255","LITS-270","LITS-271","LITS-272"]
    # runs_ensemble=["LITS-408","LITS-385","LITS-383","LITS-357","LITS-413"]
    # runs_ensemble = ["LITS-451", "LITS-452"]
    runs_ensemble = "LITS-484,LITS-485,LITS-492,LITS-487,LITS-488".split(",")
    runs_ensemble = "LITS-499,LITS-500,LITS-501,LITS-502,LITS-503".split(",")
    runs_ensemble = "LITS-625"
    fldr = Path("/s/datasets_bkp/normal/sitk/images")
    fldr =  Path("/s/datasets_bkp/litq/complete_cases/images")
    device ='cuda'
# %%
    En = EnsemblePredictor(
        proj_defaults,
        3,
        run_name_w,
        runs_ensemble,
        bs=3,
        half=True,
        device=device,
        debug=True,
        overwrite=True,
    )

# %%
    # fldr  = Path("/media/ub2/datasets/drli/sitktmp/images/")
    # fnames = list(mo_df.image_filenames)
    # fname = "/media/ub2/datasets/drli/sitktmp/images/drli_048.nrrd"
    fnames = list(fldr.glob("*"))
    # for fname in fnames[1:]:
    #     fname = Path(fname)
    fname = Path("/home/ub/code/slicer_utils/files/images/litq_76_20210528.nii.gz")
    fname = [fn for fn in fnames if "litq_0014389_20190925" in fn.name][0]
    fname=Path("/s/xnat_shadow/crc/test/images/finalised/crc_CRC83b_20130726_Abdomen.nii.gz")
    # fname = fnames[0]
    img_sitk = sitk.ReadImage(fname)
# %%
    En.set_pred_fns(fname)
    En.run(img_sitk)
# %%
    ImageMaskViewer([En.p.img_np_orgres,En.pred[1]])
# %%
    En.save_prediction()
# %%
    En.unload_case()
# %%j
    En.sitk_props
# %%
    w=En.w
    img = w.img_transformed
    pred = w.pred.clone()
    pred = w.pred_patches[0].clone()
    pred = pred.detach().cpu()
    ImageMaskViewer([pred[0],pred[1]])
    ImageMaskViewer([img,img])
# %%
    pred = En.pred
    pred = pred.clone()
    pred = pred.detach().cpu()
    ImageMaskViewer([pred[0],pred[1]])
# %%
    p = En.p
    En.localiser_bbox(img_sitk)
    En.pred_patches = []
    for n in range(len(En.runs_p)):
        En.load_patch_model(n)
        En.patch_prediction(img_sitk, n)
        En.p.pred_patches= Unlist().encodes(En.p.pred_patches)
        En.pred_patches.append(En.p.pred_patches)
    En.pred_patches = pred_mean(En.pred_patches)
    En.backsample()
    # del En.pred_patches
    # gc.collect()
    En.postprocess(En.cc3d)

# %%
    (
            model_w,
            patch_size_w,
            resample_spacings_w,
            out_channels_w,
        ) = En.load_model_neptune(En.run_name_w, device="cpu")

# %%
    En.load_localiser_model()
    En.w.run(img_sitk=img_sitk, bboxes=None, save=False)
    pred_localiser = En.w.pred_sitk_i
    pred_localiser =  sitk.DICOMOrient(pred_localiser, "LPS") # patch to erect the localiser before bbox derivation.
    En.create_bboxes_from_localiser(pred_localiser)

# %%
    dataset_params = En.NepMan.run_dict['dataset_params']
    En.w = WholeImagePredictor(
        dataset_params = dataset_params,
        proj_defaults = En.proj_defaults,
        out_channels=out_channels_w,
        resample_spacings=resample_spacings_w,
        patch_size=patch_size_w,
        bs = En.bs,
        half = En.half,
        device=En.device,
        overwrite=En.overwrite,
    )
    En.w.load_model(model_w, model_id=En.run_name_w)



# %%
    w = En.w
    bboxes=None
    w.load_case(img_sitk, bboxes)
    if w.already_processed == False:
        w.create_encode_pipeline()
        w.create_decode_pipeline()
        w.create_postprocess_pipeline()

        w.img_transformed, w.bboxes_transformed = w.encode_pipeline(
                    [w.img_np_orgres, w.bboxes]
                )
        w.make_prediction()

        w.backsample()
        w.postprocess(w.cc3d)
   
# %%
    pre = w.make_prediction()
# %%
    img = w.img_transformed
    ImageMaskViewer([img,img])
# %%
    pred_sitk = w.pred_sitk_i
    pred_np = sitk.GetArrayFromImage(pred_sitk)
    ImageMaskViewer([pred_np,pred_np])
# %%
    im_inp = w.img_transformed

    im_inp =im_inp.unsqueeze(0).unsqueeze(0)
    torch.save(im_inp,"tmp_img.pt")
# %%
        # dataset_params = self.run_dict['dataset_params']
        config_dict = En.NepMan.download_run_params()
        resample_spacings = config_dict["dataset_params"]["spacings"]
        # if not   'out_channels' in model_params:
        #     oc = {'out_channels':  out_channels_from_dict_or_cell(En.NepMan.run_dict['metadata']['src_dest_labels'])}
        #     model_params['out_channels']  = out_channels_from_dict_or_cell(metadata['src_dest_labels'])
        out_channels = config_dict["model_params"]["out_channels"]
        patch_size = config_dict["dataset_params"]["patch_size"]

        model = create_model_from_conf(
            config_dict["model_params"],
            config_dict["dataset_params"],
            config_dict["metadata"],
            deep_supervision=False,
        )

        model_dir = config_dict["metadata"]["model_dir"]
        load_checkpoint(model_dir, model, device)
# %%
        checkpoints_folder=Path(model_dir)
# %%
        list_of_files = checkpoints_folder.glob('*')

# file = max(list_of_files, key=lambda p: p.stat().st_ctime)
        file = max(list_of_files, key=get_epoch)

        print("Loading last checkpoint {}".format(file))
        if isinstance(device, int): device = torch.device('cuda', device)
        elif device is None: device = 'cpu'
        state = torch.load(file, map_location=device)
        hasopt = set(state)=={'model', 'opt'}
        chkpt_model_state = state['model'] if hasopt else state
        chkpt_model_state = reconcile_keys(model.state_dict(),chkpt_model_state)
#
# chkpt_model_keys = list(chkpt_model_state.keys())
# conflicting_string = 'module.'
# conflicting_string in chkpt_model_keys[0]
# local_model_keys = list(model.state_dict().keys())
#
# mod_keys = conflicting_string in local_model_keys[0] 
# chk_keys =  conflicting_string in chkpt_model_keys[0]
# if not mod_keys == chk_keys:
#     chkpt_model_state_fixed = {}
#     for key in chkpt_model_state.keys():
#         print(key)
#         neo_key = key.replace(conflicting_string,'')
#         chkpt_model_state_fixed[neo_key] = chkpt_model_state[key]
#     get_model(model).load_state_dict(chkpt_model_state_fixed, strict=strict)
# else:
        get_model(model).load_state_dict(chkpt_model_state, strict=True)

        #

# %%
    img  = torch.load("tmp_img.pt")
    y = model(img)
    y = y[0].detach().cpu()
    ImageMaskViewer([img[0,0],y[1]])
