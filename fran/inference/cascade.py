# %%
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm
import gc
import ipdb
from monai.transforms.post.array import KeepLargestConnectedComponent
from monai.transforms.utility.array import ToTensor
from monai.transforms.utility.dictionary import SqueezeDimd
from fran.transforms.misc_transforms import SelectLabels
from fran.transforms.spatialtransforms import ResizeDynamicMetaKeyd, ResizeDynamicd

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


class SimpleTrainer(UNetTrainer):
    def test_step(self, batch, batch_idx):
        img = batch["image"]
        outputs = self.forward(img)
        outputs2 = outputs[0]
        batch["pred"] = outputs2
        # output=outputs[0]
        # outputs = {'pred':output,'org_size':batch['org_size']}
        # outputs_backsampled=self.post_process(outputs)
        return batch


class ImageBBoxDataset(Dataset):
    def __init__(self, data, transform: Union[Callable, None] = None) -> None:
        self.ds, self.bboxes = data
        self.transform = transform

    def __getitem__(self, idx):
        im = self.ds[idx]["image"]
        bbox = self.bboxes[idx]
        img_c = im[bbox]
        outputs = {"image": im, "image_cropped": img_c, "bbox": bbox}
        if self.transform:
            outputs = self.transform(outputs)
        return outputs

    def __len__(self):
        return len(self.ds)

    


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


class PersistentDS(PersistentDataset):
    def __init__(self, imgs: Union[torch.Tensor, list], cache_dir) -> None:
        L = LoadImaged(
            keys=["image"], image_only=True, ensure_channel_first=True, simple_keys=True
        )
        # E = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")
        O = Orientationd(keys=["image"], axcodes="RAS")
        tfms = Compose([L, O])
        self.cache_dir = cache_dir
        super().__init__(imgs, tfms)

    def create_batch_transforms(self):
        # T = TransposeSITKd(keys=["image", "org_size"])
        # E = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )
        # S = Spacingd(keys=["image"], pixdim=self.dataset_params["spacing"])
        C = Compose([N], lazy=True)
        self.batch_transforms = C


class PatchDM(DataManager):
    def create_batch_transforms(self):
        T = TransposeSITKd(keys=["image", "org_size"])
        E = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )
        S = Spacingd(keys=["image"], pixdim=self.dataset_params["spacing"])
        C = Compose([T, E, N, S], lazy=True)
        self.transforms = C

    def patch_batchmaker(self, batch):  # note batch_size = 1
        imgs = []
        R = Resized(keys=["image"], spatial_size=patch_size)
        tfms = Compose([R])
        for i, item in enumerate(batch):
            img = tfms(item)["image"]
            imgs.append(img)
        output = {"image": torch.stack(imgs, 0)}
        return output

    def create_dataloader(self):
        pass


class WholeImageInferer(BaseInferer):
    def __init__(
        self,
        project,
        run_name,
        devices=[1],
        debug=True,
        save=True,
        overwrite=False,
    ):
        """
        data is a dataset from Ensemble in this base class
        """

        store_attr("project,run_name,devices,debug,overwrite,save")
        self.ckpt = checkpoint_from_model_id(run_name)
        self.dataset_params = load_dataset_params(run_name)
        # dic1 = torch.load(self.ckpt)
        # dic2 = {}
        # relevant_keys = ["datamodule_hyper_parameters"]
        # for key in relevant_keys:
        #     dic2[key] = dic1[key]
        #     self.assimilate_dict(dic2[key])


    def setup(self,data):
        self.prepare_model()
        self.prepare_data(data)
        self.create_postprocess_transforms()

    def create_postprocess_transforms(self):
        I = Invertd(keys=["pred"], transform=self.ds2.transform, orig_keys=["image"])
        Rz = ResizeDynamicMetaKeyd(
                keys=["pred"], key_spatial_size="spatial_shape", mode="nearest"
            )
        D = AsDiscreted(keys=["pred"], argmax=True, threshold=0.5)
        K = KeepLargestConnectedComponentd(keys=["pred"])
        C = ToCPUd(keys=["image", "pred"])
        B = BBoxFromPTd(
            keys=["pred"], expand_by=20, spacing=self.dataset_params["spacing"]
        )
        tfms = [ D,Rz, K, C, B]
        if self.debug == True:
            Sa = SaveMultiChanneld(
                keys=["pred"],
                output_dir=self.output_folder,
                output_postfix="",
                separate_folder=False,
            )
            #
            # Sa = SavePatchd(keys=['pred'],output_folder=self.output_folder,postfix_channel=True)
            tfms.insert(1, Sa)
        C = Compose(tfms)
        self.postprocess_transforms = C

    def create_transforms(self):
        self.R = Resized(keys=["image"], spatial_size=self.dataset_params["patch_size"])
        self.N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )

    # def prepare_model(self):
    #     self.model = UNetTrainer.load_from_checkpoint(
    #         self.ckpt,
    #         project=self.project,
    #         dataset_params=self.dataset_params,
    #         strict=False,
    #     )
    #     self.model.eval()
    #     fabric = Fabric(precision="16-mixed", devices=self.devices)
    #     self.model = fabric.setup(self.model)
    #
    # def predict(self):
    #     outputs = []
    #     self.model.eval()
    #     with torch.inference_mode():
    #         for i, batch in enumerate(self.pred_dl):
    #             img = batch["image"].cuda()
    #             if 'filename_or_obj' in img.meta.keys():
    #                 print("Processing: ",img.meta['filename_or_obj'])
    #             output = self.model(img)
    #             output = output[0]
    #             batch["pred"] = output
    #             batch["pred"].meta = batch["image"].meta
    #             outputs.append(batch)
    #     return outputs

    def postprocess(self, preds):
        out_final = []
        for batch in preds:
            out2 = decollate_batch(batch, detach=True)
            for ou in out2:
                tmp = self.postprocess_transforms(ou)
                out_final.append(tmp)
        return out_final

class PatchInferer(BaseInferer):
    def __init__(
        self,
        project,
        run_name,
        patch_overlap=0.25,
        bs=1,
        grid_mode="gaussian",
        devices=[1],
        debug=True,
        overwrite=False,
    ):
        super().__init__(project=project, run_name=run_name, devices= devices, debug=debug,overwrite= overwrite,save=False)


    def create_postprocess_transforms(self):
        Sq = SqueezeDimd(keys=["pred"], dim=0)
        I = Invertd(
            keys=["pred"], transform=self.ds.transform, orig_keys=["image"]
        )  # watchout: use detach beforeharnd. make sure spacing are correct in preds
        C = ToCPUd(keys=["image", "pred"])
        Sa = SaveMultiChanneld(
            keys=["pred"],
            output_dir=self.output_folder,
            output_postfix="",
            separate_folder=False,
        )

        tfms = [Sq, I, C]
        if self.debug == True:
            tfms = [Sq,I,Sa,C]
        C = Compose(tfms)
        self.postprocess_transforms = C

class CascadeInferer:  # SPACING HAS TO BE SAME IN PATCHES
    def __init__(
        self,
        project,
        run_name_w,
        runs_p,
        localiser_labels:list ,  #these labels will be used to create bbox
        devices=[0],

        overwrite_w=False,
        overwrite_p=True,
        profile=None,
        debug=False,
        save=True,
        k_largest=None, # assign a number if there are organs involved
    ):
        """
        Creates a single dataset (cascade dataset) which normalises images once for both patches and whole images. Hence, the model should be developed from the same dataset std, mean values.
        """
        assert profile in [None, "dataloading", "prediction", "all"], "Choose one of None , 'dataloading', 'prediction', 'all'"

        self.predictions_folder = project.predictions_folder
        self.dataset_params = load_dataset_params(runs_p[0])
        self.Ps = [PatchInferer(project=project,run_name=run, devices=devices, debug=debug, overwrite=overwrite_p) for run in runs_p]
        self.localiser_tfms= "ESN"
        WSInf = self.inferer_from_params(run_name_w)
        self.W = WSInf( project=project, run_name=run_name_w,  debug=debug, devices=devices, overwrite=overwrite_w
        )
        store_attr()

    def inferer_from_params(self,run_name_w):
          self.ckpt = checkpoint_from_model_id(run_name_w)
          dic1 = torch.load(self.ckpt)
          mode = dic1['datamodule_hyper_parameters']['dataset_params']['mode']
          if mode == "source":
            return BaseInferer
          elif mode == "whole":
            return WholeImageInferer

    def parse_input(self, imgs_inp):
        """
        input types:
            folder of img_fns
            nifti img_fns
            itk imgs (slicer)
        returns list of img_fns if folder. Otherwise just the imgs
        """

        if not isinstance(imgs_inp, list):
            imgs_inp = [imgs_inp]
        imgs_out = []
        for dat in imgs_inp:
            if any([isinstance(dat, str), isinstance(dat, Path)]):
                self.input_type = "files"
                dat = Path(dat)
                if dat.is_dir():
                    dat = list(dat.glob("*"))
                else:
                    dat = [dat]
            else:
                self.input_type = "itk"
                if isinstance(dat, sitk.Image):
                    dat = ConvertSimpleItkImageToItkImage(dat, itk.F)
                # if isinstance(dat,itk.Image):
                dat = itm(dat)
            imgs_out.extend(dat)
        imgs_out = [{"image": img} for img in imgs_out]
        return imgs_out



    def load_images(self, data):
        """
        data can be filenames or images. InferenceDatasetNii will resolve data type and add LoadImaged if it is a filename
        """

        Loader = LoadSITKd(['image'])
        data = self.parse_input(data)
        data = [Loader(d) for d in data]

        return data

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

        if self.overwrite_p==False and (isinstance(imgs[0],str) or isinstance(imgs[0], Path)):
            imgs = self.filter_existing_preds(imgs)
        else:
            pass
            # self.save = False  # don't save if input is pure images. Just output those.
        if len(imgs)>0:
            imgs = list_to_chunks(imgs, chunksize)
            for imgs_sublist in imgs:
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
        else: return 1


    def apply_bboxes(self,data,bboxes):

        data2=[]
        for i, dat in enumerate(data):
            dat['image'] = dat['image'][En.bboxes[i][1:]]
            dat['bounding_box'] = En.bboxes[i]
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
        if self.overwrite_w==False:
            print("Bbox overwrite not implemented yet")
        print("Starting localiser data prep and prediction")
        self.W.setup(data,self.localiser_tfms)
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
            P.setup(data=data,tfms='ESN', collate_fn=img_bbox_collated)
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
    project = Project(project_title="totalseg")

    run_w = "LIT-145"
    run_ps = ["LIT-143", "LIT-150", "LIT-149", "LIT-153", "LIT-161"]
    run_ps = ["LITS-630", "LITS-633", "LITS-632", "LITS-647", "LITS-650"]
    run_ps = ["LITS-720"]

    run_ps = ["LITS-787", "LITS-810", "LITS-811"]
    run_ps = ["LITS-709"]
    run_lidc2 = ["LITS-842"]
    run_ts= ["LITS-827"]
# %%
    img_fna = "/s/xnat_shadow/litq/test/images_ub/"
    fns = "/s/datasets_bkp/drli_short/images/"
# %%
    img_fldr= Path("/s/xnat_shadow/lidc2/images/")
    img_fn2= "/s/xnat_shadow/crc/wxh/images/crc_CRC198_20170718_CAP1p51.nii.gz"


    imgs_fldr = Path("/s/xnat_shadow/crc/images")
# %%
    srn_fldr = "/s/xnat_shadow/crc/srn/cases_with_findings/images/"
    srn_imgs = list(Path(srn_fldr).glob("*"))
    wxh_fldr = "/s/xnat_shadow/crc/wxh/completed/"
    wxh_imgs = list(Path(wxh_fldr).glob("*"))
    litq_fldr = "/s/xnat_shadow/litq/test/images_ub/"
    litq_imgs = list(Path(litq_fldr).glob("*"))
# %%
    run_w = run_ts[0]
    localiser_labels =[1]
    localiser_labels =[45,46,47,48,49]
    runs_p = run_ps
    En = CascadeInferer(project, run_w, run_lidc2, debug=False, devices=[0],overwrite_w=False,overwrite_p=True,localiser_labels=localiser_labels)

# %%
    # img_fns = ["/s/xnat_shadow/lidc2/images/lidc2_0001.nii.gz","/s/xnat_shadow/lidc2/images/lidc2_0003.nii.gz"]
    img_fns = list(img_fldr.glob("*"))[20:50]
    preds = En.run(img_fns)

# %%
# %%
    imgs_sublist = img_fns
    data1 = En.load_images(imgs_sublist)
    En.bboxes = En.extract_fg_bboxes(data1)
    data = En.apply_bboxes(data1,En.bboxes)
    pred_patches = En.patch_prediction(data )
    pred_patches = En.decollate_patches(pred_patches, En.bboxes)


    output = En.postprocess(pred_patches)
# %%    
    img = data1[0]['image']
    pp = pred_patches['LITS-842']
    image = pp[0]['image'][0,0]
    pred = pp[0]['pred']
# %%

    keys = En.runs_p
    A = Activationsd(keys="pred", softmax=True)
    D = AsDiscreted(keys=["pred"], argmax=True)
    K = KeepLargestConnectedComponentWithMetad(
        keys=["pred"], independent=False, 
    )  # label=1 is the organ
    F = FillBBoxPatchesd()
    if len(keys) == 1:
        MR = RenameDictKeys(new_keys=["pred"], keys=keys)
    else:
        MR = MeanEnsembled(output_key="pred", keys=keys)
    if En.k_largest:
        tfms = [MR, A, D, K, F]
    else:
        tfms = [MR, A, D,  F]
    # S = SaveListd(keys = ['pred'],output_dir=self.output_folder,output_postfix='',separate_folder=False)


    C = Compose(tfms)

# %%
    ind = 1
    pred =pred_patches[ind]
    pred= MR(pred)
    pred= A(pred)
    pred= D(pred)
# %%
    ind=1
    pr= output[ind]['pred']
    im = data[ind]['image']
    bbox = pred['bounding_box']
    img = im.permute(2,1,0)
    pr = pr[0].permute(2,1,0)
    ImageMaskViewer([img,pr.cpu()])

# %%

# %%
    for p in En.Ps:
        del p.model
    torch.cuda.empty_cache()


# %%
