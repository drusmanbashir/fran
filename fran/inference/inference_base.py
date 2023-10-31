# %%
from time import time
from monai.data import MetaTensor

from lightning.fabric import Fabric
import lightning.pytorch as pl
from collections.abc import Callable
from monai.config.type_definitions import KeysCollection
from monai.data.image_writer import ITKWriter
from monai.data.utils import decollate_batch
from monai.data.box_utils import *
from monai.inferers.merger import *
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset, PersistentDataset
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm, metatensor_to_itk_image
from monai.transforms import (
    LoadImage,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    Spacingd,
    Invertd,
)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureTyped
from fran.data.dataloader import img_metadata_collated
from fran.utils.dictopts import DictToAttr
from fran.utils.itk_sitk import *
from monai.transforms.croppad.dictionary import BoundingRectd
from monai.apps.detection.transforms.array import *
from monai.transforms.post.dictionary import Activationsd, KeepLargestConnectedComponentd, MeanEnsembled
from monai.data.box_utils import *
from monai.transforms.transform import MapTransform, Transform
from monai.transforms.spatial.dictionary import Orientationd, Resized
from monai.transforms.utility.array import EnsureType
from fran.data.dataset import FillBBoxPatches, NormaliseClipd, NormaliseClip
from fran.managers.training import DataManager, nnUNetTrainer
from fran.utils.common import *
import sys
import gc
from monai.inferers.utils import sliding_window_inference
from monai.inferers import SlidingWindowInferer
from torch.functional import Tensor
import numpy as np
from torchvision.transforms.functional import resize

from fran.utils.fileio import load_dict, load_yaml, maybe_makedirs
import SimpleITK as sitk
from pathlib import Path
from fran.utils.helpers import (
    get_available_device,
    timing,
)
from fran.utils.string import drop_digit_suffix

sys.path += ["/home/ub/code"]
from mask_analysis.helpers import to_int, to_label, to_cc

# These are the usual ipython objects, including this one you are creating
ipython_vars = ["In", "Out", "exit", "quit", "get_ipython", "ipython_vars"]
from fastcore.foundation import L, Union, listify, operator
from fastcore.all import GetAttr, ItemTransform, Pipeline, Sequence
from fran.transforms.intensitytransforms import ClipCenterI
from monai.transforms.post.array import Activations, Invert, KeepLargestConnectedComponent, VoteEnsemble,AsDiscrete
import os


import sys

sys.path += ["/home/ub/Dropbox/code/fran/"]
from fastcore.transform import Transform as TFC
import functools as fl

from fastcore.basics import store_attr

from fran.utils.imageviewers import ImageMaskViewer
import torch.nn.functional as F


class Saved(Transform):
        def __init__(self,output_folder):self.output_folder = output_folder
        def __call__(self,patch_bundle):
             

            maybe_makedirs(self.output_folder)
            try:
                fl=Path(patch_bundle['image'].meta['filename_or_obj']).name
            except:
                fl="_tmp.nii.gz"
            outname = self.output_folder/fl

            meta = {'original_affine': patch_bundle['image'].meta['original_affine'],
                    'affine':patch_bundle['image'].meta['affine']}

            writer = ITKWriter()

            writer.set_data_array(patch_bundle['pred'])
            writer.set_metadata(meta)
            assert(di:=patch_bundle['pred'].dim())==4,"Dimension should be 4. Got {}".format(di)
            writer.write(outname)


def checkpoint_from_model_id(model_id):
    common_paths = load_yaml(common_vars_filename)
    fldr = Path(common_paths["checkpoints_parent_folder"])
    all_fldrs = [
        f for f in fldr.rglob("*{}/checkpoints".format(model_id)) if f.is_dir()
    ]
    if len(all_fldrs) == 1:
        fldr = all_fldrs[0]
    else:
        tr()

    list_of_files = list(fldr.glob("*"))
    ckpt = max(list_of_files, key=lambda p: p.stat().st_ctime)
    return ckpt

def transpose_bboxes(bbox):
    bbox = bbox[2], bbox[1], bbox[0]
    return bbox

#
class TransposeSITKd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def func(self, x):
        if isinstance(x, torch.Tensor):
            x = torch.permute(x, [2, 1, 0])
        elif isinstance(x, Union[tuple, list]):
            x = transpose_bboxes(x)
        else:
            raise NotImplemented
        return x

    def __call__(self, d: dict):
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
        return d


class SlicesFromBBox(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def func(self, x):
        b = x[0] # get rid of channel
        slices=[slice(0,100)] # 100 channels
        for ind in [0,2,4]:
            s = slice(b[ind],b[ind+1])
            slices.append(s)
        return tuple(slices)



    def __call__(self, d: dict):
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
        return d


class SimpleTrainer(nnUNetTrainer):
    def test_step(self,batch,batch_idx):

        img = batch['image']
        outputs=self.forward(img)
        outputs2=outputs[0]
        batch['pred']=outputs2
        # output=outputs[0]
        # outputs = {'pred':output,'org_size':batch['org_size']}
        # outputs_backsampled=self.post_process(outputs)
        return batch



class ImageBBoxDataset(Dataset):
    def __init__(
        self, data,  transform: Union[Callable , None ]= None
    ) -> None:
        self.ds,self.bboxes=data
        self.transform=transform

    def __getitem__(self, idx):
        im = self.ds[idx]['image']
        bbox = self.bboxes[idx]
        img_c = im[bbox]
        outputs = {'image':im,'image_cropped':img_c,'bbox':bbox}
        if self.transform:
            outputs = self.transform(outputs)
        return outputs
    def __len__(self): return len(self.ds)


def img_bbox_collated( batch):
        imgs=[]
        imgs_c= []
        bboxes=[]
        for i , item in enumerate(batch):
            imgs.append(item['image'])
            imgs_c.append(item['image_cropped'])
            bboxes.append(item['bbox'])
        output = {'image':torch.stack(imgs,0),'image_cropped':torch.stack(imgs_c,0),'bbox':bboxes}
        return output



class PersistentDS(PersistentDataset):
    def __init__(self, imgs: Union[torch.Tensor, list],cache_dir) -> None:

        L=LoadImaged(keys=['image'],image_only=True,ensure_channel_first=True,simple_keys=True)
        # E = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")
        O = Orientationd(keys=['image'], axcodes="RAS")
        tfms = Compose([L,O])
        self.cache_dir=cache_dir
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
        # S = Spacingd(keys=["image"], pixdim=self.dataset_params["spacings"])
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
        S = Spacingd(keys=["image"], pixdim=self.dataset_params["spacings"])
        C = Compose([T, E, N, S], lazy=True)
        self.transforms = C
    
    def patch_batchmaker(self,batch): # note batch_size = 1
            imgs=[]

            R = Resized(
                keys=["image"],
                spatial_size=patch_size
            )
            tfms = Compose([R])
            for i , item in enumerate(batch):
                img = tfms(item)['image']
                imgs.append(img)
            output = {'image':torch.stack(imgs,0)}
            return output




    def create_dataloader(self):
        pass

class WholeImagePredictor(GetAttr,DictToAttr):
    _default = "datamodule"

    def __init__(self, project,run_name, data,devices=1,save_preds=True):
        '''
        data is a dataset from Ensemble in this base class
        '''

        store_attr('project,run_name,devices,save_preds')
        self.ckpt = checkpoint_from_model_id(run_name)
        dic1=torch.load(self.ckpt)
        dic2={}
        relevant_keys=['datamodule_hyper_parameters']
        for key in relevant_keys:
            dic2[key]=dic1[key]
            self.assimilate_dict(dic2[key])
    
        self.prepare_model()
        self.prepare_data(data)

    def create_postprocess_transforms(self,ds_transform):
        I = Invertd(keys=['pred'],transform=ds_transform,orig_keys=['image'])
        D = AsDiscreted(keys=['pred'],argmax=True,threshold=0.5)
        K= KeepLargestConnectedComponentd(keys=['pred'])
        B=BoundingRectd(keys=['pred'])
        S = SlicesFromBBox(keys=['pred_bbox'])
        tfms = [I,D,K,B,S]
        if self.save_pred==True:
            S = Saved(self.output_folder)
            tfms.insert(1,S)
        C = Compose(tfms)
        self.postprocess_transforms=C

    def prepare_data(self,ds ):
        R = Resized(
            keys=["image"],
            spatial_size=self.dataset_params['patch_size']
        )
        N= NormaliseClipd(keys=['image'],clip_range= self.dataset_params['intensity_clip_range'],mean=self.dataset_params['mean_fg'],std=self.dataset_params['std_fg'])
        tfm = Compose([R,N])
        self.ds2=Dataset(data=ds,transform=tfm)
        self.pred_dl = DataLoader(
                self.ds2, num_workers=12, batch_size=12
            )


    def prepare_model(self):
        self.model = nnUNetTrainer.load_from_checkpoint(
            self.ckpt, project=self.project, dataset_params=self.dataset_params, strict=False
        )


    def predict(self):
        outputs=[]
        self.model.eval()
        with torch.inference_mode():
            for i ,batch in enumerate(self.pred_dl):
                img = batch['image'].cuda()
                output = self.model(img)
                output=output[0]
                batch['pred']= output
                outputs.append(batch)
        return outputs

    def postprocess(self, preds):
        out_final=[]
        for batch in preds:
            out2 = decollate_batch(batch,detach=True)
            for ou in out2:
                tmp=self.postprocess_transforms(ou) 
                out_final.append(tmp)
        return out_final

    @property
    def output_folder(self):
        fldr='_'.join(self.run_name)
        fldr = self.project.predictions_folder/fldr
        return fldr



class PatchPredictor(WholeImagePredictor):
    def __init__(self,project, run_name,  data,patch_overlap=0.25,bs=8,grid_mode="gaussian",devices=1 ):
        super().__init__(project,run_name,  data,devices)
        '''
        data is a list containing a dataset from ensemble and bboxes
        '''

        self.grid_mode = grid_mode
        self.patch_size = self.dataset_params["patch_size"]
        self.inferer = SlidingWindowInferer(
            roi_size=self.dataset_params['patch_size'],
            sw_batch_size=bs,
            overlap=patch_overlap,
            mode=grid_mode,
            progress=True,
        )


    def prepare_data(self, data):

        S = Spacingd(keys=["image_cropped"], pixdim=self.dataset_params['spacings'])
        N = NormaliseClipd(
            keys=["image_cropped"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )

        # E = EnsureTyped(keys=["image" ], device="cuda", track_meta=True),
        tfm = Compose([N,S])
        self.ds2=ImageBBoxDataset(data,transform=tfm)
        self.pred_dl = DataLoader(
                self.ds2, num_workers=1, batch_size=1, collate_fn = img_bbox_collated # essential to avoid size mismatch
            )
        
    def predict(self):
        outputs = []
        for i ,batch in enumerate(self.pred_dl):
                with torch.no_grad():
                    img_input=batch['image_cropped']
                    img_input = img_input.cuda()
                    output_tensor = self.inferer(inputs=img_input, network=self.model)
                    output_tensor = output_tensor[0]
                    batch['pred']=output_tensor
                    outputs.append(batch)
        return outputs

    def postprocess(self, outputs):
        I = Invertd(keys=['pred'],transform=self.ds2.transform,orig_keys=['image_cropped'])
        out_final=[]
        for batch in outputs: # batch_length is 1
            batch['pred']=batch['pred'].squeeze(0).detach()
            batch=I(batch)
            if self.save_pred==True:
                        S = Saved(self.output_folder)
                        batch=S(batch)
            out_final.append(batch)
        return out_final


    def prepare_model(self):
        super().prepare_model()
        self.model.eval()
        fabric = Fabric(precision="16-mixed",devices=self.devices)
        self.model=fabric.setup(self.model)


class EnsemblePredictor():  # SPACING HAS TO BE SAME IN PATCHES
    def __init__(
        self,
        project,
        run_name_w,
        runs_p,
        device="cuda",
        debug=False,
        overwrite=False,
    ):
        
        self.predictions_folder = project.predictions_folder
        store_attr()


    def create_ds(self,im):
        L=LoadImaged(keys=['image'],image_only=True,ensure_channel_first=True,simple_keys=True)
        O = Orientationd(keys=['image'], axcodes="RAS")
        tfms = Compose([L,O])

        cache_dir = self.project.cold_datasets_folder/("cache")
        maybe_makedirs(cache_dir)
        self.ds=PersistentDataset(data = im,transform=tfms, cache_dir =cache_dir )

    def get_patch_spacings(self,run_name):
            ckpt = checkpoint_from_model_id(run_name)
            dic1=torch.load(ckpt)
            spacings = dic1['datamodule_hyper_parameters']['dataset_params']['spacings']
            return spacings

    def run(self,imgs):
        imgs =self.parse_input(imgs)
        self.create_ds(imgs)
        self.bboxes= self.extract_fg_bboxes()
        pred_patches = self.patch_prediction(self.ds,self.bboxes)
        pred_patches = self.decollate_patches(pred_patches,self.bboxes)
        output= self.postprocess(pred_patches)
        return output
        # output=[]
        # for i in range(num_cases):
        #     patch_bundle = self.get_mini_bundle(patch_bundles,i)
        #     preds = self.postprocess(patch_bundle)
        #     output.append(preds)
        #     self.save_pred(preds)
        # return output
        #
    def parse_input(self,imgs_inp):
        '''
        input types:
            folder of img_fns
            nifti img_fns 
            itk imgs (slicer)

        returns list of img_fns if folder. Otherwise just the imgs
        '''

        if not isinstance(imgs_inp,list): imgs_inp=[imgs_inp]
        imgs_out = []
        for dat in imgs_inp:
            if any([isinstance(dat,str),isinstance(dat,Path)]):
                self.input_type= 'files'
                dat = Path(dat)
                if dat.is_dir():
                    dat = list(dat.glob("*"))
                else:
                    dat=[dat]
            else:
                self.input_type= 'itk'
                if isinstance(dat,sitk.Image):
                    dat= ConvertSimpleItkImageToItkImage(dat, itk.F)
                # if isinstance(dat,itk.Image):
                dat=itm(dat) 
            imgs_out.extend(dat)
        imgs_out = [{'image':img} for img in imgs_out]
        return imgs_out

    def get_mini_bundle(self,patch_bundles,indx):
            patch_bundle={}
            for key,val in patch_bundles.items():
                pred_patch={key:val[indx]}
                patch_bundle.update(pred_patch)
            return patch_bundle

    def decollate_patches(self,pa,bboxes):
        num_cases = len(self.ds)
        keys = self.runs_p
        keys = En.runs_p
        output=[]
        for case_idx in range(num_cases):
            img_bbox_preds={}
            for i,run_name in enumerate(keys):
                pred =  pa[run_name][case_idx]['pred']
                img_bbox_preds[run_name]= pred
            img_bbox_preds.update(self.ds[case_idx])
            img_bbox_preds['bbox']=bboxes[case_idx]
            output.append( img_bbox_preds)
        return output


    def save_pred(self,pred):
            S=Saved(self.output_folder)
            S(pred)

    def extract_fg_bboxes(self):
        w=WholeImagePredictor(self.project,self.run_name_w,self.ds)
        print("Preparing data")
        w.create_postprocess_transforms(w.ds2.transform)
        p = w.predict()
        preds= w.postprocess(p)
        bboxes = [pred['pred_bbox'] for pred in preds]
        return bboxes
    
    def patch_prediction(self,ds,bboxes):
        data = [ds,bboxes]
        preds_all_runs={}
        for run in self.runs_p:
            p=PatchPredictor(self.project,run,data=data)
            preds=p.predict()
            preds = p.postprocess(preds)
            preds_all_runs[run] =preds
        return preds_all_runs

    @property
    def output_folder(self):
        fldr='_'.join(self.runs_p) 
        fldr = self.predictions_folder/fldr
        return fldr


    def postprocess(self,patch_bundle):
        keys= self.runs_p
        M = MeanEnsembled(keys=keys,output_key="pred")
        A = Activationsd(keys="pred", softmax=True)
        D = AsDiscreted(keys=['pred'],argmax=True)
        K = KeepLargestConnectedComponentd(keys=['pred'],independent=False)
        F = FillBBoxPatches()
        S = Saved(self.output_folder)
        C  = Compose([M,A,D,K,F,S])
        output= C(patch_bundle)
        return output


# %%

if __name__ == "__main__":
    # ... run your application ...

    proj= Project(project_title="lits32")

# %%
    run_w='LIT-41'
    run_ps=['LIT-62','LIT-63','LIT-64' ,'LIT-44','LIT-59']
# %%
    img_fn = "/s/xnat_shadow/litq/test/images_few/litq_35_20200728.nii.gz"
    img_fn3 = "/s/xnat_shadow/litq/test/images_few/"
    img_fn2 = "/s/insync/datasets/crc_project/qiba/qiba0_0000.nii.gz"
    paths = [img_fn, img_fn2]
    img_fns = listify(img_fn3)





# %%
    En=EnsemblePredictor(proj,run_w,run_ps)
    # im = [{'image':im} for im in [img_fn,img_fn2]]
    
    preds=En.run(img_fns)

# %%
    a = En.ds[0]
    im = a['image']
# %%
    a= [1,2,3]
