
# %%
from time import time
import monai
from monai.data import MetaTensor, image_writer
from monai.transforms.utils import generate_spatial_bounding_box

from fran.managers.training import checkpoint_from_model_id
from lightning.fabric import Fabric
import lightning.pytorch as pl
from collections.abc import Callable
from monai.config.type_definitions import DtypeLike, KeysCollection
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
from monai.transforms.io.array import SaveImage
from monai.transforms.io.dictionary import LoadImaged, SaveImaged
from monai.transforms.utility.dictionary import EnsureTyped
from monai.utils.enums import GridSamplePadMode
from fran.data.dataloader import img_metadata_collated
from fran.utils.dictopts import DictToAttr
from fran.utils.itk_sitk import *
from monai.transforms.croppad.dictionary import BoundingRectd, ResizeWithPadOrCropd
from monai.apps.detection.transforms.array import *
from monai.transforms.post.dictionary import Activationsd, KeepLargestConnectedComponentd, MeanEnsembled
from monai.data.box_utils import *
from monai.transforms.transform import MapTransform, Transform
from monai.transforms.spatial.dictionary import Orientationd, Resized
from monai.transforms.utility.array import EnsureType
from fran.data.dataset import FillBBoxPatches, NormaliseClipd, NormaliseClip, SaveMultiChanneld
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

def slice_list(listi,start_end:list):
    return listi[start_end[0]:start_end[1]]

def list_to_chunks(input_list:list,chunksize:int):

    n_lists = int(np.ceil(len(input_list)/chunksize))

    fpl= int(len(input_list)/n_lists)
    inds = [[fpl*x,fpl*(x+1)] for x in range(n_lists-1)]
    inds.append([fpl*(n_lists-1),None])

    chunks = list(il.starmap(slice_list,zip([input_list]*n_lists,inds)))
    return chunks






class Saved(Transform):
        def __init__(self,output_folder):self.output_folder = output_folder
        def __call__(self,patch_bundle):
             
            key='pred'

            maybe_makedirs(self.output_folder)
            try:
                fl=Path(patch_bundle['image'].meta['filename_or_obj']).name
            except:
                fl="_tmp.nii.gz"
            outname = self.output_folder/fl

            meta = {'original_affine': patch_bundle['image'].meta['original_affine'],
                    'affine':patch_bundle['image'].meta['affine']}

            writer = ITKWriter()

            array_full = patch_bundle[key].detach().cpu()
            array_full.meta = patch_bundle['image'].meta
            channels = array.shape[0]
                # ch=0
                # array = array_full[ch:ch+1,:]
            writer.set_data_array(array)
            writer.set_data_array(patch_bundle['image'])
            writer.set_metadata(meta)
            assert(di:=patch_bundle[key].dim())==4,"Dimension should be 4. Got {}".format(di)
            writer.write(outname)


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

class ToCPUd(MapTransform):
    def __call__(self, d: dict):
            for key in self.key_iterator(d):
                d[key] = d[key].cpu()
            return d


class BBoxFromPred(MapTransform):
    def __init__(
        self,
        spacings,
        expand_by:int,  # in millimeters
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        store_attr('spacings,expand_by')

    def __call__(self,d:dict):
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
        return d

    def func(self,pred):
        add_to_bbox=  [ int(self.expand_by/sp) for sp in self.spacings]
        bb = generate_spatial_bounding_box(pred,channel_indices=0,margin=add_to_bbox)
        sls = [slice(0,100,None)]+[slice(a,b,None) for a,b in zip(*bb)]
        pred.meta['bounding_box']=sls
        return pred

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

    def __init__(self, project,run_name, data,devices=1,debug=True):
        '''
        data is a dataset from Ensemble in this base class
        '''

        store_attr('project,run_name,devices,debug')
        self.ckpt = checkpoint_from_model_id(run_name)
        dic1=torch.load(self.ckpt)
        dic2={}
        relevant_keys=['datamodule_hyper_parameters']
        for key in relevant_keys:
            dic2[key]=dic1[key]
            self.assimilate_dict(dic2[key])
    
        self.prepare_model()
        self.prepare_data(data)
        self.create_postprocess_transforms()

    def create_postprocess_transforms(self):
        I = Invertd(keys=['pred'],transform=self.ds2.transform,orig_keys=['image'])
        D = AsDiscreted(keys=['pred'],argmax=True,threshold=0.5)
        K= KeepLargestConnectedComponentd(keys=['pred'])
        C = ToCPUd(keys=['image','pred'])
        B=BBoxFromPred(keys=['pred'],expand_by=20,spacings=self.dataset_params['spacings'])
        tfms = [I,D,K,C,B]
        if self.debug==True:
            Sa = SaveMultiChanneld(keys=['pred'],output_folder=self.output_folder,postfix_channel=True)
            tfms.insert(1,Sa)
        C = Compose(tfms)
        self.postprocess_transforms=C

    def prepare_data(self,ds ):
        
        R = Resized(
            keys=["image"],
            spatial_size=self.dataset_params['patch_size']
        )

        # R2 = ResizeWithPadOrCropd(
        #         keys=["image"],
        #         source_key="image",
        #         spatial_size=self.dataset_params["patch_size"],
        #     )
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
        run_name = listify(self.run_name)
        fldr='_'.join(run_name)
        fldr = self.project.predictions_folder/fldr
        return fldr



class PatchPredictor(WholeImagePredictor):
    def __init__(self,project, run_name,  data,patch_overlap=0.25,bs=8,grid_mode="gaussian",devices=1,debug=True ):
        super().__init__(project,run_name,  data,devices,debug)
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
                self.ds2, num_workers=0, batch_size=1, collate_fn = img_bbox_collated # essential to avoid size mismatch
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
        C =  ToCPUd(keys=['image','pred'])
        tfms = [I,C]
        if self.debug==True:
                        S = SaveMultiChanneld(['pred'],self.output_folder,postfix_channel=True)
                        tfms+=[S]
        C = Compose(tfms)
        out_final=[]
        for batch in outputs: # batch_length is 1
            batch['pred']=batch['pred'].squeeze(0).detach()
            batch=C(batch)
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
        save=True
    ):
        
        self.predictions_folder = project.predictions_folder
        store_attr()


    def create_ds(self,im):
        L=LoadImaged(keys=['image'],image_only=True,ensure_channel_first=True,simple_keys=True)
        O = Orientationd(keys=['image'], axcodes="RPS") # nOTE RPS
        tfms = Compose([L,O])

        cache_dir = self.project.cold_datasets_folder/("cache")
        maybe_makedirs(cache_dir)
        self.ds=PersistentDataset(data = im,transform=tfms, cache_dir =cache_dir )

    def get_patch_spacings(self,run_name):
            ckpt = checkpoint_from_model_id(run_name)
            dic1=torch.load(ckpt)
            spacings = dic1['datamodule_hyper_parameters']['dataset_params']['spacings']
            return spacings

    def run(self,imgs,chunksize=12):
        '''
        chunksize is necessary in large lists to manage system ram
        '''
        
        imgs =self.parse_input(imgs)
        imgs  = list_to_chunks(imgs,chunksize)
        for imgs_sublist in imgs:
            self.create_ds(imgs_sublist)
            self.bboxes= self.extract_fg_bboxes()
            pred_patches = self.patch_prediction(self.ds,self.bboxes)
            pred_patches = self.decollate_patches(pred_patches,self.bboxes)
            output= self.postprocess(pred_patches)
            if self.save==True: self.save_pred(output)
        return output

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

    def save_pred(self,preds):
        S = SaveImaged(keys = ['pred'],output_dir=self.output_folder,output_postfix='',separate_folder=False)
        for pp in preds:
            S(pp)



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



    def extract_fg_bboxes(self):
        w=WholeImagePredictor(self.project,self.run_name_w,self.ds,debug=True)
        print("Preparing data")
        p = w.predict()
        preds= w.postprocess(p)
        bboxes = [pred['pred'].meta['bounding_box'] for pred in preds]
        return bboxes
    
    def patch_prediction(self,ds,bboxes):
        data = [ds,bboxes]
        preds_all_runs={}
        for run in self.runs_p:
            p=PatchPredictor(self.project,run,data=data,debug=self.debug)
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
        # S = SaveListd(keys = ['pred'],output_dir=self.output_folder,output_postfix='',separate_folder=False)
        C  = Compose([M,A,D,K,F])
        output= C(patch_bundle)
        return output


# %%

if __name__ == "__main__":
    # ... run your application ...

    proj= Project(project_title="lits32")

# %%
    run_w='LIT-145'
    run_ps=['LIT-143','LIT-150', 'LIT-149','LIT-153','LIT-161']

# %%
    img_fn = "/s/datasets_bkp/drli_short/images/drli_005.nrrd"
    img_fn2 = "/s/insync/datasets/crc_project/qiba/qiba0_0000.nii.gz"
    img_fn3 = "/s/xnat_shadow/litq/test/images_ub/"
    fns="/s/datasets_bkp/drli_short/images/"
    paths = [img_fn, img_fn2]
    img_fns = listify(img_fn3)



    crc_fldr= "/s/xnat_shadow/crc/test/images/finalised/"
    crc_imgs = list(Path(crc_fldr).glob("*"))
    chunk = 10
# %%
    n= 3
    imgs = crc_imgs[n*chunk:(n+1)*chunk]

# %%
    # im = [{'image':im} for im in [img_fn,img_fn2]]
    En=EnsemblePredictor(proj,run_w,run_ps,debug=False)
    
    preds=En.run(imgs)
# %%
# %%
