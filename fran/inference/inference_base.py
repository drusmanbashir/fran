# %%
from time import time
from lightning.fabric import Fabric
import lightning.pytorch as pl
from collections.abc import Callable
from monai.config.type_definitions import KeysCollection
from monai.data.image_writer import ITKWriter
from monai.data.utils import decollate_batch
from monai.data.box_utils import *
from monai.inferers.merger import *
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm
from monai.transforms import (
    LoadImage,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    Spacingd,
    Invertd,
)
from fran.utils.itk_sitk import *
from monai.transforms.croppad.dictionary import BoundingRectd
from monai.apps.detection.transforms.array import *
from monai.transforms.post.dictionary import Activationsd, KeepLargestConnectedComponentd, MeanEnsembled
from monai.data.box_utils import *
from monai.transforms.transform import MapTransform, Transform
from monai.transforms.spatial.dictionary import Resized
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
            fl=Path(patch_bundle['image'].meta['filename_or_obj']).name
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

class PredFlToInt(TFC):
    def encodes(self, pred_fl: Tensor):
        pred_int = torch.argmax(pred_fl, 0, keepdim=False)
        pred_int = pred_int.to(torch.uint8)
        return pred_int


class ToNumpy(TFC):
    def __init__(self, encode_dtype=np.uint8):
        if encode_dtype == np.uint8:
            self.decode_dtype = torch.uint8
        self.encode_dtype = encode_dtype

    def encodes(self, tnsrs):
        return [np.array(tnsr, dtype=self.encode_dtype) for tnsr in tnsrs]



class SimpleDataset(Dataset):
    def __init__(
        self, imgs: Union[torch.Tensor,list], transform: Union[Callable , None ]= None
    ) -> None:
        imgs=listify(imgs)
        store_attr("imgs,transform")

    def __getitem__(self, idx):
        im = {'image':self.imgs[idx]}
        im=self.transform(im)
        return im

    def __len__(self): return len(self.imgs)

class WholeImageDM(DataManager):
    def prepare_data(self, imgs):
        self.create_transforms()
        self.pred_ds = SimpleDataset(imgs, transform=self.transforms)
        self.create_dataloader()

    def create_transforms(self):

        # E = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")

        # E = EnsureType( device="cuda", track_meta=True)
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )
        R = Resized(
            keys=["image"],
            spatial_size=self.dataset_params["patch_size"],
        )

        C = Compose([ N, R], lazy=True)
        self.transforms = C

    def create_dataloader(self):
        self.pred_dl = DataLoader(
            self.pred_ds, num_workers=4, batch_size=8#, collate_fn=img_metadata_collated
        )


class PatchDM(DataManager):
    def create_transforms(self):
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

    def create_dataloader(self):
        pass


class WholeImagePredictor(GetAttr,DictToAttr):
    _default = "datamodule"

    def __init__(self, project,run_name,  devices=1):
        store_attr('project,run_name,devices')
        self.ckpt = checkpoint_from_model_id(run_name)
        dic1=torch.load(self.ckpt)
        dic2={}
        relevant_keys=['datamodule_hyper_parameters']
        for key in relevant_keys:
            dic2[key]=dic1[key]
            self.assimilate_dict(dic2[key])
    



        self.predictions_folder=project.predictions_folder

        self.prepare_model()
        self.prepare_trainer(devices=devices)

    def prepare_data(self, imgs):
        self.datamodule = WholeImageDM.load_from_checkpoint(self.ckpt)
        self.datamodule.prepare_data(imgs)

    def prepare_model(self):
        self.model = nnUNetTrainer.load_from_checkpoint(
            self.ckpt, project=self.project, dataset_params=self.dataset_params
        )


    def prepare_trainer(self, devices):
        self.trainer = pl.Trainer(
            accelerator="gpu", devices=devices, precision="16-mixed"
        )

    def predict(self):
        out = self.trainer.predict(model=self.model, dataloaders=self.pred_dl)
        self.preds = out[0]

    def postprocess(self):

        out1 = self.preds
        out2 = decollate_batch(out1,detach=True)

        I = Invertd(keys=['pred'],transform=self.datamodule.transforms,orig_keys=['image'])
        D = AsDiscreted(keys=['pred'],argmax=True,threshold=0.5)
        K= KeepLargestConnectedComponentd(keys=['pred'])
        B=BoundingRectd(keys=['pred'])
        S = SlicesFromBBox(keys=['pred_bbox'])
        C = Compose([I,D,K,B,S])

        out_final=[]
        for ou in out2:
            tmp=C(ou) 
            out_final.append(tmp)
        return out_final

    @property
    def output_folder(self):
        fldr='_'.join(self.run_name)
        fldr = self.predictions_folder/fldr
        return fldr



class PatchPredictor(WholeImagePredictor):
    def __init__(self,project, run_name,  patch_overlap=0.25,bs=8,grid_mode="gaussian", devices=1):
        super().__init__(project,run_name,  devices)

        # device='cuda'
        self.grid_mode = grid_mode
        self.patch_size = self.dataset_params["patch_size"]
        self.inferer = SlidingWindowInferer(
            roi_size=self.dataset_params['patch_size'],
            sw_batch_size=bs,
            overlap=patch_overlap,
            mode=grid_mode,
            progress=True,
        )


    def prepare_data(self, imgs_c):

        N = NormaliseClip(
            # keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )
        imgs_c =[N(img) for img in imgs_c]
        # imgs_c = [img.unsqueeze(0) for img in imgs_c] # add batch dim since Dataloader will not be used
        # imgs_c=[i.to('cuda') for i in imgs_c]
        self.imgs_c=[i.half() for i in imgs_c]
        


    def predict(self):
        pred_patches = []
        for i in range(len(self.imgs_c)):
            img_input=self.imgs_c[i]
            img_input=img_input.unsqueeze(0)
            with torch.no_grad():
                img_input= img_input.to('cuda')
                output_tensor = self.inferer(inputs=img_input, network=self.model)
                output_tensor = output_tensor[0]
                pred_patches.append(output_tensor)
        return pred_patches

    def prepare_model(self):
        super().prepare_model()
        self.model.eval()
        fabric = Fabric(precision="16-mixed",devices=self.devices)
        self.model=fabric.setup(self.model)


class EnsemblePredictor():
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



    def run(self,imgs):
        img_tnsrs, num_cases=self.parse_input(imgs)
        bboxes= self.extract_fg_bboxes(img_tnsrs)
        pred_patches = self.patch_prediction(img_tnsrs,bboxes)
        pred_patches = self.decollate_patches(pred_patches,num_cases)

        patch_bundles={'image':img_tnsrs,'bbox':bboxes}
        patch_bundles.update(pred_patches)
        output=[]
        for i in range(num_cases):
            patch_bundle = self.get_mini_bundle(patch_bundles,i)
            preds = self.postprocess(patch_bundle)
            output.append(preds)
            self.save_pred(preds)
        return output

    def parse_input(self,imgs_inp):
        imgs_inp=listify(imgs_inp)
        loader=LoadImage(image_only=True,ensure_channel_first=True,simple_keys=True)
        imgs_out = []
        for dat in imgs_inp:
            if any([isinstance(dat,str),isinstance(dat,Path)]):
                dat = Path(dat)
                if dat.is_dir():
                    dat = list(dat.glob("*"))
                else:
                    dat=[dat]
                imgs=[loader(dd) for dd in dat]
            else:
                if isinstance(dat,sitk.Image):
                    dat= ConvertSimpleItkImageToItkImage(dat, itk.F)
                if isinstance(dat,itk.Image):
                    img=itm(dat) 
                else: 
                    print("Not implemented")
                    tr()
                imgs=[img]
            imgs_out.extend(imgs)
        return imgs_out,len(imgs_out)

    def get_mini_bundle(self,patch_bundles,indx):
            patch_bundle={}
            for key,val in patch_bundles.items():
                pred_patch={key:val[indx]}
                patch_bundle.update(pred_patch)
            return patch_bundle

    def decollate_patches(self,pa,num_cases):
        keys = self.runs_p
        preds={}
        for i,key in enumerate(keys):
            preds_per_run=[]
            for j in range(num_cases):
                pred =  pa[key][j]
                pred=pred.squeeze(0)
                preds_per_run.append(pred)
            preds[key]=preds_per_run
        return preds

    def save_pred(self,pred):
            S=Saved(self.output_folder)
            S(pred)

    def extract_fg_bboxes(self,img_tnsrs):

        w=WholeImagePredictor(self.project,self.run_name_w)
        print("Preparing data")
        w.prepare_data(img_tnsrs)
        w.predict()
        print("Predicted. Now postprocessing")
        preds = w.postprocess()
        bboxes = [pred['pred_bbox'] for pred in preds]
        return bboxes
    
    def patch_prediction(self,img_tnsrs,bboxes):
        imgs_c = [tnsr[bbo] for tnsr,bbo in zip(img_tnsrs,bboxes)]
        preds_all_runs={}
        for run in self.runs_p:
            p=PatchPredictor(project,run)
            p.prepare_data(imgs_c)
            preds=p.predict()
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
        C  = Compose([M,A,D,K,F])
        output= C(patch_bundle)
        return output

# %%
if __name__ == "__main__":
    # ... run your application ...

    common_vars_filename = os.environ["FRAN_COMMON_PATHS"]
    project = Project(project_title="lits32")

    dataset_params = load_dict(project.global_properties_filename)
    configs = ConfigMaker(project, raytune=False).config
# %%
    run_w='LIT-41'
    run_ps=['LIT-62', 'LIT-44','LIT-59']
# %%
    img_fn = "/s/xnat_shadow/litq/test/images_few/litq_35_20200728.nii.gz"
    img_fn2 = "/s/xnat_shadow/litq/images/litq_31_20220826.nii.gz"
    img_fn3 = "/s/xnat_shadow/litq/test/images_few/"
    paths = [img_fn, img_fn2]
    img_fns = listify(img_fn)


    


# %%

    En=EnsemblePredictor(project,run_w,run_ps)
    out = En.run(img_fns)
# %%
    En.parse_input(img_fns)
    

# %%

