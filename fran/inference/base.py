# %%
from collections.abc import Callable, Sequence
import SimpleITK as sitk
from fastcore.all import listify, store_attr
from fastcore.foundation import GetAttr
from lightning.fabric import Fabric
from monai.data.dataloader import DataLoader    
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm
import itk
from monai.data.dataset import Dataset, PersistentDataset
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged, SaveImaged
from monai.transforms.post.dictionary import AsDiscreted, Invertd
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, SqueezeDimd
from prompt_toolkit.shortcuts import input_dialog
from pathlib import Path
import torch
from fran.data.dataset import InferenceDatasetNii, InferenceDatasetPersistent
from fran.managers.training import UNetTrainer, checkpoint_from_model_id
from fran.transforms.inferencetransforms import ToCPUd
from fran.utils.dictopts import DictToAttr
from fran.utils.fileio import maybe_makedirs
from fran.utils.itk_sitk import ConvertSimpleItkImageToItkImage
import numpy as np
import itertools as il



def list_to_chunks(input_list: list, chunksize: int):
    n_lists = int(np.ceil(len(input_list) / chunksize))

    fpl = int(len(input_list) / n_lists)
    inds = [[fpl * x, fpl * (x + 1)] for x in range(n_lists - 1)]
    inds.append([fpl * (n_lists - 1), None])

    chunks = list(il.starmap(slice_list, zip([input_list] * n_lists, inds)))
    return chunks

def load_dataset_params(model_id):
    ckpt = checkpoint_from_model_id(model_id)
    dic_tmp=torch.load(ckpt)
    dataset_params=dic_tmp['datamodule_hyper_parameters']['dataset_params']
    return dataset_params

class BaseInferer(GetAttr, DictToAttr):
    def __init__(self, project,run_name,bs=8,patch_overlap=.25,mode='gaussian', device=[1],debug=True):
        '''
        data is a dataset from Ensemble in this base class
        '''

        store_attr('project,run_name,device,debug')
        self.dataset_params  = load_dataset_params(run_name)
    
        self.inferer = SlidingWindowInferer(
            roi_size=self.dataset_params['patch_size'],
            sw_batch_size=bs,
            overlap=patch_overlap,
            mode=mode,
            progress=True,
        )
        self.prepare_model()
        # self.prepare_data(data)

    def run(self,imgs,chunksize=12):
        '''
        chunksize is necessary in large lists to manage system ram
        '''
        imgs  = list_to_chunks(imgs,chunksize)
        for imgs_sublist in imgs:
            self.prepare_data(imgs_sublist)
            self.create_postprocess_transforms()
            preds= self.predict()
            # preds = self.decollate(preds)
            output= self.postprocess(preds)
            # if self.save==True: self.save_pred(output)
        return output


    def prepare_data(self,imgs):
        '''
        imgs: list
        '''
        if len(imgs)>3:
            self.ds = InferenceDatasetPersistent(data=imgs,cache_dir = self.project.cold_datasets_folder/("cache"))
        else:
            self.ds = InferenceDatasetNii(imgs,self.dataset_params)
        self.pred_dl = DataLoader(
                self.ds, num_workers=0, batch_size=1, collate_fn = None
            )


    def save_pred(self,preds):
        S = SaveImaged(keys = ['pred'],output_dir=self.output_folder,output_postfix='',separate_folder=False)
        for pp in preds:
            S(pp)

    def create_postprocess_transforms(self):

        Sq = SqueezeDimd(keys = ['pred'], dim=0)
        I = Invertd(keys=['pred'],transform=self.ds.transform,orig_keys=['image'])
        D = AsDiscreted(keys=['pred'],argmax=True,threshold=0.5)
        C = ToCPUd(keys=['image','pred'])
        tfms = [Sq,I,D,C]
        if self.debug==True:
            Sa = SaveImaged(keys = ['pred'],output_dir=self.output_folder,output_postfix='',separate_folder=False)
            tfms.insert(1,Sa)
        C = Compose(tfms)
        self.postprocess_transforms=C

    def prepare_model(self):
        model = UNetTrainer.load_from_checkpoint(
            self.ckpt, project=self.project, dataset_params=self.dataset_params, strict=False
        )

        fabric = Fabric(precision="16-mixed",devices=self.devices)
        self.model=fabric.setup(model)

    def predict(self):
        outputs = []
        for i ,batch in enumerate(self.pred_dl):
                with torch.no_grad():
                    img_input=batch['image']
                    img_input = img_input.cuda()
                    output_tensor = self.inferer(inputs=img_input, network=self.model)
                    output_tensor = output_tensor[0]
                    batch['pred']=output_tensor
                    batch['pred'].meta = batch['image'].meta
                    outputs.append(batch)
        return outputs


    def postprocess(self, preds):
        out_final=[]
        for batch in preds:
            tmp=self.postprocess_transforms(batch) 
            out_final.append(tmp)
        return out_final

    @property
    def output_folder(self):
        run_name = listify(self.run_name)
        fldr='_'.join(run_name)
        fldr = self.project.predictions_folder/fldr
        return fldr


# %%

if __name__ == "__main__":
    # ... run your application ...
    from fran.utils.common import *
    proj= Project(project_title="nodes")



    run_ps=['LITS-702']
    run_name = run_ps[0]

# %%
    img_fn = "/s/xnat_shadow/nodes/imgs_no_mask/nodes_4_20201024_CAP1p5mm_thick.nii.gz"

    img_fns = [img_fn]
    input_data = [{'image':im_fn} for im_fn in img_fns]
    debug = True


# %%
    P=BaseInferer(proj, run_ps[0], debug=debug)

    preds= P.run(img_fns)
# %%
    imgs = img_fns
    P.prepare_data(imgs)
    P.create_postprocess_transforms()
    preds= P.predict()
# %%
    a = P.ds[0]
    im = a['image']
    im = im[0]
    ImageMaskViewer([im,im])
# %%
    # preds = P.decollate(preds)
    # output= P.postprocess(preds)
# %%

    out_final=[]
    # for batch in preds:

    batch= preds[0]

    C = ToCPUd(keys=['image','pred'])
    Sq = SqueezeDimd(keys = ['pred'], dim=0)
    batch = C(batch)
    batch = Sq(batch)
    batch['pred'].shape

    I = Invertd(keys=['pred'],transform=P.ds.transform,orig_keys=['image'])
    tmp = I(batch)
    tmp=P.postprocess_transforms(batch) 
    out_final.append(tmp)
# %%
    data = P.ds[0]

    P.ds.transform
    P.ds.transform.inverse(data)
# %%
    I = Invertd(keys=['pred'],transform=P.ds.transform,orig_keys=['image'])
    I = Invertd(keys=['image'],transform=P.ds.transform,orig_keys=['image'])
    pp = preds[0].copy()
    print(pp.keys())
    pp['pred']=pp['pred'][0:1,0]

    pp['pred'].shape
    pp['pred'].meta
    a =  I(pp)

    dici = {'image': img_fn}
    L=LoadImaged(keys=['image'],image_only=True,ensure_channel_first=True,simple_keys=True)

    S = Spacingd(keys=["image"], pixdim=P.ds.dataset_params['spacings'])
    tfms = ([L,S])
    Co = Compose(tfms)

    dd = L(dici)
    dda = S(dd)

# %%
    dd = Co(dici)
# %%
    Co.inverse(dd)
# %%
