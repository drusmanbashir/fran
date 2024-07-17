# NOTE: UTILITY functions to reconcile previous version with new.
# %%

import ast
import itertools as il
from collections.abc import Callable, Sequence
from pathlib import Path

import itk
from label_analysis.overlap import pbar
import numpy as np
import SimpleITK as sitk
import torch
from fastcore.all import listify, store_attr
from fastcore.foundation import GetAttr
from lightning.fabric import Fabric
from lightning.pytorch import LightningModule
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset, PersistentDataset
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm
from monai.data.utils import decollate_batch
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged, SaveImaged
from monai.transforms.post.dictionary import Activationsd, AsDiscreted, Invertd
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, SqueezeDimd
from prompt_toolkit.shortcuts import input_dialog

from fran.data.dataset import (
    InferenceDatasetNii,
    InferenceDatasetPersistent,
    NormaliseClipd,
)
from fran.managers.training import UNetTrainer, checkpoint_from_model_id
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import SaveMultiChanneld, ToCPUd
from fran.transforms.spatialtransforms import ResizeToMetaSpatialShaped
from fran.utils.dictopts import DictToAttr
from fran.utils.fileio import maybe_makedirs
from fran.utils.helpers import pp, slice_list
from fran.utils.imageviewers import ImageMaskViewer, view_sitk
from fran.utils.itk_sitk import ConvertSimpleItkImageToItkImage


def list_to_chunks(input_list: list, chunksize: int):
    n_lists = int(np.ceil(len(input_list) / chunksize))

    fpl = int(len(input_list) / n_lists)
    inds = [[fpl * x, fpl * (x + 1)] for x in range(n_lists - 1)]
    inds.append([fpl * (n_lists - 1), None])

    chunks = list(il.starmap(slice_list, zip([input_list] * n_lists, inds)))
    return chunks


def load_params(model_id):
    ckpt = checkpoint_from_model_id(model_id)
    dic_tmp = torch.load(ckpt, map_location="cpu")
    return dic_tmp["datamodule_hyper_parameters"]

def remove_loss_key_state_dict(model_id):
        ckpt_fn = checkpoint_from_model_id(model_id)
        ckpt= torch.load(ckpt_fn)
        ckpt_state = ckpt['state_dict']
        keys = [k for k in ckpt_state.keys() if "loss" in k]
        if len(keys)>0:
            print("Found loss keys:",keys)
            for k in keys:
                del ckpt_state[k]
            torch.save(ckpt,ckpt_fn)
        else:
            print("No loss keys in state_dict. No change")

    
# %%
if __name__ == "__main__":
    model_id = "LITS-957"

    run_w = "LIT-145"
    ckpt = checkpoint_from_model_id(run_w)
    dic_tmp = torch.load(ckpt, map_location="cpu")

    keys = ['spacing']
    dici = dic_tmp['datamodule_hyper_parameters']['plan']
    dici['spacing'] = '.8,.8,1.5'

# %%
    fn = "/s/fran_storage/checkpoints/lidc2/lidc2/LITS-911/checkpoints/epoch=499-step=8000.ckpt"
    std =  torch.load(fn)
    ckpt_state = std['state_dict']
    ckpt_state = remove_loss_key_state_dict(ckpt_state)
    torch.save(std,fn)


# %%
    remove_loss_key_state_dict("LITS-911")

    pp(dic_tmp.keys())
    if not 'plan' in dic_tmp['datamodule_hyper_parameters'].keys():
        spacing  =dic_tmp['datamodule_hyper_parameters']['dataset_params']['spacing']
        dic_tmp['datamodule_hyper_parameters']['plan']= {'spacing':spacing}
        torch.save(dic_tmp, ckpt)
# %%
#SECTION:-------------------- filename_or_obj--------------------------------------------------------------------------------------

# %%
    fldr = Path("/s/fran_storage/datasets/preprocessed/fixed_spacing/litsmc/spc_080_080_150")
    fns = list(fldr.rglob("*.pt"))
    for fn in pbar( fns):
        lm = torch.load(fn)
        lm.meta
        lm.meta['filename_or_obj']=lm.meta['filename']
        del lm.meta['filename']
        torch.save(lm,fn)
# %%
