# %%
# NOTE: UTILITY functions to reconcile previous version with new.

import itertools as il
from pathlib import Path

from label_analysis.overlap import pbar
import numpy as np
import SimpleITK as sitk
import torch
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm

from fran.managers.training import UNetTrainer, checkpoint_from_model_id
from fran.utils.helpers import pp, slice_list
from fran.utils.imageviewers import ImageMaskViewer, view_sitk



def insert_plan_key(ckpt_fn):

    dic_tmp = torch.load(ckpt_fn)
    pp(dic_tmp.keys())
    if not 'plan' in dic_tmp['datamodule_hyper_parameters'].keys():
        print("No plan key. Adding")
        spacing  =dic_tmp['datamodule_hyper_parameters']['dataset_params']['spacing']
        dic_tmp['datamodule_hyper_parameters']['plan']= {'spacing':spacing}
        torch.save(dic_tmp, ckpt)
    else:
        print("Found plan key. No change")


<<<<<<< HEAD
=======

def remove_plan_key_add_config(ckpt_fn, config):

    # fn = '/s/fran_storage/checkpoints/litsmc/litsmc/LITS-999/checkpoints/epoch=106-val_loss=0.78.ckpt'
    ckp = torch.load(ckpt_fn)
    print(ckp.keys())
    config['plan'] =ckp['datamodule_hyper_parameters']['plan']
    ckp['datamodule_hyper_parameters']['plan']
    ckp['datamodule_hyper_parameters'].pop('plan')
    ckp['datamodule_hyper_parameters']['config'] = config
    torch.save(ckp,ckpt_fn)
# %%


>>>>>>> efc2e4fb (jj)
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

    run_w = "LITS-860"
    ckpt = checkpoint_from_model_id(run_w)
    dic_tmp = torch.load(ckpt, map_location="cpu")
    insert_plan_key(ckpt)

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
<<<<<<< HEAD
    fldr = Path("/s/fran_storage/datasets/preprocessed/fixed_spacing/litsmc/spc_080_080_150")
=======
    fldr = Path("/s/fran_storage/datasets/preprocessed/fixed_spacing/lidc2/spc_080_080_150")
>>>>>>> efc2e4fb (jj)
    fns = list(fldr.rglob("*.pt"))
    for fn in pbar( fns):
        lm = torch.load(fn)
        lm.meta
        lm.meta['filename_or_obj']=lm.meta['filename']
        del lm.meta['filename']
        torch.save(lm,fn)
# %%
