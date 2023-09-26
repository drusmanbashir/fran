from fastai.torch_core import get_model
import torch
import ipdb
from pathlib import Path

from fran.utils.fileio import str_to_path
tr = ipdb.set_trace
import re
def get_epoch(fn:Path):
    pat = r"model_(\d*)"
    name = fn.name
    m = re.match(pat,name)
    epoch = int(m.groups()[0])
    return epoch



def make_patch_size(patch_dim0, patch_dim1):
    patch_size = [
        patch_dim0,
    ] + [
        patch_dim1,
    ] * 2
    return patch_size



@str_to_path(0)
def load_checkpoint(checkpoints_folder, model,device='cuda',strict = True, **torch_load_kwargs):
    try:
        list_of_files = checkpoints_folder.glob('*')
        # file = max(list_of_files, key=lambda p: p.stat().st_ctime)
        file = max(list_of_files, key=get_epoch)

        print("Loading last checkpoint {}".format(file))
        if isinstance(device, int): device = torch.device('cuda', device)
        elif device is None: device = 'cpu'
        state = torch.load(file, map_location=device, **torch_load_kwargs)
        hasopt = set(state)=={'model', 'opt'}
        chkpt_model_state = state['model'] if hasopt else state
        chkpt_model_keys = list(chkpt_model_state.keys())
        conflicting_string = 'module.'
        conflicting_string in chkpt_model_keys[0]
        model_keys = list(model.state_dict().keys())

        mod_keys = conflicting_string in model_keys[0] 
        chk_keys =  conflicting_string in chkpt_model_keys[0]
        if not mod_keys == chk_keys:
            chkpt_model_state_fixed = {}
            for key in chkpt_model_state.keys():
                print(key)
                neo_key = key.replace(conflicting_string,'')
                chkpt_model_state_fixed[neo_key] = chkpt_model_state[key]
            get_model(model).load_state_dict(chkpt_model_state_fixed, strict=strict)
        else:
            get_model(model).load_state_dict(chkpt_model_state, strict=strict)
        print("\n --- Successfully loaded model from checkpoint.")

        with_opt=False   # see fastai to add opt option
        # if hasopt and with_opt:
        #     try: opt.load_state_dict(state['opt'])
        #     except:
        #         if with_opt: warn("Could not load the optimizer state.")
        # elif with_opt: warn("Saved filed doesn't contain an optimizer state.")
    except:
        print("\n ---No checkpoints found. Initializing weights.")
# %%
# %%
