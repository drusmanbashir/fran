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
        model_state = state['model'] if hasopt else state
        get_model(model).load_state_dict(model_state, strict=strict)
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
