import torch
import ipdb
from pathlib import Path

from utilz.fileio import str_to_path
tr = ipdb.set_trace
import re
def get_epoch(fn:Path):
    pat = r"model_(\d*)"
    name = fn.name
    m = re.match(pat,name)
    epoch = int(m.groups()[0])
    return epoch


def get_ds_remapping(ds:str,global_properties):
#BUG: this needs  to go.  remappings to be unique to each plan in excel  (see #5)
        key = 'lm_group'
        keys=[]
        for k in global_properties.keys():
            if key in k:
                keys.append(k)

        for k in keys:
            dses  = global_properties[k]['ds']
            if ds in dses:
                labs_src = global_properties[k]['labels']
                if hasattr (global_properties[k],'labels'):
                    labs_dest = global_properties[k]['labels_neo']
                else:
                    labs_dest = labs_src
                remapping = {src:dest for src,dest in zip(labs_src,labs_dest)}
                return remapping
        raise Exception("No lm group for dataset {}".format(ds))


def make_patch_size(patch_dim0, patch_dim1):
    patch_size = [
        patch_dim0,
    ] + [
        patch_dim1,
    ] * 2
    return patch_size

def reconcile_keys(local_model_state,ckpt_model_state,ckpt_string,local_string ):
        chkpt_model_keys = list(ckpt_model_state.keys())
        local_model_keys = list(local_model_state.keys())
        conflicting_string in chkpt_model_keys[0]
        mod_keys_flag = conflicting_string in local_model_keys[0] 
        chk_keys_flag =  conflicting_string in chkpt_model_keys[0]
        if not mod_keys_flag == chk_keys_flag:
                    chkpt_model_state_fixed = {}
                    for key in ckpt_model_state.keys():
                        neo_key = key.replace(conflicting_string,'')
                        chkpt_model_state_fixed[neo_key] = ckpt_model_state[key]
                      
                    return chkpt_model_state_fixed
        else:
            return ckpt_model_state


def reconcile_keys(local_model_state,chkpt_model_state,conflicting_string='module.'):
        chkpt_model_keys = list(chkpt_model_state.keys())
        local_model_keys = list(local_model_state.keys())
        conflicting_string in chkpt_model_keys[0]
        mod_keys_flag = conflicting_string in local_model_keys[0] 
        chk_keys_flag =  conflicting_string in chkpt_model_keys[0]
        if not mod_keys_flag == chk_keys_flag:
                    chkpt_model_state_fixed = {}
                    for key in chkpt_model_state.keys():
                        neo_key = key.replace(conflicting_string,'')
                        chkpt_model_state_fixed[neo_key] = chkpt_model_state[key]
                      
                    return chkpt_model_state_fixed
        else:
            return chkpt_model_state

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
        get_model(model).load_state_dict(chkpt_model_state, strict=strict)
        print("\n --- Successfully loaded model from checkpoint.")

        with_opt=False   # see fastai to add opt option
        # if hasopt and with_opt:
        #     try: opt.load_state_dict(state['opt'])
        #     ecept:
        #         if with_opt: warn("Could not load the optimizer state.")
        # elif with_opt: warn("Saved filed doesn't contain an optimizer state.")
    except Exception as e:
        print("Exception  occurred : {}".format(e))
        print("\n ---Initializing weights.")
# %%
# %%
