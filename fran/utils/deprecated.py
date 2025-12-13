# %%
# NOTE: UTILITY functions to reconcile previous version with new.
from fran.trainers.base import checkpoint_from_model_id
from tqdm.auto import tqdm as pbar

import itertools as il
from pathlib import Path

import numpy as np
import torch

from utilz.helpers import pp, slice_list


def insert_plan_key(ckpt_fn):

    dic_tmp = torch.load(ckpt_fn)
    pp(dic_tmp.keys())
    if not "plan" in dic_tmp["datamodule_hyper_parameters"].keys():
        print("No plan key. Adding")
        spacing = dic_tmp["datamodule_hyper_parameters"]["dataset_params"]["spacing"]
        dic_tmp["datamodule_hyper_parameters"]["plan"] = {"spacing": spacing}
        torch.save(dic_tmp, ckpt)
    else:
        print("Found plan key. No change")


def remove_plan_key_add_config(ckpt_fn, config):

    ckp = torch.load(ckpt_fn)
    print(ckp.keys())
    config["plan"] = ckp["datamodule_hyper_parameters"]["plan"]
    ckp["datamodule_hyper_parameters"]["plan"]
    ckp["datamodule_hyper_parameters"].pop("plan")
    ckp["datamodule_hyper_parameters"]["config"] = config
    torch.save(ckp, ckpt_fn)


def move_key_plan_to_dataset_params(ckpt_fn, key):
    # ckpt_fn = '/s/fran_storage/checkpoints/litsmc/litsmc/LITS-999/checkpoints/epoch=106-val_loss=0.78.ckpt'
    ckp = torch.load(ckpt_fn)
    ckp["datamodule_hyper_parameters"].keys()
    config = ckp["datamodule_hyper_parameters"]["config"]
    value = config["dataset_params"][key]
    ckp["datamodule_hyper_parameters"]["config"]["plan"][key] = value
    torch.save(ckp, ckpt_fn)


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
    ckpt = torch.load(ckpt_fn)
    ckpt_state = ckpt["state_dict"]
    keys = [k for k in ckpt_state.keys() if "loss" in k]
    if len(keys) > 0:
        print("Found loss keys:", keys)
        for k in keys:
            del ckpt_state[k]
        torch.save(ckpt, ckpt_fn)
    else:
        print("No loss keys in state_dict. No change")


def find_key_in_dict(d, search_key, parent_keys=None):
    """Recursively find all occurrences of search_key in nested dictionaries and return their parent keys

    Args:
        d: Dictionary to search in
        search_key: Key to search for
        parent_keys: List to track the current path of parent keys

    Returns:
        list: List of lists, each containing the sequence of parent keys to reach the search_key
    """
    if parent_keys is None:
        parent_keys = []

    found_paths = []

    if not isinstance(d, dict):
        return found_paths

    for k, v in d.items():
        if k == search_key:
            found_paths.append(parent_keys + [k])

        if isinstance(v, dict):
            found_paths.extend(find_key_in_dict(v, search_key, parent_keys + [k]))

    return found_paths


def copy_dict_structure(dict_src, dict_dest, path="root", missing_keys=None):
    """Copy structure from dict_src and fill values from dict_dest where possible.

    Args:
        dict_src: Source dictionary whose structure will be copied
        dict_dest: Destination dictionary to get values from
        path: Current path in the dictionary (used for recursion)
        missing_keys: Dictionary to track keys not found in dict_dest

    Returns:
        new_dict: New dictionary with src structure and matching values from dest
    """
    if missing_keys is None:
        missing_keys = {}

    if not isinstance(dict_src, dict):
        return dict_src

    new_dict = {}
    for key, value in dict_src.items():
        current_path = f"{path}->{key}"

        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            new_dict[key] = copy_dict_structure(
                value, dict_dest, current_path, missing_keys
            )
        else:
            # For leaf nodes, try to find matching key anywhere in dict_dest
            found_value = find_value_in_dict(dict_dest, key)
            if found_value is not None:
                new_dict[key] = found_value
            else:
                # If key not found in dest, use source value and track it
                new_dict[key] = value
                missing_keys[current_path] = value
                print(
                    f"Key not found in destination dict: {current_path}, using source value: {value}"
                )

    return new_dict


def get_key_path(key_path):
    """Convert a list of keys into a single string representation

    Args:
        key_path: List of strings representing the path to the desired key

    Returns:
        str: A string representation of the key path (e.g., "key1.key2.key3")
    """
    return ".".join(key_path)


def find_value_in_dict(d, search_key):
    """Recursively search for a key in nested dictionary and return its value"""
    if not isinstance(d, dict):
        return None

    if search_key in d:
        return d[search_key]

    for value in d.values():
        if isinstance(value, dict):
            result = find_value_in_dict(value, search_key)
            if result is not None:
                return result

    return None


def add_subdict(model_id):
    pass


# %%
if __name__ == "__main__":

    run_src = "LITS-933"
    run_src = "LITS-1217"
    project_title = "litsmc"
    value = 4
# %%
    project_title = "lidc2"
    value = 3
# %%
    run_src = "LITS-1230"
    project_title = "nodes"
    value = 3
# %%
    run_src = "LITS-1088"
    project_title = "totalseg"
    value = 2


# %%
    run_src = "LITS-1120"
    project_title = "totalseg"
    value = 2
# %%
# SECTION:-------------------- OLDER settings did not have a config param in UNet. Block below copies that from datamodule-------------------------------------------------------------------------------------- <CR>
    ckpt_src = checkpoint_from_model_id(run_src)
    dict_src = torch.load(ckpt_src, map_location="cpu", weights_only=False)
# %%
#SECTION:-------------------- CONFIG --> CONFIGS--------------------------------------------------------------------------------------

    dici = dict_src.copy()
    dici['datamodule_hyper_parameters'].keys()
    dici['datamodule_hyper_parameters']['configs']
    dici["hyper_parameters"]['configs']#
    dici["hyper_parameters"]['configs']= dici["hyper_parameters"]['config'].copy()
    dici["datamodule_hyper_parameters"]["configs"]=  dici["datamodule_hyper_parameters"]["config"].copy()  # ['plan_train']#=pln
    # dici["datamodule_hyper_parameters"]["configs"]
    dici['datamodule_hyper_parameters']['configs']['plan_train']
    del dici['datamodule_hyper_parameters']['config']
    del dici['hyper_parameters']['config']
    dici['hyper_parameters'].keys()
    torch.save(dici, ckpt_src)
# %%
    dici['datamodule_hyper_parameters']['configs']['plan_train']
    dici['datamodule_hyper_parameters']['configs'].keys()
    dici["hyper_parameters"]['configs']
    dici["hyper_parameters"]['configs']
# %%
#SECTION:-------------------- OTHER--------------------------------------------------------------------------------------
    pp(dici.keys())
    pp(dici["hyper_parameters"].keys())
    dici['hyper_parameters']['lr']

    dici['lr_schedulers']
    pp(dici["hyper_parameters"]['config'].keys())
    dici["hyper_parameters"]['configs']= dici["hyper_parameters"]['config'].copy()
    # pln = dici['datamodule_hyper_parameters']['config']['plan{}'.format(value)]
    pln = dici["datamodule_hyper_parameters"]["config"]["plan"]
    datamod = dici["datamodule_hyper_parameters"]["config"]
    dici["hyper_parameters"][
        "config"
    ]  # = dici['datamodule_hyper_parameters']['config'].copy()#['plan_train']=pln
    dici["hyper_parameters"]["config"]["plan_train"] #= pln
    dici["hyper_parameters"]["config"]["plan_train"] ["remapping"]= None
    dici["hyper_parameters"]["config"]["plan_valid"] = pln

# %%
    dici["datamodule_hyper_parameters"]["config"]["plan_train"]["remapping"] = None
    dici["hyper_parameters"]["config"]["plan_train"] = plan
    dici["hyper_parameters"]["config"]["plan_valid"] = plan

    # CODE: adding plan index for NODES only
    dici["hyper_parameters"]["config"]["dataset_params"]["plan_train"] = 2
    dici["hyper_parameters"]["config"]["dataset_params"]["plan_valid"] = 2
    dici["hyper_parameters"]["config"]["dataset_params"]
# %%
    dici["datamodule_hyper_parameters"]["config"]["plan_valid"]  # =pln
    dici["datamodule_hyper_parameters"]["config"]  # ['plan_train']#=pln
    torch.save(dici, ckpt_src)
# %%

    dici["hyper_parameters"].keys()
    dici["hyper_parameters"]["project_title"] = project_title

    dici["datamodule_hyper_parameters"]["project_title"]
    dici["datamodule_hyper_parameters"]["project_title"] = project_title
    torch.save(dici, ckpt_src)

# %%
    run2 = "LITS-1088"
    ckpt2 = checkpoint_from_model_id(run2)
    ckpt2_neo = ckpt2.str_replace(".ckpt", "_neo.ckpt")
    dici = torch.load(ckpt2, map_location="cpu")
    dici["hyper_parameters"]["project_title"] = "total_seg"
    dici["datamodule_hyper_parameters"]["project_title"] = "total_seg"
    dici["datamodule_hyper_parameters"].keys()
    torch.save(dici, ckpt_src)
    torch.save(dici, ckpt_src)

# %%

    dd = copy_dict_structure(dict_src, dict_dest)

    dd["hyper_parameters"].keys()
    dd["hyper_parameters"]["config"].keys()
    torch.save(dd, ckpt2_neo)
# %%
# SECTION:-------------------- PATCH_SIZE TO PLAN-------------------------------------------------------------------------------------- <CR>

    # dici['state_dict']= dict_dest['state_dict'].copy()

    keys = find_key_in_dict(dici, "plan")
    key_j = get_key_path(keys[0])

    val = get_nested_value(dici, keys[0])
    dici["hyper_parameters"]["config"][
        "plan_train"
    ]  # =dici['hyper_parameters']['config']['plan'].copy()
    dici["hyper_parameters"]["config"]["plan_train"] = dici["hyper_parameters"][
        "config"
    ]["plan"].copy()
    dici["hyper_parameters"]["config"]["plan_valid"] = dici["hyper_parameters"][
        "config"
    ]["plan"].copy()

    dici["datamodule_hyper_parameters"]["config"] = dici["hyper_parameters"][
        "config"
    ].copy()
    dici["datamodule_hyper_parameters"]["config"]
    torch.save(dici, ckpt_src)
# %%
    dici["hyper_parameters"]["config"] = {}

    dici["hyper_parameters"]["config"].keys()
    dici["hyper_parameters"]["config"][
        "dataset_params"
    ]  # = dici['datamodule_hyper_parameters']['dataset_params'].copy()
    dici["hyper_parameters"]["config"]["model_params"] = dici["hyper_parameters"][
        "model_params"
    ].copy()

    dici["hyper_parameters"]["config"]["plan"] = dici["hyper_parameters"]["config"][
        "dataset_params"
    ].copy()
    dici["hyper_parameters"]["config"]["plan"]["mode"] = "whole"
    dici["hyper_parameters"]["config"]["plan"]["spacing"]
    dici["hyper_parameters"]["config"]["loss_params"] = dici["hyper_parameters"][
        "loss_params"
    ].copy()
# %%
    dici["hyper_parameters"]["config"].keys()
    dici["hyper_parameters"]["config"]["plan"]["patch_size"]  # =[96,96,96]

    dici["hyper_parameters"]["config"]["dataset_params"]["patch_size"] = [
        96,
        96,
        96,
    ]  # =
    dici[
        "datamodule_hyper_parameters"
    ].keys()  # ['dataset_params']['patch_size'] = [96,96,96]#=
    dici["hyper_parameters"]["project"]
    dict_src["hyper_parameters"]["project"]
# %%
    dici["datamodule_hyper_parameters"].keys()
    dici["datamodule_hyper_parameters"]["dataset_params"]
    dici["datamodule_hyper_parameters"]["config"] = dici["hyper_parameters"][
        "config"
    ].copy()
    dici["datamodule_hyper_parameters"]["config"]["dataset_params"] = dici[
        "datamodule_hyper_parameters"
    ]["dataset_params"].copy()
    dici["datamodule_hyper_parameters"]["config"]["plan"] = dici[
        "datamodule_hyper_parameters"
    ]["dataset_params"].copy()
    dici["datamodule_hyper_parameters"]["config"]["plan"]["spacing"]
    dici["datamodule_hyper_parameters"]["config"]["plan"]["mode"] = "whole"
    dici["datamodule_hyper_parameters"]["config"]["loss_params"] = dici[
        "hyper_parameters"
    ]["loss_params"].copy()
    dici["hyper_parameters"]["project"] = dici["datamodule_hyper_parameters"][
        "project"
    ].copy()
    dici["hyper_parameters"].keys()
    torch.save(dici, ckpt2_neo)
# %%
    pp(dict_dest["hyper_parameters"].keys())
    pp(dict_src["hyper_parameters"].keys())
# %%
    dd = copy_dict_structure(
        dict_src["hyper_parameters"], dict_dest["hyper_parameters"]
    )
    dd2 = copy_dict_structure(
        dict_src["datamodule_hyper_parameters"],
        dict_dest["datamodule_hyper_parameters"],
    )
    dict_dest["hyper_parameters"].keys()  # =dd
    dict_dest["datamodule_hyper_parameters"].keys()  # =dd2
    torch.save(dict_dest, ckpt2)
# %%
    dic_dest["hyper_parameters"]["plan"] = dic_dest["datamodule_hyper_parameters"][
        "config"
    ]["plan"].copy()
    dic_dest["hyper_parameters"]["plan"]
    dic_dest["datamodule_hyper_parameters"]["config"]["plan"]["patch_size"] = dic_dest[
        "datamodule_hyper_parameters"
    ]["config"]["dataset_params"]["patch_size"]
    dic_dest["datamodule_hyper_parameters"]["config"]["plan"]
    dic_dest["datamodule_hyper_parameters"]["dataset_params"]["plan"]
    dic_dest["datamodule_hyper_parameters"]["plan"] = dic_dest[
        "datamodule_hyper_parameters"
    ]["config"]["plan"].copy()

    dic_tmp["datamodule_hyper_parameters"]["config"].keys()
    dic_tmp["datamodule_hyper_parameters"]["config"]["plan"]
# %%

# %%
    dic_tmp["hyper_parameters"]["plan"] = dic_tmp["datamodule_hyper_parameters"][
        "dataset_params"
    ].copy()
    dic_tmp["hyper_parameters"]  # .pop('plan')
    dic_tmp["datamodule_hyper_parameters"].pop("plan")
    # dic_tmp['hyper_parameters']['plan'] =dic_tmp['datamodule_hyper_parameters']['dataset_params'].copy()
    dic_tmp["datamodule_hyper_parameters"]["plan"] = dic_tmp[
        "datamodule_hyper_parameters"
    ]["dataset_params"].copy()
    dic_tmp["datamodule_hyper_parameters"]["dataset_params"]
    dic_tmp["datamodule_hyper_parameters"]["config"]["plan"]
    # dic_tmp['datamodule_hyper_parameters']['config'] = {'plan':dic_tmp['datamodule_hyper_parameters']['plan'].copy()}
    torch.save(dic_tmp, ckpt)
# %%
    insert_plan_key(ckpt)
# %%
# SECTION:-------------------- Spacing to config key-------------------------------------------------------------------------------------- <CR>

    keys = ["spacing"]
    dici = dic_tmp["datamodule_hyper_parameters"]["plan"]
    dici["spacing"] = ".8,.8,1.5"

# %%

# %%
# SECTION:-------------------- CKPT manual fix-------------------------------------------------------------------------------------- <CR>

    ckpt_fn = "/s/fran_storage/checkpoints/litsmc/litsmc/LITS-999/checkpoints/last.ckpt"
# %%
    ckp = torch.load(ckpt_fn)
    ckp.keys()
    ckp["datamodule_hyper_parameters"]["dataset_params"]["batch_size"] = 12
    ckp["datamodule_hyper_parameters"]["batch_size"] = 12
    ckp["datamodule_hyper_parameters"]["config"]
    torch.save(ckp, ckpt_fn)
# %%
    fn = "/s/fran_storage/checkpoints/lidc2/lidc2/LITS-911/checkpoints/epoch=499-step=8000.ckpt"
    std = torch.load(fn)
    ckpt_state = std["state_dict"]
    ckpt_state = remove_loss_key_state_dict(ckpt_state)
    torch.save(std, fn)

# %%
    remove_loss_key_state_dict("LITS-911")

    pp(dic_tmp.keys())
    if not "plan" in dic_tmp["datamodule_hyper_parameters"].keys():
        spacing = dic_tmp["datamodule_hyper_parameters"]["dataset_params"]["spacing"]
        dic_tmp["datamodule_hyper_parameters"]["plan"] = {"spacing": spacing}
        torch.save(dic_tmp, ckpt)
# %%
# SECTION:-------------------- filename_or_obj-------------------------------------------------------------------------------------- <CR>

# %%
    fldr = Path(
        "/s/fran_storage/datasets/preprocessed/fixed_spacing/totalseg/spc_150_150_150"
    )
    fns = list(fldr.rglob("*.pt"))
    for fn in pbar(fns):
        lm = torch.load(fn)
        lm.meta
        lm.meta["filename_or_obj"] = lm.meta["filename"]
        del lm.meta["filename"]
        torch.save(lm, fn)
# %%
