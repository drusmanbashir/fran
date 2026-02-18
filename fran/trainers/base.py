import time
import re
import torch
from pathlib import Path
from typing import Union
import ipdb
from utilz.stringz import headline
tr = ipdb.set_trace
import shutil
from fran.utils.common import COMMON_PATHS

def checkpoint_from_model_id(model_id, sort_method="last", normalize_keys=True): #CODE: Move this function to utils 
    fldr = Path(COMMON_PATHS["checkpoints_parent_folder"])
    project_fldrs = [f for f in fldr.rglob(model_id) if f.is_dir()]
    if len(project_fldrs) > 1:
        raise Exception(
            "No local files. Model may be on remote path. use download_neptune_checkpoint() \n{}".format(project_fldrs)
        )
    elif len(project_fldrs) == 0:
        raise Exception("No project found {}".format(model_id))
    project_fldr = project_fldrs[0]/("checkpoints")

    list_of_files = [f for f in project_fldr.glob("*") if f.is_file()]
    if sort_method == "last":
        ckpt = max(list_of_files, key=lambda p: p.stat().st_mtime)
    elif sort_method == "best":
        tr()
    if normalize_keys==True:
        ckpt = write_normalized_ckpt(ckpt)
    return ckpt

def backup_ckpt(ckpt):
    """Create a backup copy of current ckpt under a `backup/` subfolder."""
    if not ckpt:
        return None
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        return None

    backup_dir = ckpt_path.parent / "backup"
    backup_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"{ckpt_path.stem}.{timestamp}{ckpt_path.suffix}"
    shutil.copy2(ckpt_path, backup_path)
    print(f"Backup created: {backup_path}, before overriding ckpt configuration")
    return backup_path


def normalize_orig_mod_prefix(sd: dict) -> dict:
    pat = re.compile(r"^(model)(?:\._orig_mod)+(\.)")
    return {pat.sub(r"\1\2", k): v for k, v in sd.items()}


def write_normalized_ckpt(ckpt_path: Union[str, Path]) -> Path:
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu",weights_only=False)
    st = ckpt.get("state_dict", {})
    if any(k.startswith("model._orig_mod") for k in st):
        ckpt["state_dict"] = normalize_orig_mod_prefix(st)
        out = ckpt_path.with_suffix(".norm.ckpt")
        torch.save(ckpt, out)
        return out
    return ckpt_path



def fix_dict_keys(input_dict, old_string, new_string):
    output_dict = {}
    for key in input_dict.keys():
        neo_key = key.replace(old_string, new_string)
        output_dict[neo_key] = input_dict[key]
    return output_dict

def switch_state_keys(state_dict)->dict:
        ckpt_state = state_dict["state_dict"]
        k1 = list(ckpt_state.keys())[0]
        k1_splits = k1.split(".")
        
        if k1_splits[1]  == "_orig_mod":
            bad_str = "model._orig_mod"
            good_str = "model"
        else:
            bad_str = "model"
            good_str = "model._orig_mod"
        
        ckpt_state_updated = fix_dict_keys(ckpt_state, bad_str, good_str)
        state_dict_neo = state_dict.copy()
        state_dict_neo["state_dict"] = ckpt_state_updated

        headline ( "Switch keys from {} to {}".format(bad_str, good_str) )
        return state_dict_neo

def switch_ckpt_keys(ckpt_path: Union[str, Path])->None:
        state_dict = torch.load(ckpt_path)
        state_dict_neo = switch_state_keys(state_dict)

        ckpt_old = str(ckpt_path).replace(".ckpt", ".ckpt_bkp")
        shutil.move(ckpt_path, ckpt_old)
        torch.save(state_dict_neo, ckpt_path)
        # print(ckpt_state_updated.keys())
        print ( "Old ckpt saved as: {}".format(ckpt_old) )


# %%

if __name__ == '__main__':
    ckpt_path = "/s/fran_storage/checkpoints/nodes/nodes/LITS-1290/checkpoints/last.ckpt"
    ckpt_path2 = "/s/fran_storage/checkpoints/nodes/nodes/LITS-1290/checkpoints/last.ckpt_bkp_bkp_bkp"
    state_dict = torch.load(ckpt_path)
    ckpt_state = state_dict["state_dict"]
    k1 = list(ckpt_state.keys())[0]
    k1_splits = k1.split(".")

    print(k1)

