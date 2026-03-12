import time
import re
import torch
from pathlib import Path
from typing import Union
import ipdb
from utilz.cprint import cprint
from utilz.stringz import headline
tr = ipdb.set_trace
import shutil
from fran.utils.common import COMMON_PATHS


def checkpoint_from_model_id_remote(model_id, project, remote_dir):

        remote_dir =str(Path(self.model_checkpoint).parent)
        remote_dir = str(Path(ckpt_path).parent)
        latest_ckpt = self.shadow_remote_ckpts(remote_dir)
        hpc_settings = load_yaml(os.environ["HPC_SETTINGS"])
        local_dir = self.project.checkpoints_parent_folder / self.run_id / "checkpoints"
        print(f"\nSSH to remote folder {remote_dir}")

        client = SSHClient()
        client.load_system_host_keys()
        client.connect(
            hpc_settings["host"],
            username=hpc_settings["username"],
            password=hpc_settings["password"],
        )

        ftp_client = client.open_sftp()
        try:
            fnames = []
            for f in sorted(
                ftp_client.listdir_attr(remote_dir),
                key=lambda k: k.st_mtime,
                reverse=True,
            ):
                fnames.append(f.filename)
        except FileNotFoundError:
            print("\n------------------------------------------------------------------")
            print(f"Error:Could not find {remote_dir}.\nIs this a remote folder and exists?\n")
            return

        remote_fnames = [os.path.join(remote_dir, f) for f in fnames]
        local_fnames = [os.path.join(local_dir, f) for f in fnames]
        maybe_makedirs(local_dir)
        downloaded_files = []
        for rem, loc in zip(remote_fnames, local_fnames):
            if Path(loc).exists():
                print(f"Local file {loc} exists already.")
                downloaded_files.append(loc)
            else:
                print(f"Copying file {rem} to local folder {local_dir}")
                ftp_client.get(rem, loc)
                downloaded_files.append(loc)

        if not downloaded_files:
            return None
        latest_ckpt = max(downloaded_files, key=lambda f: Path(f).stat().st_mtime)
        return latest_ckpt



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

    list_of_files = [f for f in project_fldr.glob("*.ckpt") if f.is_file()]
    if len(list_of_files) == 0:
        raise Exception("No checkpoint files found {}".format(project_fldr))
    if sort_method == "last":
        candidates = sorted(list_of_files, key=lambda p: p.stat().st_mtime, reverse=True)
        ckpt = candidates[0]
        try:
            torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception as e1:
            cprint("Bad checkpoint file: {} ({})".format(ckpt, e1), color="yellow")
            if len(candidates) < 2:
                raise Exception("Latest checkpoint is bad and no fallback exists: {}".format(ckpt)) from e1
            ckpt = candidates[1]
            try:
                torch.load(ckpt, map_location="cpu", weights_only=False)
            except Exception as e2:
                raise Exception(
                    "Latest two checkpoints are unreadable: {} ({}) ; {} ({})".format(
                        candidates[0], e1, candidates[1], e2
                    )
                ) from e2
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

