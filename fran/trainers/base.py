import time
from pathlib import Path
import ipdb
tr = ipdb.set_trace
import shutil
from fran.utils.common import COMMON_PATHS

def checkpoint_from_model_id(model_id, sort_method="last"): #CODE: Move this function to utils 
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
