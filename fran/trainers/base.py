from pathlib import Path
import ipdb
tr = ipdb.set_trace
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

    list_of_files = list(project_fldr.glob("*"))
    if sort_method == "last":
        ckpt = max(list_of_files, key=lambda p: p.stat().st_mtime)
    elif sort_method == "best":
        tr()
    return ckpt
