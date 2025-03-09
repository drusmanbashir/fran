from pathlib import Path
import ipdb
tr = ipdb.set_trace
from fran.utils.common import COMMON_PATHS

def checkpoint_from_model_id(model_id, sort_method="last"):
    fldr = Path(COMMON_PATHS["checkpoints_parent_folder"])
    all_fldrs = [
        f for f in fldr.rglob("*{}/checkpoints".format(model_id)) if f.is_dir()
    ]
    if len(all_fldrs) == 1:
        fldr = all_fldrs[0]
    else:
        print(
            "no local files. Model may be on remote path. use download_neptune_checkpoint() "
        )
        tr()

    list_of_files = list(fldr.glob("*"))
    if sort_method == "last":
        ckpt = max(list_of_files, key=lambda p: p.stat().st_mtime)
    elif sort_method == "best":
        tr()
    return ckpt
