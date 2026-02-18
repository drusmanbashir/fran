# %%
# training.py — minimal runner to Tm.fit()
import ipdb
tr = ipdb.set_trace


import torch
from typing import List, Union

from utilz.stringz import ast_literal_eval


def is_hpc()->bool:
    import sys
    if "mpx" in sys.executable:
        return True
    return False

def parse_device_str(dev_arg: str) -> Union[int, List[int]]:
    """
    Parse device argument for Lightning Trainer.

    Rules:
      - "0" -> [0]  (CUDA:0 only)
      - "1" -> [1]  (CUDA:1 only)
      - "0,1" -> [0,1]
      - "2"  -> 2   (two devices, Lightning chooses which)
    """
    sat = str(dev_arg).strip()
    if "," in sat:
        return [int(x) for x in sat.split(",") if x != ""]
    try:
        val = int(sat)
    except ValueError:
        print("Not valid device")
        return None
        # return [0]  # default to GPU 0

    if val in (0, 1):  # treat as explicit GPU id
        return [val]
    return val  # for 2, 3, ... treat as count of devices

def convert_remapping(rem: dict|tuple|list):
    #if remapping is list/tuple -> output dict
    #if remapping is dict -> output list
    if isinstance (rem , tuple|list) and len(rem)==2:
        src = rem[0]
        dst = rem[1]
        dici = {a:b for a,b in zip(src,dst)}
        return dici
    if isinstance (rem , dict):
        src = list(rem.keys())
        dst = list(rem.values())
        list_out = [src,dst]
        return list_out



#HACK: align this and simplify this as per lightngin-ai devices arg signature
def parse_devices(arg=None, format_as_cuda=False):
    """
    Flexible device parser:
      [0]      → device('cuda:0')
      [0,1]    → [device('cuda:0'), device('cuda:1')]
      0 or 1   → device('cuda:0' / 'cuda:1')
      2        → first 2 CUDA devices  [cuda:0, cuda:1]
      None     → all available CUDA devices
      CPU fallback if CUDA unavailable
    """
    if not torch.cuda.is_available():
        print("No CUDA devices found")
        return torch.device("cpu")

    n_devices = torch.cuda.device_count()

    # --- normalize to list of IDs ---
    arg = ast_literal_eval(arg)
    if arg is None or arg == []:
        ids = list(range(n_devices))

    elif isinstance(arg, int):
        if arg < n_devices:
            # treat small integers as counts if >0
            ids = list(range(min(arg, n_devices))) if arg > 1 else [arg]
        else:
            # if arg exceeds device count, use all
            ids = list(range(n_devices))

    elif isinstance(arg, (list, tuple)):
        ids = [int(i) for i in arg]

    else:
        raise ValueError(f"Unsupported device arg: {arg}")

    # clamp to valid range
    ids = [i for i in ids if 0 <= i < n_devices]
    if not ids:
        ids = [0]

    if format_as_cuda ==True:
        ids = [torch.device(f"cuda:{i}") for i in ids]
    return ids
# %%
# parse_device_arg(None)  → all GPUs (e.g., [cuda:0, cuda:1, ...])
if __name__ == '__main__':
    ast_literal_eval('[1]')
    parse_device_str('[1]')
    aa = parse_devices('[1]')
    print(aa)
    aa = parse_devices('[1]')
    aa = parse_devices(0)
    # aa = parse_device_arg([0])
    # aa = parse_device_arg([0,1])
    # aa
# %%


# %%
# %%
# %%

