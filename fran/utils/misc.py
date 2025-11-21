# %%

# training.py — minimal runner to Tm.fit()

import torch
from typing import List, Union

from utilz.string import ast_literal_eval

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


    if val in (0, 1):  # treat as explicit GPU id
        return [val]
    return val  # for 2, 3, ... treat as count of devices



def parse_devices(arg=None):
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
        return torch.device("cpu")

    n_devices = torch.cuda.device_count()

    # --- normalize to list of IDs ---
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

    devices = [torch.device(f"cuda:{i}") for i in ids]
    return devices[0] if len(devices) == 1 else devices# --- examples ---
# %%
# parse_device_arg(None)  → all GPUs (e.g., [cuda:0, cuda:1, ...])
if __name__ == '__main__':
    ast_literal_eval('[1]')
    aa = parse_devices('[1]')
    aa = parse_devices('[1]')
    # aa = parse_device_arg([0])
    # aa = parse_device_arg([0,1])
    # aa
# %%
