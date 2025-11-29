
# metatensor_to_cpp.py
import sys, io
import torch
from monai.data import MetaTensor
import numpy as np


def to_list(x):
    # Convert NumPy arrays or tensors or lists into pure Python lists of floats
    if isinstance(x, np.ndarray):
        return  
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(float).tolist()
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return float(x)

def convert(img_fn: str, out_path: str):
    obj = torch.load(img_fn, map_location="cpu", weights_only=False)

    if isinstance(obj, MetaTensor):
        t = torch.Tensor(obj)
        meta = dict(obj.meta)
    elif isinstance(obj, torch.Tensor):
        t = obj
        meta = {}
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")

    spacing = meta.get("spacing", [1.0,1.0,1.0])
    spacing = torch.Tensor(spacing)
    origin  = meta.get("origin",  [0.0,0.0,0.0])
    origin = torch.Tensor(origin)
    direction = meta.get("affine")
    direction = torch.Tensor(direction)

    torch.save({
        "data": t,
        "meta": {
            "spacing": spacing[:3],
            "origin": origin[:3],
            "direction": direction
        }
    }, out_path)

# get_metadata.py
import io

def convert_to_bytes(img_fn: str) -> bytes:
    obj = torch.load(img_fn, map_location="cpu", weights_only=False)

    if isinstance(obj, MetaTensor):
        t = torch.Tensor(obj)
        meta = dict(obj.meta)
    elif isinstance(obj, torch.Tensor):
        t = obj
        meta = {}
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")

    spacing   = torch.as_tensor(meta.get("spacing", [1.0, 1.0, 1.0]),
                                dtype=torch.float64)
    origin    = torch.as_tensor(meta.get("origin",  [0.0, 0.0, 0.0]),
                                dtype=torch.float64)
    direction = torch.as_tensor(meta.get("affine",  np.eye(4)),
                                dtype=torch.float64)

    clean = {
        "data": t,
        "meta": {
            "spacing":   spacing[:3],
            "origin":    origin[:3],
            "direction": direction,
        },
    }

    buf = io.BytesIO()
    torch.save(clean, buf)
    return buf.getvalue()
# %%
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python get_metadata.py <input.pt> <output.pt>")
        sys.exit(1)
#
#     convert(sys.argv[1], sys.argv[2])
# #
#
#     img_fn = "/r/datasets/preprocessed/lidc/lbd/spc_080_080_150_ric8c38fe68_ex000/lms/lidc_0011.pt"
# #
#     obj = torch.load(img_fn, map_location="cpu", weights_only=False)
# #     t = torch.Tensor(obj)
# #     meta = obj.meta
# #     out_path = "/tmp/tmp12.pt"
# #     convert(img_fn, out_path)
#     out_path = "/tmp/cpp_tnsr.pt"
# #
#     convert(img_fn, out_path)
#     t = torch.load(out_path, map_location="cpu", weights_only=False)
#     print(type(t['data']))
# #
# # %%
