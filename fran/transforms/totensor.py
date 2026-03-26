# %%
from pathlib import Path
from typing import Union

import ipdb
import numpy as np
import SimpleITK as sitk
import torch
from fasttransform.transform import Transform
from torch.functional import Tensor

tr = ipdb.set_trace


class ToTensorT(Transform):
    def __init__(self, encode_dtype=None):
        self.encode_dtype = encode_dtype

    "Convert item to appropriate tensor class"
    order = 0

    def decodes(self, x, decode_type: np.ndarray):
        return np.array(x)

    def encodes(self, x: Tensor):
        return x

    def encodes(self, x: np.ndarray):
        if x.dtype == np.uint16:
            x = x.astype(np.uint8)
        x_pt = torch.tensor(x)
        return x_pt

    def encodes(self, x: sitk.Image):
        x_np = sitk.GetArrayFromImage(x)
        x_pt = torch.tensor(x_np, dtype=self.encode_dtype)
        return x_pt

    def encodes(self, x: sitk.Image):
        x_np = sitk.GetArrayFromImage(x)
        if x_np.dtype == np.uint16:
            x_np = x_np.astype(np.uint8)
        x_pt = torch.tensor(x_np, dtype=self.encode_dtype)
        return x_pt

    def encodes(self, x: Union[Path, str]):
        x_sitk = sitk.ReadImage(x)
        x_np = sitk.GetArrayFromImage(x_sitk)
        if x_np.dtype == np.uint16:
            x_np = x_np.astype(np.uint8)
        x_pt = torch.tensor(x_np, dtype=self.encode_dtype)
        return x_pt


# %%
if __name__ == "__main__":
    x = np.random.rand(10, 10)
    T = ToTensorT()
    print(type(T.encodes(x)))

# %%
