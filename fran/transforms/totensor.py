# %%
from functools import wraps
from pathlib import Path
from typing import Union
from fastcore.basics import GetAttr

import numpy as np
import SimpleITK as sitk
from batchgenerators.dataloading.multi_threaded_augmenter import torch
from fastcore.transform import ItemTransform, Transform, store_attr
from torch.functional import Tensor
from fran.transforms.base import KeepBBoxTransform


import ipdb
tr = ipdb.set_trace

# %%
class ToTensorT(Transform):
    def __init__(self,encode_dtype=None):store_attr()
    "Convert item to appropriate tensor class"
    order = 0


def enc_wrapper(encode_func):
        @wraps(encode_func)
        def _inner(cls, x):
            cls.decode_type = type(x)
            return encode_func(cls,x)
        return _inner
def dec_wrapper(decode_func):
    def _inner(cls,x):
        return decode_func(cls,x,cls.decode_type)
    return _inner

@ToTensorT
@enc_wrapper
def encodes(self,x:Tensor): return x

@ToTensorT
@enc_wrapper
def encodes(self,x:np.ndarray): 
    if x.dtype == np.uint16:
       x = x.astype(np.uint8)
    x_pt = torch.tensor(x)
    return x_pt

@ToTensorT
@dec_wrapper
def decodes(self,x,decode_type:np.ndarray):
    return np.array(x)

@ToTensorT
@enc_wrapper
def encodes(self,x:sitk.Image): 
   x_np = sitk.GetArrayFromImage(x)
   x_pt = torch.tensor(x_np,dtype=self.encode_dtype)
   return x_pt

@ToTensorT
@enc_wrapper
def encodes(self,x:sitk.Image): 
   x_np = sitk.GetArrayFromImage(x)
   if x_np.dtype == np.uint16:
        x_np = x_np.astype(np.uint8)
   x_pt = torch.tensor(x_np,dtype=self.encode_dtype)
   return x_pt


@ToTensorT
@enc_wrapper
def encodes(self,x:Union[Path,str]): 
   x_sitk = sitk.ReadImage(x)
   x_np = sitk.GetArrayFromImage(x_sitk)
   if x_np.dtype == np.uint16:
        x_np = x_np.astype(np.uint8)
   x_pt = torch.tensor(x_np,dtype=self.encode_dtype)
   return x_pt

class ToTensorImgMask(ItemTransform):
    def encodes(self,x):
        x = [ToTensorT()(xx) for xx in x]
        return x


class ToTensorI(KeepBBoxTransform,GetAttr):
    '''
    works on img/bbox pair
    '''
    order = 0
    def func(self,img:np.ndarray):
        return torch.tensor(img)
# %%
if __name__ == "__main__":
   T= ToTensorI
   x = np.random.rand(10,10)
   T = ToTensorT()
   print(type(T.encodes(x)))

