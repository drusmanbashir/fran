# %%
from fastai.callback.tracker import TrackerCallback
import pandas as pd
from torchinfo import summary

from fastai.callback.core import Callback, CancelFitException
from fastai.callback.fp16 import MixedPrecision
import torch.nn.functional as F
import os
from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p
from pathlib import Path
from fastai.callback.core import Callback
from fastcore.basics import listify, store_attr
from neptune.new.types.atoms.file import File
import torch.nn as nn
from torchvision.utils import make_grid
import torchvision
import torch
import matplotlib.pyplot as plt
from fran.transforms.spatialtransforms import one_hot
import neptune.new as neptune
import ipdb
tr = ipdb.set_trace
import ray
tr2 = ray.util.pdb.set_trace
import numpy as np
# %%

class PredAsList(Callback):
    def after_pred(self):
        self.learn.pred= listify(self.learn.pred)
    def after_loss(self):
        self.learn.pred = self.learn.pred[0]

class DownsampleMaskForDS(Callback):
    def __init__(self, ds_scales):
        self.ds_scales = ds_scales

    def before_batch(self):
        mask = self.learn.y
        output = []
        for s in self.ds_scales:
            if all([i == 1 for i in s]):
                output.append(mask)
            else:
                size = [np.round(ss*aa).astype(int) for ss,aa in zip(s,mask.shape[2:])]
                mask_downsampled = F.interpolate(mask,size=size,mode="nearest")
                output.append(mask_downsampled)
        self.learn.yb = [output]
        

class FixPredNan(Callback):
    "A `Callback` that terminates training if loss is NaN."
    order = -9

    def after_pred(self):
        self.learn.pred = torch.nan_to_num(self.learn.pred, nan=0.5)
        "Test if `last_loss` is NaN and interrupts training."

def make_grid_5d_input(a:torch.Tensor,batch_size_to_plot=16):
    '''
    this function takes in a 5d tensor (BxCxDxWXH) e.g., shape 4,1,64,128,128)
    and creates a grid image for tensorboard
    '''
    middle_point= int(a.shape[2]/2)
    middle = slice(int(middle_point-batch_size_to_plot/2), int(middle_point+batch_size_to_plot/2))
    slc = [0,slice(None), middle,slice(None),slice(None)]
    img_to_save= a [slc]
# BxCxHxW
    img_to_save2= img_to_save.permute(1,0,2,3)  # re-arrange so that CxBxHxW (D is now minibatch)
    img_grid = torchvision.utils.make_grid(img_to_save2,nrow=int(batch_size_to_plot))
    return img_grid


def make_grid_5d_input_numpy_version(a:torch.Tensor,batch_size_to_plot=16):
    img_grid = make_grid_5d_input(a)
    img_grid_np = img_grid.cpu().detach().permute(1,2,0).numpy()
    plt.imshow(img_grid_np)

   

    
class TerminateOnNaNCallback_ub(Callback):
    "A `Callback` that terminates training if loss is NaN."

    order = -9
    def after_batch(self):
        "Test if `last_loss` is NaN and interrupts training."
        if torch.isinf(self.loss) or torch.isnan(self.loss):
            print("NaNs !!")
            raise CancelFitException

# Cell
class GradientClip(Callback):
    "Clip norm of gradients"
    order = MixedPrecision.order + 1

    def __init__(self, max_norm: float = 1., norm_type: float = 2.0):
        store_attr()

    def before_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), self.max_norm, self.norm_type, error_if_nonfinite=True)


