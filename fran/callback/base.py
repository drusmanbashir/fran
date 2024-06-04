# %%

from fastai.callback.fp16 import MixedPrecision
from fastai.callback.schedule import CancelFitException
from lightning.pytorch.callbacks import Callback
from fastai.callback.core import Callback as CBF
import torch.nn.functional as F
from fastcore.basics import listify, store_attr
import torch.nn as nn
import torchvision
import torch
import matplotlib.pyplot as plt
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
    order =3 #after DropBBox
    def __init__(self, ds_scales):
        self.ds_scales = ds_scales

    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        mask = self.learn.y
        output = []
        for s in self.ds_scales:
            if all([i == 1 for i in s]):
                output.append(mask)
            else:
                size = [round(ss*aa) for ss,aa in zip(s,mask.shape[2:])]
                mask_downsampled = F.interpolate(mask,size=size,mode="nearest")
                output.append(mask_downsampled)
        self.learn.yb = [output]
        
class DownsampleMaskForDS_Fastai(CBF):
    def __init__(self, ds_scales):
        self.ds_scales = ds_scales

    def before_batch(self):
        mask = self.learn.y
        output = []
        for s in self.ds_scales:
            if all([i == 1 for i in s]):
                output.append(mask)
            else:
                size = [round(ss*aa) for ss,aa in zip(s,mask.shape[2:])]
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

   

    
class TerminateOnNaNCallback_ub(CBF):
    "A `Callback` that terminates training if loss is NaN."

    order = -9
    def after_batch(self):
        "Test if `last_loss` is NaN and interrupts training."
        if torch.isinf(self.loss) or torch.isnan(self.loss):
            print("NaNs !!")
            raise CancelFitException

# Cell
class GradientClip(CBF):
    "Clip norm of gradients"
    order = MixedPrecision.order + 1

    def __init__(self, max_norm: float = 1., norm_type: float = 2.0):
        store_attr()

    def before_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), self.max_norm, self.norm_type, error_if_nonfinite=True)




