# %o%
from itertools import accumulate
from typing import Union
from fastai.torch_core import TensorBase
from fastcore.meta import delegates, use_kwargs_dict
from torch.functional import Tensor
import torch.nn as nn
import torch
import ipdb
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, DC_and_BCE_loss, SoftDiceLoss, SoftDiceLossSquared, get_tp_fp_fn_tn
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper
from fran.architectures.unet3d.model import UNet3D

from fran.utils.helpers import pp

tr = ipdb.set_trace
from fastai.losses import *
from fastcore.basics import listify, store_attr
import torch.nn.functional as F
# Cell

@delegates
class CrossEntropyLossFlat(BaseLoss):
    "Same as `nn.CrossEntropyLoss`, but flattens input and target."
    y_int = True
    @use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, *args, axis=-1, **kwargs): super().__init__(nn.CrossEntropyLoss, *args, axis=axis, **kwargs)
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)


# Cell
class FocalLoss(nn.Module):
    y_int=True
    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = 'mean') -> None:
        store_attr()

    def __call__ (self, inp: torch.Tensor, targ: torch.Tensor):
        ce_loss = F.cross_entropy(inp, targ, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t)**self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class FocalLossFlat(BaseLoss):
    """
    Same as CrossEntropyLossFlat but with focal paramter, `gamma`. Focal loss is introduced by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf. Note the class weighting factor in the paper, theta, can be
    implemented through pytorch `weight` argument passed through to F.cross_entropy.
    """
    y_int = True
    @use_kwargs_dict(keep=True, weight=None, reduction='mean')
    def __init__(self, *args, gamma=2.0, axis=-1, **kwargs):
        super().__init__(FocalLoss, *args, gamma=gamma, axis=axis, **kwargs)

    def decodes(self, x): return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)

class CrossEntropyLoss_ub(nn.Module): # needed because my data masks have channel dimension which needs to be squeezed before calling CrossEntropyLoss
    def __init__(self, *args,**kwargs) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(*args,**kwargs)
    def forward(self,input,target):
        if not target.dtype == torch.long:
            target=target.long()
        if input.ndim == target.ndim:
            target= target.squeeze(1)
        return self.loss(input,target)
# %
class DiceLoss_ub:
    "Dice loss for segmentation"
    def __init__(self, axis=1, smooth=1e-6, reduction="mean", square_in_union=False, include_bg=False):
        store_attr()
    def __call__(self, pred, targ): # does a batch loss
        targ = self._one_hot(targ, pred.shape[self.axis])
        assert pred.shape == targ.shape, 'input and target dimensions differ, DiceLoss expects non one-hot targs'
        pred = torch.cat([t for t in pred],1).unsqueeze(0)
        targ= torch.cat([t for t in targ],1).unsqueeze(0)
        sum_dims = list(range(2, len(pred.shape)))

        inter = torch.sum(pred*targ, dim=sum_dims)
        union = (torch.sum(pred**2+targ, dim=sum_dims) if self.square_in_union
            else torch.sum(pred+targ, dim=sum_dims))

        dice_score = ((2. * inter + self.smooth)/(union + self.smooth)).flatten() # verified correct code. 

        if self.include_bg==False:
            dice_score = dice_score[1:]
        loss = -dice_score
        if self.reduction == 'mean':
            loss_final = loss.mean()  
        elif self.reduction == 'sum':
            loss_final = loss.sum()
        dice_dic = {'loss_dice':loss_final}
        dice_dic.update({'loss_dice_label'+str(ind+(not self.include_bg)):loss[ind] for ind in range(len(loss))})
        return dice_dic
    @staticmethod
    def _one_hot(x, classes, axis=1):
        "Creates one binay mask per class"
        return torch.stack([torch.where(x==c, 1, 0) for c in range(classes)], axis=axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)
    def decodes(self, x):    return x.argmax(dim=self.axis)

class CombinedLoss(nn.Module):
    "Dice, CE and Focal combined"
    def __init__(self, axis=1, smooth=1., theta=.5, apply_activation=True,ce_or_focal='ce', weight: Union[list,Tensor]=None,return_single_loss=False,**kwargs_dice):
        super().__init__()
        assert ce_or_focal in ['ce','focal'], "Either choose 'ce' or 'focal' "
        store_attr(but='weight')
        if weight:
            weight = torch.tensor(weight,dtype=torch.float, device=torch.cuda.current_device())
        self.dice_loss =  DiceLoss_ub(axis, smooth,**kwargs_dice)
        if ce_or_focal=='focal':
            self.loss2 = FocalLossFlat(axis=axis, weight=weight)
            self.targ_dtype = torch.uint8
        else:
            self.loss2 = nn.CrossEntropyLoss(weight=weight)
            self.targ_dtype = torch.int64
        
    def forward(self, pred, targ):
        if isinstance(pred,list):
            pred = pred[0]
        if targ.ndim==pred.ndim: # i.e., channel target should NOT have channel dim
            targ= targ.squeeze(1)
        l1 = self.loss2(pred, targ.type(self.targ_dtype))
        if self.apply_activation == True:
            pred = self.activation(pred)
        l2 = self.dice_loss(pred, targ)
        final =self.theta*l1 + (1-self.theta)*l2['loss_dice']
        loss = {'loss':final}
        if self.return_single_loss==False:
            loss.update({"loss_ce_focal":l1,**l2})
        return loss

    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis) 

class MultipleOutputLoss2_nnunet(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        this will output a dict
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super().__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors
        
        #loss at full res
        loss_full_res = self.loss(x[0],y[0])

        l = [[weights[0] *loss_item for loss_item in loss_full_res.values()]]

        #iterative applies the loss at all resolutions and collects the results
        for i in range(1, len(x)):
            if weights[i] != 0:
                l.append([weights[i] *loss_item for loss_item in self.loss(x[i], y[i]).values()])

        #summs results and produces a dictionary
        losses_summed={}
        for els in zip(loss_full_res.keys(),*l):
            losses_summed.update({els[0]:sum(els[1:])})

        return losses_summed
# %%
def setup_multioutputloss_nnunet(net_numpool ,batch_dice=True):  # verified to work like nnunet
            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            loss = CombinedLoss()
            return  MultipleOutputLoss2_nnunet(loss, weights)


# %%
if __name__ == "__main__":

        softmax_helper = lambda x: F.softmax(x, 1)
        img = torch.load('/home/ub/Dropbox/code/fran/tmp/img.pt')
        mask = torch.load('/home/ub/Dropbox/code/fran/tmp/mask.pt')
        pred = torch.load('/home/ub/Dropbox/code/fran/tmp/pred.pt')
# %%
        loss_func = setup_multioutputloss_nnunet(5)
        l = loss_func(pred,mask)
        pp(l)
# %%
# %%
        dice_ub = DiceLoss_ub()
        dice_nnunet  = SoftDiceLoss(do_bg = False,smooth=1e-5,batch_dice=True, apply_nonlin=softmax_helper)
        
# %%
        targ = mask[0].clone()
        targ = targ.squeeze(1)
        dice_ub(pred[0],targ)
        loss_func = CombinedLoss(ce_or_focal="ce")
        loss_func(pred[0],targ)
        
        loss2 = DC_and_CE_loss_nnunet({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        loss2(pred[0],mask[0])
        dice_nnunet(pred[0],mask[0])
        ce_loss2 = loss2.ce(pred[0], mask[0][:, 0].long())
        print(ce_loss)
# %%

        smooth = 1e-5
        loss_mask = None
        apply_nonlin = softmax_helper
        do_bg = False
        batch_dice = True


# %%
        import time
        start= time.time()
        x = pred[0].clone()
        y = mask[0].clone()
        shp_x = x.shape
        if batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if apply_nonlin is not None:
            x = apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + smooth
        denominator = 2 * tp + fp + fn + smooth

        dc = nominator / (denominator + 1e-8)
