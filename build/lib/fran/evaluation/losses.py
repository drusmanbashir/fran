# %%

from fastcore.basics import  store_attr
import numpy as np
from monai.utils.enums import DiceCEReduction, LossReduction
import lightning.pytorch as pl
from monai.utils.module import look_up_option
from typing import Callable, Optional
import torch.nn as nn
import itertools as il

import torch
import ipdb

from nnunet.utilities.nd_softmax import softmax_helper
from torchvision.utils import Union

from utilz.helpers import range_inclusive

tr = ipdb.set_trace
from monai.losses import DiceLoss, DiceCELoss
import torch.nn.functional as F

# Cell

class _DiceCELossMultiOutput(DiceCELoss):
    def __init__(
        self,
        include_background: bool = True,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        super().__init__()
        dice_reduction = LossReduction.NONE
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=True,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=dice_reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        # self.reduction=reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss_unreduced = self.dice(input, target)
        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean
        else:
            f = torch.sum
        dice_loss_reduced = f(dice_loss_unreduced)
        ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = (
            self.lambda_dice * dice_loss_reduced + self.lambda_ce * ce_loss
        )
        # return total_loss, ce_loss, dice_loss_reduced , dice_loss_unreduced.squeeze((2,3,4))
        losses_for_logging_only = [ ce_loss.detach(), dice_loss_reduced.detach() , dice_loss_unreduced.squeeze(2).squeeze(2).squeeze(2).detach()]

        return total_loss,*losses_for_logging_only


class CombinedLoss(_DiceCELossMultiOutput):
    def __init__(self,fg_classes,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def forward(self,input,target):
        losses = super().forward(input,target)
        if not hasattr(self,'labels'):
            bs = target.shape[0]
            self.create_labels(bs,self.fg_classes)
        self.set_loss_dict(losses)
        return losses[0]

    def set_loss_dict(self,l):
        l[1:] = [ll.detach() for ll in l[1:]] # only l[0] needs gradient. rest are for logging
        class_losses = l[-1].mean(0)
        separate_case_losses = list(l[-1].view(-1))
        self.loss_dict = {x: y.item() for x, y in zip(self.labels, il.chain(l[:3],class_losses,separate_case_losses))}



    def create_labels(self,bs,fg_classes):
       label_maker = lambda x: "loss_dice_batch{0}_label{1}".format(*x)
       batches = list(range(bs))
       classes = range_inclusive(1,fg_classes)

       self.neptune_labels =  ["loss", "loss_ce","loss_dice"] + ["loss_dice_label{}".format(x) for x in classes]

       self.case_recorder_labels = il.product(batches,classes) 
       self.case_recorder_labels= map(label_maker,self.case_recorder_labels)
       self.labels =list(il.chain(self.neptune_labels,self.case_recorder_labels))



class DeepSupervisionLoss(pl.LightningModule):
    def __init__(self, levels: int,deep_supervision_scales,  fg_classes: int):
        super().__init__()
        store_attr()
        self.LossFunc = _DiceCELossMultiOutput(include_background=False,softmax=True)

    def create_labels(self,bs, fg_classes):
       label_maker = lambda x: "loss_dice_batch{0}_label{1}".format(*x)
       classes = range_inclusive(1,fg_classes)

       batches = list(range(bs))
       self.neptune_labels =  ["loss", "loss_ce","loss_dice"] + ["loss_dice_label{}".format(x) for x in classes]

       self.case_recorder_labels = list(il.product(batches,classes) )
       self.case_recorder_labels= map(label_maker,self.case_recorder_labels)
       self.labels =list(il.chain(self.neptune_labels,self.case_recorder_labels))


    def create_weights(self,device):
        weights = torch.tensor([1 / (2**i) for i in range(self.levels)],requires_grad=False,device=device)

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        mask = torch.tensor(
            [True] + [True if i < self.levels - 1 else False for i in range(1, self.levels)]
        )
        weights[~mask] = 0
        self.weights = weights / weights.sum()


    def maybe_create_labels(self,target):
        if not hasattr(self,'labels'):
            if isinstance(target, list):
                bs = target[0].shape[0]
            else:
                bs = target.shape[0]
            self.create_labels(bs,self.fg_classes)

    def maybe_create_weights(self,preds):
        if not hasattr(self,'weights'):
            self.create_weights(preds[0].device)

    #
    # def apply_ds_scales(self, preds, target):
    #         target_listed = []
    #         preds_listed = []
    #         preds = torch.unbind(preds, dim=1)
    #         for ind,sc in enumerate(self.deep_supervision_scales):
    #             pred = preds[ind]
    #             if all([i == 1 for i in sc]):
    #                 target_listed.append(target)
    #                 preds_listed.append(pred)
    #             else:
    #                 t_shape = list(target.shape[2:])
    #                 size = [int(np.round(ss * aa)) for ss, aa in zip(sc, t_shape)]
    #                 target_downsampled = F.interpolate(target, size=size, mode="nearest")
    #                 pred_downsampled = F.interpolate(pred, size=size, mode="trilinear")
    #                 target_listed.append(target_downsampled)
    #                 preds_listed.append(pred_downsampled)
    #         target = target_listed
    #         preds = preds_listed
    #         return preds,target
    #
    #
    def apply_ds_scales(self,tnsr_inp:Union[list,torch.Tensor],mode):
            tnsr_listed = []
            for ind,sc in enumerate(self.deep_supervision_scales):
                if isinstance(tnsr_inp, Union[tuple,list]):
                    tnsr = tnsr_inp[ind]
                else:
                    tnsr = tnsr_inp
                if all([i == 1 for i in sc]):
                    tnsr_listed.append(tnsr)
                else:
                    t_shape = list(tnsr.shape[2:])
                    size = [int(np.round(ss * aa)) for ss, aa in zip(sc, t_shape)]
                    tnsr_downsampled = F.interpolate(tnsr, size=size, mode=mode)
                    tnsr_listed.append(tnsr_downsampled)
            return tnsr_listed



    def forward(self, preds, target):

        self.maybe_create_weights(preds)
        self.maybe_create_labels(target)
        
        if isinstance(preds, Union[list,tuple]) and isinstance(target, torch.Tensor): # multires lists in training
            target = self.apply_ds_scales(target, "nearest")
            losses = [self.LossFunc(xx, yy) for xx,yy in zip(preds,target)]

        elif isinstance(preds, torch.Tensor):  #tensor
            if preds.dim() == target.dim()+1:# i.e., training phase has deep supervision:
                preds = torch.unbind(preds, dim=1)
                preds = self.apply_ds_scales(preds,"trilinear")
                target = self.apply_ds_scales(target, "nearest")
                losses = [self.LossFunc(xx, yy) for xx,yy in zip(preds,target)]
            else:  #validation loss
                losses = [self.LossFunc(preds, target)]
        else:
            raise NotImplementedError("preds must be either list or stacked tensor")

        self.set_loss_dict(losses[0])
        losses_weighted = torch.stack(
            [self.weights * loss[0] for loss in losses]
        )  # total_loss * weight for each level
        losses_out = losses_weighted.sum()
        return losses_out

    def set_loss_dict(self,l):
        class_losses = l[-1].mean(0)
        separate_case_losses = list(l[-1].view(-1))
        self.loss_dict = {x: y.item() for x, y in zip(self.labels, il.chain(l[:3],class_losses,separate_case_losses))}

# %%
if __name__ == "__main__":
    softmax_helper = lambda x: F.softmax(x, 1)
    # targ = torch.load("fran/tmp/ed.pt")
    # pred = torch.load("fran/tmp/pred.pt")
# %%



# %%

