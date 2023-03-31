# %o%
from pipe import map  as pmap
from monai.utils.enums import DiceCEReduction, LossReduction
from monai.utils.module import look_up_option
from typing import Callable, Optional
import torch.nn as nn
import itertools as il

import torch
import ipdb

from nnunet.utilities.nd_softmax import softmax_helper

from fran.utils.helpers import range_inclusive

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
        return total_loss, ce_loss, dice_loss_reduced , dice_loss_unreduced.squeeze(2).squeeze(2).squeeze(2)


class CombinedLoss(_DiceCELossMultiOutput):
    def __init__(self,bs,fg_classes,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.create_labels(bs,fg_classes)

    def forward(self,input,target):
        losses = super().forward(input,target)
        self.set_loss_dict(losses)
        return losses[0]

    def set_loss_dict(self,l):
        dice_unreduced = l[-1].clone().detach()
        dice_labels = dice_unreduced.mean(0)
        inferred_classes = range(len(dice_labels))
        self.labels = ['loss','loss_ce','loss_dice'] + ['loss_dice_label{}'.format(x) for x in inferred_classes]
        self.loss_dict = {x: y.item() for x, y in zip(self.labels,il.chain(l[:3],dice_labels))}

    def set_loss_dict(self,l):
        class_losses = l[-1].mean(0)
        separate_case_losses = list(l[-1].view(-1))
        self.loss_dict = {x: y.item() for x, y in zip(self.labels, il.chain(l[:3],class_losses,separate_case_losses))}



    def create_labels(self,bs,fg_classes):
       label_maker = lambda x: "loss_dice_batch{0}_label{1}".format(*x)
       batches = list(range(bs))
       classes = range_inclusive(1,fg_classes)

       self.neptune_labels =  ["loss", "loss_ce","loss_dice"] + ["loss_dice_label{}".format(x) for x in classes]

       self.case_recorder_labels = list(il.product(batches,classes) | pmap (label_maker))
       self.labels =list(il.chain(self.neptune_labels,self.case_recorder_labels))



class DeepSupervisionLoss(nn.Module):
    def __init__(self, levels: int, bs:int, fg_classes: int,device=None):
        super().__init__()
        weights = torch.tensor([1 / (2**i) for i in range(levels)],device=device)

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        mask = torch.tensor(
            [True] + [True if i < levels - 1 else False for i in range(1, levels)]
        )
        weights[~mask] = 0
        self.create_labels(bs,fg_classes)
        self.weights = weights / weights.sum()
        self.LossFunc = _DiceCELossMultiOutput(include_background=False,softmax=True)

    def create_labels(self,bs,fg_classes):
       label_maker = lambda x: "loss_dice_batch{0}_label{1}".format(*x)
       batches = list(range(bs))
       classes = range_inclusive(1,fg_classes)

       self.neptune_labels =  ["loss", "loss_ce","loss_dice"] + ["loss_dice_label{}".format(x) for x in classes]

       self.case_recorder_labels = list(il.product(batches,classes) | pmap (label_maker))
       self.labels =list(il.chain(self.neptune_labels,self.case_recorder_labels))


    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        # loss at full res

        losses = [self.LossFunc(xx, yy) for xx, yy in zip(x, y)]
        self.set_loss_dict(losses[0])
        losses_weighted = torch.stack(
            [self.weights * loss[0] for loss in losses]
        )  # total_loss * weight for each level
        losses_weighted_summed = losses_weighted.sum()
        return losses_weighted_summed

    def set_loss_dict(self,l):
        class_losses = l[-1].mean(0)
        separate_case_losses = list(l[-1].view(-1))
        self.loss_dict = {x: y.item() for x, y in zip(self.labels, il.chain(l[:3],class_losses,separate_case_losses))}

# %%
if __name__ == "__main__":
    softmax_helper = lambda x: F.softmax(x, 1)
    targ = torch.load("fran/tmp/ed.pt")
    pred = torch.load("fran/tmp/pred.pt")
# %%
    DS = DeepSupervisionLoss(levels=3,bs=4, fg_classes=2)

    x = [pred.cuda()]*3
    y = [targ.cuda()]*3


# %%

