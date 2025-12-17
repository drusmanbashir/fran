# %%
from fastcore.basics import store_attr
from typing import Union
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
from typing import Union
from utilz.helpers import range_inclusive

tr = ipdb.set_trace
from monai.losses import DiceLoss
import torch.nn.functional as F

# Cell


class _DiceCELossMultiOutput(nn.Module):
    def __init__(
        self,
        include_background: bool = True,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction , str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        super().__init__()
        
        dice_reduction = LossReduction.NONE # unreduced loss is outputted for logging and will be reduced manually
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

        target_for_ce = target.clone()
        target_for_ce = target_for_ce.squeeze(1).long()
        dice_loss_unreduced = self.dice(input, target)
        dice_loss_reduced = torch.mean(dice_loss_unreduced)
        ce_loss = self.cross_entropy(input, target_for_ce)
        total_loss: torch.Tensor = (self.lambda_dice * dice_loss_reduced + self.lambda_ce * ce_loss )
        # return total_loss, ce_loss, dice_loss_reduced , dice_loss_unreduced.squeeze((2,3,4))
        losses_for_logging_only = [
            ce_loss.detach(),
            dice_loss_reduced.detach(),
            dice_loss_unreduced.squeeze(2).squeeze(2).squeeze(2).detach(),
        ]

        return total_loss, *losses_for_logging_only

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: [N, C, ...] logits
        # target: [N,1,...] or [N,...] with class indices (possibly >1)

        if len(input.shape) != len(target.shape) and not (target.ndim == input.ndim - 1):
            raise ValueError(f"input {tuple(input.shape)} vs target {tuple(target.shape)} mismatch")

        N, C, *sp = input.shape

        # --- canonicalize to index tensor [N, ...] ---
        if target.ndim == input.ndim and target.size(1) == 1:
            t_idx = target.select(1, 0)
        elif target.ndim == input.ndim - 1:
            t_idx = target
        else:
            raise ValueError("Unsupported target shape")

        t_idx = t_idx.long()

        # --- binary collapse for 2-class setup ---
        if C == 2:
            # everything >0 becomes foreground 1
            t_idx = (t_idx > 0).long()
        else:
            # guard against out-of-range
            if (t_idx < 0).any() or (t_idx >= C).any():
                u = t_idx.unique()
                raise ValueError(f"labels {u.tolist()} out of range for C={C}")

        # --- losses ---
        # DiceLoss has to_onehot_y=True, so pass indices
        t_dice = t_idx.unsqueeze(1) 
        dice_loss_unreduced = self.dice(input, t_dice)          # [N, C, ...] unreduced
        dice_loss_reduced   = dice_loss_unreduced.mean()

        ce_loss = self.cross_entropy(input, t_idx)              # CE expects indices

        total_loss = self.lambda_dice * dice_loss_reduced + self.lambda_ce * ce_loss

        losses_for_logging_only = [
            ce_loss.detach(),
            dice_loss_reduced.detach(),
            dice_loss_unreduced.squeeze(2).squeeze(2).squeeze(2).detach(),
        ]
        return total_loss, *losses_for_logging_only

class CombinedLoss(_DiceCELossMultiOutput):
    def __init__(self, fg_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fg_classes=fg_classes

    def forward(self, input, target):
        losses = super().forward(input, target)
        if not hasattr(self, "labels"):
            bs = target.shape[0]
            self.create_labels(bs, self.fg_classes)
        self.set_loss_dict(losses)
        return losses[0]
    #
    # def set_loss_dict(self, losses):
    #     losses[1:] = [
    #         ll.detach() for ll in losses[1:]
    #     ]  # only l[0] needs gradient. rest are for logging
    #     class_losses = losses[-1].mean(0)
    #     separate_case_losses = list(losses[-1].view(-1))
    #     self.loss_dict = {
    #         x: y.item()
    #         for x, y in zip(
    #             self.labels, il.chain(losses[:3], class_losses, separate_case_losses)
    #         )
    #     }


    def set_loss_dict(self, losses):
        class_losses = losses[-1].mean(0)
        separate_case_losses = list(losses[-1].view(-1))
        self.loss_dict = {
            x: y.item()
            for x, y in zip(
                self.labels, il.chain(losses[:3], class_losses, separate_case_losses)
            )
        }
    def create_labels(self, bs, fg_classes):
        label_maker = lambda x: "loss_dice_batch{0}_label{1}".format(*x)
        batches = list(range(bs))
        classes = range_inclusive(1, fg_classes)
        self.neptune_labels = ["loss", "loss_ce", "loss_dice"] + [
            "loss_dice_label{}".format(x) for x in classes
        ]

        self.case_recorder_labels = il.product(batches, classes)
        self.case_recorder_labels = map(label_maker, self.case_recorder_labels)
        self.labels = list(il.chain(self.neptune_labels, self.case_recorder_labels))


class DeepSupervisionLoss(pl.LightningModule):
    def __init__(self, levels: int, deep_supervision_scales, fg_classes: int):
        super().__init__()
        store_attr()
        assert fg_classes > 0, "fg_classes should be at least 1"
        sigmoid = False
        softmax = True
        include_background = True
        self.LossFunc = _DiceCELossMultiOutput(
            include_background=include_background, softmax=softmax, sigmoid=sigmoid
        )

    def create_labels(self, bs, fg_classes):
        label_maker = lambda x: "loss_dice_batch{0}_label{1}".format(*x)
        classes = range_inclusive(1, fg_classes)

        batches = list(range(bs))
        self.neptune_labels = ["loss", "loss_ce", "loss_dice"] + [
            "loss_dice_label{}".format(x) for x in classes
        ]

        self.case_recorder_labels = list(il.product(batches, classes))
        self.case_recorder_labels = map(label_maker, self.case_recorder_labels)
        self.labels = list(il.chain(self.neptune_labels, self.case_recorder_labels))

    def create_weights(self, device):
        weights = torch.tensor(
            [1 / (2**i) for i in range(self.levels)], requires_grad=False, device=device
        )

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        mask = torch.tensor(
            [True]
            + [True if i < self.levels - 1 else False for i in range(1, self.levels)]
        )
        weights[~mask] = 0
        self.weights = weights / weights.sum()

    def maybe_create_labels(self, target):
        if not hasattr(self, "labels"):
            if isinstance(target, list):
                bs = target[0].shape[0]
            else:
                bs = target.shape[0]
            self.create_labels(bs, self.fg_classes)

    def maybe_create_weights(self, preds):
        if not hasattr(self, "weights"):
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
    def apply_ds_scales(self, tnsr_inp: Union[list, torch.Tensor], mode):
        tnsr_listed = []
        for ind, sc in enumerate(self.deep_supervision_scales):
            if isinstance(tnsr_inp, Union[tuple, list]):
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

        if isinstance(preds, Union[list, tuple]) and isinstance(
            target, torch.Tensor
        ):  # multires lists in training
            target = self.apply_ds_scales(target, "nearest")
            losses = [self.LossFunc(xx, yy) for xx, yy in zip(preds, target)]

        elif isinstance(preds, torch.Tensor):  # tensor
            if (
                preds.dim() == target.dim() + 1
            ):  # i.e., training phase has deep supervision:
                preds = torch.unbind(preds, dim=1)
                preds = self.apply_ds_scales(preds, "trilinear")
                target = self.apply_ds_scales(target, "nearest")
                losses = [self.LossFunc(xx, yy) for xx, yy in zip(preds, target)]
            else:  # validation loss
                losses = [self.LossFunc(preds, target)]
        else:
            raise NotImplementedError("preds must be either list or stacked tensor")

        self.set_loss_dict(losses[0])
        losses_weighted = torch.stack(
            [self.weights * loss[0] for loss in losses]
        )  # total_loss * weight for each level
        losses_out = losses_weighted.sum()
        return losses_out

    def set_loss_dict(self, losses):
        class_losses = losses[-1].mean(0)
        separate_case_losses = list(losses[-1].view(-1))
        self.loss_dict = {
            x: y.item()
            for x, y in zip(
                self.labels, il.chain(losses[:3], class_losses, separate_case_losses)
            )
        }


# %%
if __name__ == "__main__":
    softmax_helper = lambda x: F.softmax(x, 1)
    P = Project("nodes")
    conf = ConfigMaker(P,  configuration_filename=None).config
    loss_params = conf["loss_params"]
    
    targ = torch.load("tests/files/image.pt", weights_only=False)
    pred = torch.load("tests/files/pred.pt",weights_only=False)
    target = torch.load("tests/files/target.pt",weights_only=False)

    fg_classes = 1
# %%
    loss_func = CombinedLoss(**loss_params, fg_classes =fg_classes)
    
    loss = loss_func.forward(pred, target)
# %%


# %%
