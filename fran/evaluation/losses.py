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
from utilz.helpers import info_from_filename, range_inclusive

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

    def forward(self, input: torch.Tensor, target: torch.Tensor) ->dict:
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

        # --- FOR  a 2 class setup, we keep only the foreground and compute its dice
        if C == 2: 
            # convert target to a binary mask, everything >0 becomes foreground 1 , i.e., labels 2 ,3 ,4 -> 1
            t_idx = (t_idx > 0).long()
        else:
            # guard against out-of-range
            if (t_idx < 0).any() or (t_idx >= C).any():
                u = t_idx.unique()
                raise ValueError(f"labels {u.tolist()} out of range for C={C}")

        # --- losses ---
        # DiceLoss has to_onehot_y=True, so pass indices
        t_dice = t_idx.unsqueeze(1) 
        loss_dice_unreduced = self.dice(input, t_dice)          # [N, C, ...] unreduced
        if loss_dice_unreduced.ndim > 2:
            # reduce spatial dims only, keep [N, C] for per-case/per-class logging
            loss_dice_unreduced = loss_dice_unreduced.flatten(start_dim=2).mean(-1)
        loss_dice_reduced   = loss_dice_unreduced.mean()

        loss_ce = self.cross_entropy(input, t_idx)              # CE expects indices

        total_loss = self.lambda_dice * loss_dice_reduced + self.lambda_ce * loss_ce
        meta = getattr(target, "meta", {})

        output_dici= {
                "loss": total_loss,
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice_reduced.detach(),
            "loss_dice_unreduced": loss_dice_unreduced.detach(),
                "meta": meta
        }
        return output_dici

class CombinedLoss(_DiceCELossMultiOutput):
    def __init__(self, fg_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fg_classes=fg_classes

    def forward(self, input, target):
        losses = super().forward(input, target)
        self.set_loss_dict(losses)
        return losses["loss"]
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
        per_case_class = losses["loss_dice_unreduced"]  # [N, C]
        bs, num_classes = per_case_class.shape[:2]
        class_losses = per_case_class.mean(0)
        case_labels = [
            f"loss_dice_batch{b}_label{c}" for b, c in il.product(range(bs), range(num_classes))
        ]
        keys = ["loss", "loss_ce", "loss_dice"] + [f"loss_dice_label{i}" for i in range(num_classes)] + case_labels
        vals = list(
            il.chain(
                [losses["loss"], losses["loss_ce"], losses["loss_dice"]],
                class_losses,
                per_case_class.reshape(-1),
            )
        )
        self.loss_dict = {k: (v.item() if torch.is_tensor(v) else v) for k, v in zip(keys, vals)}
    def create_labels(self, bs, fg_classes):
        label_maker = lambda x: "loss_dice_batch{0}_label{1}".format(*x)
        batches = list(range(bs))
        # For binary segmentation we exclude background from Dice logging.
        # For multiclass we include background (label 0) to match Dice output channels.
        classes = range_inclusive(0, fg_classes)
        self.class_labels =  [
            "loss_dice_label{}".format(x) for x in classes
        ]

        self.case_recorder_labels = il.product(batches, classes)
        self.case_recorder_labels = map(label_maker, self.case_recorder_labels)
        self.labels = list(il.chain(self.class_labels, self.case_recorder_labels))


class DeepSupervisionLoss(pl.LightningModule):
    def __init__(self, levels: int, deep_supervision_scales, fg_classes: int):
        super().__init__()
        store_attr()
        assert fg_classes > 0, "fg_classes should be at least 1"
        sigmoid = False
        softmax = True
        include_background = True
        self.fg_classes = fg_classes
        self.LossFunc = _DiceCELossMultiOutput(
            include_background=include_background, softmax=softmax, sigmoid=sigmoid
        )

    def create_labels(self, bs, fg_classes):
        label_maker = lambda x: "loss_dice_batch{0}_label{1}".format(*x)
        filename_maker = lambda x: "batch{0}_filename".format(x)
        caseid_maker = lambda x: "batch{0}_case_id".format(x)
        # For binary segmentation we exclude background from Dice logging.
        # For multiclass we include background (label 0) to match Dice output channels.
        classes = range_inclusive(0, fg_classes)

        batches = list(range(bs))
        self.loss_labels = ["loss", "loss_ce", "loss_dice"] + [
            "loss_dice_label{}".format(x) for x in classes
        ]

        self.case_recorder_labels = list(il.product(batches, classes))
        self.case_recorder_labels = map(label_maker, self.case_recorder_labels)
        self.filename_labels = map(filename_maker, batches)
        self.caseid_labels  =map(caseid_maker, batches)

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
                was_fp = tnsr.is_floating_point()
                interp_inp = tnsr if was_fp else tnsr.float()
                tnsr_downsampled = F.interpolate(interp_inp, size=size, mode=mode)
                if mode == "nearest" and not was_fp:
                    tnsr_downsampled = tnsr_downsampled.to(tnsr.dtype)
                tnsr_listed.append(tnsr_downsampled)
        return tnsr_listed

    def forward(self, preds, target):

        self.maybe_create_weights(preds)
        self.maybe_create_labels(target)

        if isinstance(preds, (list, tuple)):  # multires lists in training
            if isinstance(target, torch.Tensor):
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

        self.set_loss_dict(losses[0])  # 
        weights = self.weights[: len(losses)]
        losses_weighted = torch.stack(
            [w * loss["loss"] for w, loss in zip(weights, losses)]
        )  # total_loss * weight for each level
        losses_out = losses_weighted.sum()
        return losses_out

            # ce_loss.detach(),
            # loss_dice_reduced.detach(),
            # loss_dice_unreduced.squeeze(2).squeeze(2).squeeze(2).detach(),
    def set_loss_dict(self, losses:dict):
        #0. Total loss
        #1. CE loss
        #2  loss_dice_reduced
        #3 loss_dice_unreduced, ie.e., NXC if foreground on ly , then its Nx1, if 2 fg_classes, Nx3 and so on by including background 
        per_case_class = losses["loss_dice_unreduced"]  # [N, C]
        bs, num_classes = per_case_class.shape[:2]
        class_losses = per_case_class.mean(0) # this collapses the batches , now only per class dice losses remain
        meta = losses["meta"]
        filenames = meta.get("filename_or_obj")
        if isinstance(filenames, str):
            filenames = [filenames]
        case_ids = []
        for fn in filenames:
                if fn is None:
                    case_ids.append(None)
                    continue
                fn_name = fn.split("/")[-1]
                parsed = info_from_filename(fn_name)
                case_ids.append(parsed["case_id"])

        self.loss_dict = {
            "loss": losses["loss"].item() if torch.is_tensor(losses["loss"]) else losses["loss"],
            "loss_ce": losses["loss_ce"].item() if torch.is_tensor(losses["loss_ce"]) else losses["loss_ce"],
            "loss_dice": losses["loss_dice"].item() if torch.is_tensor(losses["loss_dice"]) else losses["loss_dice"],
        }
        for i, loss in enumerate(class_losses):
            self.loss_dict[f"loss_dice_label{i}"] = loss.item() if torch.is_tensor(loss) else loss
        for batch_ind in range(bs):
            self.loss_dict[f"batch{batch_ind}_filename"] = filenames[batch_ind]
            self.loss_dict[f"batch{batch_ind}_caseid"] = case_ids[batch_ind]
            for class_ind in range(num_classes):
                val = per_case_class[batch_ind][class_ind]
                self.loss_dict[f"loss_dice_batch{batch_ind}_label{class_ind}"] = val.item() if torch.is_tensor(val) else val


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
