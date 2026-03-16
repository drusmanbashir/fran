# %%
import itertools as il
from typing import Callable, Optional

import ipdb
import lightning.pytorch as pl
from monai.losses.dice import DiceLoss
import numpy as np
import torch
import torch.nn as nn
from fastcore.basics import store_attr
from monai.utils.enums import DiceCEReduction, LossReduction
from monai.utils.module import look_up_option
from nnunet.utilities.nd_softmax import softmax_helper
from utilz.helpers import info_from_filename, range_inclusive

tr = ipdb.set_trace
import torch.nn.functional as F
from monai.losses import MaskedDiceLoss

from fran.utils.common import PAD_VALUE

# Cell


class _DiceCELossMultiOutput(nn.Module):
    def __init__(
        self,
        include_background: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction| str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        super().__init__()
        self.include_background = bool(include_background)

        dice_reduction = (
            LossReduction.NONE
        )  # unreduced loss is outputted for logging and will be reduced manually
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.dice_m = MaskedDiceLoss(
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
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction="mean")
            
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        # self.reduction=reduction


    def compute_ce_loss(self,input,target,mask):
        t_idx = target.select(1, 0).long()
        if mask is None:
            return self.cross_entropy(input, t_idx)

        ce_unreduced = F.cross_entropy(
            input,
            t_idx,
            weight=self.cross_entropy.weight,
            reduction="none",
            ignore_index=PAD_VALUE,
        )
        valid_mask = mask.select(1, 0).to(dtype=ce_unreduced.dtype)
        valid_count = valid_mask.sum().clamp_min(1.0)
        return (ce_unreduced * valid_mask).sum() / valid_count


    def compute_dice_loss(self,input,target,mask):
        if mask is not None:
            loss_dice_unreduced = self.dice_m(input, target, mask=mask)
        else:
            loss_dice_unreduced = self.dice(input, target)
        loss_dice_unreduced = loss_dice_unreduced.flatten(start_dim=2).mean(-1)
        return loss_dice_unreduced




    def forward(
        self, input: torch.Tensor, target: torch.Tensor, use_mask=False
    ) -> dict:
        # input: [N, C, ...] logits
        # target: [N,1,...] , a channel dim is MUST

        if not target.ndim == input.ndim : 
            raise ValueError(
                f"input {tuple(input.shape)} vs target {tuple(target.shape)} mismatch"
            )

        if use_mask==True:
            mask = target != PAD_VALUE
        else:
            mask=None

        loss_dice_unreduced = self.compute_dice_loss(input,target, mask)
        loss_dice_reduced = loss_dice_unreduced.mean()
        loss_ce = self.compute_ce_loss(input, target, mask)

        total_loss = self.lambda_dice * loss_dice_reduced + self.lambda_ce * loss_ce
        meta = getattr(target, "meta", {})

        output_dici = {
            "loss": total_loss,
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice_reduced.detach(),
            "loss_dice_unreduced": loss_dice_unreduced.detach(),
            "meta": meta,
        }
        return output_dici


class CombinedLoss(_DiceCELossMultiOutput):
    def __init__(self, fg_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fg_classes = fg_classes

    def forward(self, input, target, use_mask=False):
        losses = super().forward(input, target, use_mask=use_mask)
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
        first_label = 0 if self.include_background else 1
        class_ids = list(range(first_label, first_label + num_classes))
        case_labels = [
            f"loss_dice_batch{b}_label{c}" for b, c in il.product(range(bs), class_ids)
        ]
        keys = (
            ["loss", "loss_ce", "loss_dice"]
            + [f"loss_dice_label{i}" for i in class_ids]
            + case_labels
        )
        vals = list(
            il.chain(
                [losses["loss"], losses["loss_ce"], losses["loss_dice"]],
                class_losses,
                per_case_class.reshape(-1),
            )
        )
        self.loss_dict = {
            k: (v.item() if torch.is_tensor(v) else v) for k, v in zip(keys, vals)
        }

    def create_labels(self, bs, fg_classes):
        label_maker = lambda x: "loss_dice_batch{0}_label{1}".format(*x)
        batches = list(range(bs))
        class_start = 0 if self.include_background else 1
        classes = range_inclusive(class_start, fg_classes)
        self.class_labels = ["loss_dice_label{}".format(x) for x in classes]

        self.case_recorder_labels = il.product(batches, classes)
        self.case_recorder_labels = map(label_maker, self.case_recorder_labels)
        self.labels = list(il.chain(self.class_labels, self.case_recorder_labels))


class DeepSupervisionLoss(pl.LightningModule):
    def __init__(
        self,
        levels: int,
        deep_supervision_scales,
        fg_classes: int,
        include_background=False,
    ):
        super().__init__()
        store_attr()
        assert fg_classes > 0, "fg_classes should be at least 1"
        sigmoid = False
        softmax = True

        self.fg_classes = fg_classes
        self.LossFunc = _DiceCELossMultiOutput(
            include_background=include_background, softmax=softmax, sigmoid=sigmoid
        )

    def create_labels(self, bs, fg_classes):
        label_maker = lambda x: "loss_dice_batch{0}_label{1}".format(*x)
        filename_maker = lambda x: "batch{0}_filename".format(x)
        caseid_maker = lambda x: "batch{0}_case_id".format(x)
        class_start = 0 if self.include_background else 1
        classes = range_inclusive(class_start, fg_classes)

        batches = list(range(bs))
        self.loss_labels = ["loss", "loss_ce", "loss_dice"] + [
            "loss_dice_label{}".format(x) for x in classes
        ]

        self.case_recorder_labels = list(il.product(batches, classes))
        self.case_recorder_labels = map(label_maker, self.case_recorder_labels)
        self.filename_labels = map(filename_maker, batches)
        self.caseid_labels = map(caseid_maker, batches)

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

    def apply_ds_scales(self, tnsr_inp:  torch.Tensor, mode):
        tnsr_listed = []
        for ind, sc in enumerate(self.deep_supervision_scales):
            if isinstance(tnsr_inp, (tuple, list)):
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

    def forward(self, preds :torch.Tensor|list|tuple, target: torch.Tensor, use_mask=False):

        self.maybe_create_weights(preds)
        self.maybe_create_labels(target)

        target = self.apply_ds_scales(target, "nearest")
        if isinstance(preds, (list, tuple)):  # multires lists in training
            if isinstance(target, torch.Tensor):
            losses = [
                    self.LossFunc(xx, yy, use_mask=use_mask)
                    for xx, yy in zip(preds, target)
                ]

        elif isinstance(preds, torch.Tensor):  # tensor
            if (
                preds.dim() == target.dim() + 1
            ):  # i.e., training phase has deep supervision:
                preds = torch.unbind(preds, dim=1)
                preds = self.apply_ds_scales(preds, "trilinear")
                target = self.apply_ds_scales(target, "nearest")
                losses = [
                    self.LossFunc(xx, yy, use_mask=use_mask)
                    for xx, yy in zip(preds, target)
                ]
            else:  # validation loss
                losses = [self.LossFunc(preds, target, use_mask=use_mask)]
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

    def set_loss_dict(self, losses: dict):
        # 0. Total loss
        # 1. CE loss
        # 2  loss_dice_reduced
        # 3 loss_dice_unreduced, ie.e., NXC if foreground on ly , then its Nx1, if 2 fg_classes, Nx3 and so on by including background
        per_case_class = losses["loss_dice_unreduced"]  # [N, C]
        bs, num_classes = per_case_class.shape[:2]
        first_label = 0 if self.include_background else 1
        class_ids = list(range(first_label, first_label + num_classes))
        class_losses = per_case_class.mean(
            0
        )  # this collapses the batches , now only per class dice losses remain
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
            "loss": (
                losses["loss"].item()
                if torch.is_tensor(losses["loss"])
                else losses["loss"]
            ),
            "loss_ce": (
                losses["loss_ce"].item()
                if torch.is_tensor(losses["loss_ce"])
                else losses["loss_ce"]
            ),
            "loss_dice": (
                losses["loss_dice"].item()
                if torch.is_tensor(losses["loss_dice"])
                else losses["loss_dice"]
            ),
        }
        for class_id, loss in zip(class_ids, class_losses):
            self.loss_dict[f"loss_dice_label{class_id}"] = (
                loss.item() if torch.is_tensor(loss) else loss
            )
        for batch_ind in range(bs):
            self.loss_dict[f"batch{batch_ind}_filename"] = filenames[batch_ind]
            self.loss_dict[f"batch{batch_ind}_case_id"] = case_ids[batch_ind]
            for class_ind, class_id in enumerate(class_ids):
                val = per_case_class[batch_ind][class_ind]
                self.loss_dict[f"loss_dice_batch{batch_ind}_label{class_id}"] = (
                    val.item() if torch.is_tensor(val) else val
                )


# %%
if __name__ == "__main__":
    softmax_helper = lambda x: F.softmax(x, 1)
    P = Project("nodes")
    conf = ConfigMaker(P, configuration_filename=None).config
    loss_params = conf["loss_params"]

    targ = torch.load("tests/files/image.pt", weights_only=False)
    pred = torch.load("tests/files/pred.pt", weights_only=False)
    target = torch.load("tests/files/target.pt", weights_only=False)

    fg_classes = 1
    # %%
    loss_func = CombinedLoss(**loss_params, fg_classes=fg_classes)

    loss = loss_func.forward(pred, target)
# %%


# %%
