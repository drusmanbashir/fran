# %%
from pathlib import Path
from typing import Callable, Optional, TypeAlias

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import MaskedDiceLoss
from monai.losses.dice import DiceLoss
from monai.utils.enums import DiceCEReduction, LossReduction
from monai.utils.module import look_up_option
from utilz.helpers import info_from_filename

from fran.utils.common import PAD_VALUE

TensorSeq: TypeAlias = list[torch.Tensor] | tuple[torch.Tensor, ...]


def _detach_for_logging(value):
    if torch.is_tensor(value):
        return value.detach()
    return value


def _class_ids(include_background: bool, num_classes: int) -> list[int]:
    first_label = 0 if include_background else 1
    return list(range(first_label, first_label + num_classes))


def _meta_get(meta, key: str):
    if hasattr(meta, "get"):
        return meta.get(key)
    return None


def _normalize_filenames(meta, batch_size: int) -> list[Optional[str]]:
    filenames = _meta_get(meta, "filename_or_obj")
    if isinstance(filenames, (str, Path)):
        filenames = [str(filenames)]
    elif filenames is None:
        filenames = [None] * batch_size
    else:
        filenames = [None if fn is None else str(fn) for fn in filenames]

    if len(filenames) < batch_size:
        filenames = filenames + [None] * (batch_size - len(filenames))
    return filenames[:batch_size]


def _case_ids_from_filenames(filenames: list[Optional[str]]) -> list[Optional[str]]:
    case_ids = []
    for filename in filenames:
        if filename is None:
            case_ids.append(None)
            continue
        parsed = info_from_filename(Path(filename).name, full_caseid=True)
        case_ids.append(parsed["case_id"])
    return case_ids


def _build_loss_dict(losses: dict, include_background: bool) -> dict:
    per_case_class = losses["loss_dice_unreduced"]
    batch_size, num_classes = per_case_class.shape[:2]
    class_ids = _class_ids(include_background, num_classes)
    class_losses = per_case_class.mean(dim=0)
    filenames = _normalize_filenames(losses["meta"], batch_size)
    case_ids = _case_ids_from_filenames(filenames)
    loss_dict = {
        "loss": _detach_for_logging(losses["loss"]),
        "loss_ce": _detach_for_logging(losses["loss_ce"]),
        "loss_dice": _detach_for_logging(losses["loss_dice"]),
    }
    for class_id, loss in zip(class_ids, class_losses):
        loss_dict[f"loss_dice_label{class_id}"] = _detach_for_logging(loss)
    for batch_ind, case_losses in enumerate(per_case_class):
        loss_dict[f"batch{batch_ind}_filename"] = filenames[batch_ind]
        loss_dict[f"batch{batch_ind}_case_id"] = case_ids[batch_ind]
        for class_id, loss in zip(class_ids, case_losses):
            loss_dict[f"loss_dice_batch{batch_ind}_label{class_id}"] = (
                _detach_for_logging(loss)
            )
    return loss_dict


class _DiceCELossMultiOutput(nn.Module):
    def __init__(
        self,
        include_background: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 0.5,
        lambda_ce: float = 0.5,
    ) -> None:
        super().__init__()
        self.include_background = bool(include_background)
        self.target_label_sanitizer = None

        dice_reduction = LossReduction.NONE
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

    def set_target_label_sanitizer(self, sanitizer: Callable) -> None:
        self.target_label_sanitizer = sanitizer

    def _sanitize_target(self, target: torch.Tensor) -> torch.Tensor:
        if self.target_label_sanitizer is None:
            return target
        target_sanitized = self.target_label_sanitizer(target)
        pad_mask = target == PAD_VALUE
        if pad_mask.any():
            target_sanitized = target_sanitized.clone()
            target_sanitized[pad_mask] = PAD_VALUE
        return target_sanitized

    def compute_ce_loss(
        self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        target_idx = target.select(1, 0).long()
        if mask is None:
            return self.cross_entropy(input, target_idx)

        ce_unreduced = F.cross_entropy(
            input,
            target_idx,
            weight=self.cross_entropy.weight,
            reduction="none",
            ignore_index=PAD_VALUE,
        )
        valid_mask = mask.select(1, 0).to(dtype=ce_unreduced.dtype)
        valid_count = valid_mask.sum().clamp_min(1.0)
        return (ce_unreduced * valid_mask).sum() / valid_count

    def compute_dice_loss(
        self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is not None:
            loss_dice_unreduced = self.dice_m(input, target, mask=mask)
        else:
            loss_dice_unreduced = self.dice(input, target)
        loss_dice_unreduced = loss_dice_unreduced.flatten(start_dim=2).mean(-1)
        return loss_dice_unreduced

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, use_mask: bool = False
    ) -> dict:
        if input.ndim != target.ndim:
            raise ValueError(
                f"input {tuple(input.shape)} vs target {tuple(target.shape)} mismatch"
            )

        meta = getattr(target, "meta", {})
        mask = target != PAD_VALUE if use_mask else None
        target = self._sanitize_target(target)

        loss_dice_unreduced = self.compute_dice_loss(input, target, mask)
        loss_dice = loss_dice_unreduced.mean()
        loss_ce = self.compute_ce_loss(input, target, mask)
        total_loss = self.lambda_dice * loss_dice + self.lambda_ce * loss_ce

        return {
            "loss": total_loss,
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
            "loss_dice_unreduced": loss_dice_unreduced.detach(),
            "meta": meta,
        }


class CombinedLoss(_DiceCELossMultiOutput):
    def __init__(self, fg_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fg_classes = fg_classes

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, use_mask: bool = False
    ) -> torch.Tensor:
        losses = super().forward(input, target, use_mask=use_mask)
        self.loss_dict = _build_loss_dict(losses, self.include_background)
        return losses["loss"]


class DeepSupervisionLoss(pl.LightningModule):
    def __init__(
        self,
        levels: int,
        deep_supervision_scales,
        fg_classes: int,
        include_background: bool = False,
    ):
        super().__init__()
        assert fg_classes > 0, "fg_classes should be at least 1"

        self.include_background = bool(include_background)
        self.levels = levels
        self.fg_classes = fg_classes
        self.deep_supervision_scales = [list(scale) for scale in deep_supervision_scales]
        self.LossFunc = _DiceCELossMultiOutput(
            include_background=include_background,
            softmax=True,
            sigmoid=False,
        )

    def set_target_label_sanitizer(self, sanitizer: Callable) -> None:
        self.LossFunc.set_target_label_sanitizer(sanitizer)

    def create_labels(self, bs, fg_classes):
        class_start = 0 if self.include_background else 1
        classes = list(range(class_start, class_start + fg_classes))
        self.loss_labels = ["loss", "loss_ce", "loss_dice"] + [
            "loss_dice_label{}".format(x) for x in classes
        ]

    def create_weights(self, device):
        weights = torch.tensor(
            [1 / (2**i) for i in range(self.levels)], requires_grad=False, device=device
        )
        mask = torch.tensor(
            [True]
            + [True if i < self.levels - 1 else False for i in range(1, self.levels)]
        )
        weights[~mask] = 0
        self.weights = weights / weights.sum()

    def maybe_create_labels(self, target: torch.Tensor | TensorSeq) -> None:
        if not hasattr(self, "loss_labels"):
            if isinstance(target, (list, tuple)):
                bs = target[0].shape[0]
            else:
                bs = target.shape[0]
            self.create_labels(bs, self.fg_classes)

    def maybe_create_weights(self, preds: torch.Tensor | TensorSeq) -> None:
        if not hasattr(self, "weights"):
            if isinstance(preds, (list, tuple)):
                self.create_weights(preds[0].device)
            else:
                self.create_weights(preds.device)

    def apply_ds_scales(
        self,
        tnsr_inp: torch.Tensor | TensorSeq,
        mode: str,
    ) -> list[torch.Tensor]:
        tnsr_listed: list[torch.Tensor] = []
        for ind, sc in enumerate(self.deep_supervision_scales):
            if isinstance(tnsr_inp, (list, tuple)):
                tnsr = tnsr_inp[ind]
            else:
                tnsr = tnsr_inp
            if all(i == 1 for i in sc):
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

    def forward(
        self,
        preds: torch.Tensor | TensorSeq,
        target: torch.Tensor,
        use_mask: bool = False,
    ) -> torch.Tensor:
        self.maybe_create_weights(preds)
        self.maybe_create_labels(target)

        scaled_target = self.apply_ds_scales(target, "nearest")
        if isinstance(preds, (list, tuple)):
            losses = [
                self.LossFunc(xx, yy, use_mask=use_mask)
                for xx, yy in zip(preds, scaled_target)
            ]
        elif isinstance(preds, torch.Tensor):
            if preds.dim() == scaled_target[0].dim() + 1:
                pred_levels: tuple[torch.Tensor, ...] = torch.unbind(preds, dim=1)
                scaled_preds = self.apply_ds_scales(pred_levels, "trilinear")
                losses = [
                    self.LossFunc(xx, yy, use_mask=use_mask)
                    for xx, yy in zip(scaled_preds, scaled_target)
                ]
            else:
                losses = [self.LossFunc(preds, target, use_mask=use_mask)]
        else:
            raise NotImplementedError("preds must be either list or stacked tensor")

        self.set_loss_dict(losses[0])
        weights = self.weights[: len(losses)]
        losses_weighted = torch.stack(
            [w * loss["loss"] for w, loss in zip(weights, losses)]
        )
        return losses_weighted.sum()

    def set_loss_dict(self, losses: dict):
        per_case_class = losses["loss_dice_unreduced"]
        bs, num_classes = per_case_class.shape[:2]
        class_ids = _class_ids(self.include_background, num_classes)
        class_losses = per_case_class.mean(0)
        meta = losses["meta"]
        filenames = _normalize_filenames(meta, bs)
        case_ids = _case_ids_from_filenames(filenames)

        self.loss_dict = {
            "loss": _detach_for_logging(losses["loss"]),
            "loss_ce": _detach_for_logging(losses["loss_ce"]),
            "loss_dice": _detach_for_logging(losses["loss_dice"]),
        }
        for class_id, loss in zip(class_ids, class_losses):
            self.loss_dict[f"loss_dice_label{class_id}"] = _detach_for_logging(loss)
        for batch_ind in range(bs):
            self.loss_dict[f"batch{batch_ind}_filename"] = filenames[batch_ind]
            self.loss_dict[f"batch{batch_ind}_case_id"] = case_ids[batch_ind]
            for class_ind, class_id in enumerate(class_ids):
                val = per_case_class[batch_ind][class_ind]
                self.loss_dict[f"loss_dice_batch{batch_ind}_label{class_id}"] = (
                    _detach_for_logging(val)
                )


# %%
if __name__ == "__main__":
    from nnunet.utilities.nd_softmax import softmax_helper

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
