from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses.dice import compute_tp_fp_fn

from fran.evaluation.losses import _class_ids, _detach_for_logging
from fran.utils.common import PAD_VALUE


def _dice_source_from_loss(loss_module: nn.Module):
    if hasattr(loss_module, "dice"):
        return loss_module.dice
    if hasattr(loss_module, "LossFunc") and hasattr(loss_module.LossFunc, "dice"):
        return loss_module.LossFunc.dice
    raise AttributeError("Could not locate DiceLoss instance on validation loss module.")


def _extract_primary_logits(preds: Any, target: torch.Tensor) -> torch.Tensor:
    if isinstance(preds, (list, tuple)):
        return preds[0]
    if isinstance(preds, torch.Tensor) and preds.dim() == target.dim() + 2:
        return preds[:, 0]
    if isinstance(preds, torch.Tensor):
        return preds
    raise NotImplementedError("preds must be tensor, list, or tuple")


class PatchStreamValidationLoss(nn.Module):
    """
    Validation-only adapter.

    - Keeps existing patch-level loss/logging from `base_loss`.
    - Accumulates exact soft-Dice numerators/denominators per case across patches.
    - Emits case-level Dice loss logs only when a case completes.
    """

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss
        self.dice_source = _dice_source_from_loss(base_loss)
        self.include_background = bool(self.dice_source.include_background)
        self.smooth_nr = float(self.dice_source.smooth_nr)
        self.smooth_dr = float(self.dice_source.smooth_dr)
        self.loss_dict: dict[str, Any] = {}
        self.completed_cases: list[dict[str, Any]] = []
        self._case_state: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        self.loss_dict = {}
        self.completed_cases = []
        self._case_state = {}

    def _prepare_for_dice(
        self, logits: torch.Tensor, target: torch.Tensor, use_mask: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred = logits.detach()
        if self.dice_source.sigmoid:
            pred = torch.sigmoid(pred)
        if self.dice_source.softmax:
            pred = torch.softmax(pred, dim=1)
        if self.dice_source.other_act is not None:
            pred = self.dice_source.other_act(pred)

        target_idx = target.select(1, 0).detach().long()
        valid_mask = target_idx != PAD_VALUE if use_mask else torch.ones_like(target_idx, dtype=torch.bool)
        if (~valid_mask).any():
            target_idx = target_idx.clone()
            target_idx[~valid_mask] = 0

        num_classes = pred.shape[1]
        target_1h = F.one_hot(target_idx, num_classes=num_classes).movedim(-1, 1).to(pred.dtype)
        if use_mask:
            valid_mask_ch = valid_mask.unsqueeze(1).to(pred.dtype)
            pred = pred * valid_mask_ch
            target_1h = target_1h * valid_mask_ch

        if not self.include_background and num_classes > 1:
            pred = pred[:, 1:]
            target_1h = target_1h[:, 1:]
        return pred, target_1h

    def _accumulate_case_stats(
        self, preds: Any, target: torch.Tensor, batch: dict[str, Any], use_mask: bool
    ) -> list[dict[str, Any]]:
        logits = _extract_primary_logits(preds, target)
        pred, target_1h = self._prepare_for_dice(logits, target, use_mask=use_mask)
        reduce_axis = list(range(1, pred.ndim))
        if len(reduce_axis) > 1:
            reduce_axis = reduce_axis[1:]
        ord_val = 2 if self.dice_source.squared_pred else 1

        completed = []
        class_labels = _class_ids(self.include_background, pred.shape[1])
        case_ids = list(batch["case_id"])
        patch_index = list(batch["patch_index"])
        patches_in_case = list(batch["patches_in_case"])

        for i, case_id in enumerate(case_ids):
            tp, fp, fn = compute_tp_fp_fn(
                pred[i : i + 1],
                target_1h[i : i + 1],
                reduce_axis=reduce_axis,
                ord=ord_val,
                soft_label=self.dice_source.soft_label,
            )
            tp = tp.squeeze(0).to(torch.float64).cpu()
            fp = fp.squeeze(0).to(torch.float64).cpu()
            fn = fn.squeeze(0).to(torch.float64).cpu()

            state = self._case_state.setdefault(
                case_id,
                {
                    "tp": torch.zeros_like(tp),
                    "fp": torch.zeros_like(fp),
                    "fn": torch.zeros_like(fn),
                    "class_labels": class_labels,
                },
            )
            state["tp"] += tp
            state["fp"] += fp
            state["fn"] += fn

            if int(patch_index[i]) == int(patches_in_case[i]) - 1:
                completed.append(self._finalize_case(case_id))
        return completed

    def _finalize_case(self, case_id: str) -> dict[str, Any]:
        state = self._case_state.pop(case_id)
        tp = state["tp"]
        fp = state["fp"]
        fn = state["fn"]
        if not self.dice_source.jaccard:
            fp = fp * 0.5
            fn = fn * 0.5
        numerator = 2 * tp + self.smooth_nr
        denominator = 2 * (tp + fp + fn) + self.smooth_dr
        class_losses = 1.0 - (numerator / denominator)
        class_labels = state["class_labels"]

        out = {
            "case_id": case_id,
            "loss_dice_case": float(class_losses.mean().item()),
        }
        for class_label, loss in zip(class_labels, class_losses):
            out[f"loss_dice_case_label{class_label}"] = float(loss.item())
        return out

    def _build_case_log_dict(self, completed: list[dict[str, Any]]) -> dict[str, Any]:
        if not completed:
            return {}
        out: dict[str, Any] = {
            "num_completed_cases": torch.tensor(float(len(completed)), dtype=torch.float32)
        }
        loss_dice_case = [row["loss_dice_case"] for row in completed]
        out["loss_dice_case"] = torch.tensor(
            sum(loss_dice_case) / len(loss_dice_case), dtype=torch.float32
        )

        label_keys = sorted(k for k in completed[0].keys() if k.startswith("loss_dice_case_label"))
        for key in label_keys:
            vals = [row[key] for row in completed]
            out[key] = torch.tensor(sum(vals) / len(vals), dtype=torch.float32)
        return {k: _detach_for_logging(v) for k, v in out.items()}

    def forward(
        self, preds: Any, target: torch.Tensor, batch: dict[str, Any], use_mask: bool = True
    ) -> torch.Tensor:
        patch_loss = self.base_loss(preds, target, use_mask=use_mask)
        patch_logs = dict(self.base_loss.loss_dict)
        completed = self._accumulate_case_stats(preds, target, batch, use_mask=use_mask)
        self.completed_cases.extend(completed)
        case_logs = self._build_case_log_dict(completed)
        self.loss_dict = {**patch_logs, **case_logs}
        return patch_loss
