import os
import re
from pathlib import Path

import ipdb
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT

from fran.managers.project import Project

tr = ipdb.set_trace

from typing import Any, Union

import numpy as np
import torch._dynamo
from fastcore.basics import store_attr
from ipdb.__main__ import get_ipython
from utilz.stringz import info_from_filename

from fran.evaluation.losses import CombinedLoss, DeepSupervisionLoss

torch._dynamo.config.suppress_errors = True
import itertools as il
import operator

import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fran.architectures.create_network import (create_model_from_conf,
                                               pool_op_kernels_nnunet)

try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass



class UNetManager(LightningModule):
    def __init__(
        self,
        project_title,
        configs,
        lr=None,
        sync_dist=False,
    ):
        super().__init__()

        self.sync_dist = sync_dist
        self.project = Project(project_title)
        self.save_hyperparameters("project_title", "configs", "lr")
        self.plan = configs["plan_train"]
        self.model_params = configs["model_params"]
        self.loss_params = configs["loss_params"]
        self.lr = lr if lr else self.model_params["lr"]
        self.model = self.create_model()

    # def on_fit_start(self):
    #     self.create_loss_fnc()
    #     super().on_fit_start()
    #
    def setup(self, stage="fit"):
        self.create_loss_fnc()
        super().setup(stage="fit")
    def _common_step(self, batch, batch_idx):
        if not hasattr(self, "batch_size"):
            self.batch_size = batch["image"].shape[0]
        inputs, target = batch["image"], batch["lm"]
        pred = self.forward(
            inputs
        )  # self.pred so that NeptuneImageGridCallback can use it

        pred, target = self.maybe_apply_ds_scales(pred, target)
        loss = self.loss_fnc(pred, target)
        loss_dict = self.loss_fnc.loss_dict
        self.maybe_store_preds(pred)
        return loss, loss_dict

    def maybe_store_preds(self, pred):
        if hasattr(self.trainer, "store_preds") and self.trainer.store_preds == True:
            if isinstance(pred, Union[tuple, list]):
                self.pred = [p.detach().cpu() for p in pred]
            else:
                self.pred = pred.detach().cpu()

    def maybe_apply_ds_scales(self, pred, target):
        if isinstance(pred, list) and isinstance(target, torch.Tensor):
            target_listed = []
            for s in self.deep_supervision_scales:
                if all([i == 1 for i in s]):
                    target_listed.append(target)
                else:
                    size = [
                        int(np.round(ss * aa)) for ss, aa in zip(s, target.shape[2:])
                    ]
                    target_downsampled = F.interpolate(
                        target, size=size, mode="nearest"
                    )
                    target_listed.append(target_downsampled)
            target = target_listed
        return pred, target

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self._common_step(batch, batch_idx)
        self.log_losses(loss_dict, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, loss_dict = self._common_step(batch, batch_idx)
        self.log_losses(loss_dict, prefix="val{}".format(dataloader_idx))
        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_dict = self._common_step(batch, batch_idx)
        self.log_losses(loss_dict, prefix="test")
        return loss

    def log_losses(self, loss_dict, prefix):
        self.loss_dict_full = loss_dict
        metrics = loss_dict.keys()
        metrics = [
            metric for metric in metrics if "batch" not in metric
        ]  # too detailed otherwise
        renamed = [prefix + "_" + nm for nm in metrics]
        logger_dict = {
            neo_key: loss_dict[key] for neo_key, key in zip(renamed, metrics)
        }
        self.log_dict(
            logger_dict,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=self.sync_dist,
            add_dataloader_idx=False,
        )
        # self.log(prefix + "_" + "loss_dice", loss_dict["loss_dice"], logger=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=30)
        output = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_dice",
                "frequency": 2,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        return output

    def forward(self, inputs):
        return self.model(inputs)

    def _case_ids_from_batch(self, batch) -> list[str]:
        image = batch.get("image")
        if image is None or not hasattr(image, "meta"):
            return []
        meta = image.meta
        fns = meta.get("filename_or_obj", None) or meta.get("src_filename", None)
        if fns is None:
            return []
        if isinstance(fns, (str, Path)):
            fns = [fns]
        out = []
        for fn in fns:
            name = Path(str(fn)).name
            try:
                cid = info_from_filename(name, full_caseid=True)["case_id"]
            except Exception:
                cid = Path(name).stem
            out.append(cid)
        return out

    def _per_case_dice_from_loss_dict(self, loss_dict: dict) -> dict[int, float]:
        grouped = {}
        for key, val in loss_dict.items():
            matched = re.match(r"loss_dice_batch(\d+)_label(\d+)", str(key))
            if not matched:
                continue
            batch_idx = int(matched.group(1))
            dice = max(0.0, min(1.0, 1.0 - float(val)))
            grouped.setdefault(batch_idx, []).append(dice)
        return {
            idx: float(sum(vals) / max(1, len(vals))) for idx, vals in grouped.items()
        }

    def _predict_metrics_from_logits(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=1)
        eps = 1e-8
        entropy = -(probs * (probs + eps).log()).sum(dim=1)
        entropy_case = entropy.flatten(1).mean(dim=1)
        c = max(2, probs.shape[1])
        entropy_case = entropy_case / float(
            torch.log(torch.tensor(float(c), device=logits.device))
        )
        flat = probs.flatten(2)
        mean_c = flat.mean(dim=2)
        std_c = flat.std(dim=2)
        emb = torch.cat([mean_c, std_c], dim=1)
        return entropy_case, emb

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, target = batch["image"], batch["lm"]
        pred = self.forward(inputs)
        logits = pred[0] if isinstance(pred, (list, tuple)) else pred
        if isinstance(logits, torch.Tensor) and logits.dim() == target.dim() + 2:
            logits = logits[:, 0]
        pred_loss, target_loss = self.maybe_apply_ds_scales(pred, target)
        _ = self.loss_fnc(pred_loss, target_loss)
        dice_by_idx = self._per_case_dice_from_loss_dict(self.loss_fnc.loss_dict)
        entropy_case, emb = self._predict_metrics_from_logits(logits)
        case_ids = self._case_ids_from_batch(batch)
        rows = []
        for i, cid in enumerate(case_ids):
            if i in dice_by_idx:
                rows.append(
                    {
                        "case_id": cid,
                        "dice": float(dice_by_idx[i]),
                        "difficulty_dice": float(1.0 - dice_by_idx[i]),
                        "uncertainty": float(max(0.0, min(1.0, entropy_case[i].item()))),
                        "embedding": emb[i].detach().cpu().numpy(),
                    }
                )
        return rows

    def create_model(self):
        model = create_model_from_conf(self.model_params, self.plan)
        return model

    def create_loss_fnc(self):
        fg_classes = max(self.model_params["out_channels"] - 1, 1)
        if self.model_params["arch"] == "nnUNet":
            num_pool = 5
            self.net_num_pool_op_kernel_sizes = pool_op_kernels_nnunet(
                self.plan["patch_size"]
            )
            self.deep_supervision_scales = [[1, 1, 1]] + list(
                list(i)
                for i in 1
                / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0)
            )[:-1]
            loss_func = DeepSupervisionLoss(
                levels=num_pool,
                deep_supervision_scales=self.deep_supervision_scales,
                fg_classes=fg_classes,
            )
            self.loss_fnc = loss_func

        elif (
            self.model_params["arch"] == "DynUNet"
            or self.model_params["arch"] == "DynUNet_UB"
        ):
            num_pool = 4  # this is a hack i am not sure if that's the number of pools . this is just to equalize len(mask) and len(pred)
            ds_factors = list(
                il.accumulate(
                    [1]
                    + [
                        2,
                    ]
                    * (num_pool - 1),
                    operator.truediv,
                )
            )
            ds = [1, 1, 1]
            self.deep_supervision_scales = list(
                map(
                    lambda list1, y: [x * y for x in list1],
                    [
                        ds,
                    ]
                    * num_pool,
                    ds_factors,
                )
            )
            loss_func = DeepSupervisionLoss(
                levels=num_pool,
                deep_supervision_scales=self.deep_supervision_scales,
                fg_classes=fg_classes,
            )
            self.loss_fnc = loss_func

        else:
            loss_func = CombinedLoss(**self.loss_params, fg_classes=fg_classes)
            self.loss_fnc = loss_func

class UNetManagerFabric(UNetManager):
    def on_train_batch_start(self, trainer, batch, batch_idx):
        pass

    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        # pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        pass

    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        # pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        pass
