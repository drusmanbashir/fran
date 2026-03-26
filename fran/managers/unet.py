import os
import re
from pathlib import Path

import ipdb
import lightning as L
import torch._dynamo as dynamo
from fran.managers.project import Project
from lightning.pytorch.utilities.types import STEP_OUTPUT
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.post.array import AsDiscrete
from monai.utils.enums import LossReduction
from torch.nn import CrossEntropyLoss
from utilz.cprint import cprint

tr = ipdb.set_trace

from typing import Any, Union

import numpy as np
import torch._dynamo
from fran.evaluation.losses import CombinedLoss, DeepSupervisionLoss
from utilz.stringz import info_from_filename

torch._dynamo.config.suppress_errors = True
import itertools as il
import operator

import torch
import torch.nn.functional as F
from fran.architectures.create_network import (
    create_model_from_conf,
    pool_op_kernels_nnunet,
)
from fran.managers.project import Project
from lightning.pytorch import LightningModule
from monai.losses import DiceLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        self.create_val_loss_fnc()
        super().setup(stage="fit")

    def on_fit_start(self) -> None:
        self.create_val_inferer()
        return super().on_fit_start()

    def create_val_inferer(self):
        sw_device = "cuda"
        device = "cuda"  # or "cpu" if cuda fails oom
        cprint(f"batch size")

        batch_size = 1
        self.val_inferer = SlidingWindowInferer(
            roi_size=self.plan["patch_size"],
            sw_batch_size=batch_size,
            overlap=0,
            mode="constant",
            progress=False,
            sw_device=sw_device,
            device=device,
        )

    @dynamo.disable()
    def _run_swi(self, img):
        # the only thing inside is the SlidingWindowInferer call
        return self.val_inferer(inputs=img, network=self.model)

    def swi_on_val_batch(self, batch, batch_idx):

        img = batch["image"]
        logits = self._run_swi(img)
        if isinstance(logits, tuple):
            logits = logits[0]  # model has deep supervision only 0 channel is needed

        batch["pred"] = logits
        batch["pred"].meta = batch["image"].meta.copy()
        return batch

    def _common_step(self, batch, batch_idx, use_mask=False):
        if not hasattr(self, "batch_size"):
            self.batch_size = batch["image"].shape[0]
        inputs, target = batch["image"], batch["lm"]
        pred = self.forward(
            inputs
        )  # self.pred so that NeptuneImageGridCallback can use it
        batch["pred"] = pred

        loss = self.loss_fnc(pred, target, use_mask=use_mask)
        loss_dict = self.loss_fnc.loss_dict
        self.maybe_store_preds(pred)
        return loss, loss_dict

    def maybe_store_preds(self, pred):
        if hasattr(self.trainer, "store_preds") and self.trainer.store_preds == True:
            if isinstance(pred, Union[tuple, list]):
                self.pred = [p.detach().cpu() for p in pred]
            else:
                self.pred = pred.detach().cpu()

    def _get_deep_supervision_scales(self):
        strides = getattr(self.model, "ds_strides", None)
        if strides is None:
            cprint("Model has no ds_strides", color="red")
            return None
        return list(list(i) for i in 1 / np.cumprod(np.vstack(strides), axis=0))[:-1]

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, loss_dict = self._common_step(batch, batch_idx, use_mask=False)
        self.log_losses(loss_dict, prefix=f"train{dataloader_idx}")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self.swi_on_val_batch(batch, batch_idx)
        pred = batch["pred"]
        target = batch["lm"]
        loss = self.loss_fnc(pred, target, use_mask=False)
        loss_dict = self.loss_fnc.loss_dict

        self.log_losses(loss_dict, prefix=f"val{dataloader_idx}")
        self.maybe_store_preds(pred)
        return loss, loss_dict


    def test_step(self, batch, batch_idx):
        loss, loss_dict = self._common_step(batch, batch_idx, use_mask=True)
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

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=30)
        output = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train0_loss_dice",
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
        _ = self.loss_fnc(pred, target)
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
                        "uncertainty": float(
                            max(0.0, min(1.0, entropy_case[i].item()))
                        ),
                        "embedding": emb[i].detach().cpu().numpy(),
                    }
                )
        return rows

    def create_model(self):
        model = create_model_from_conf(self.model_params, self.plan)
        return model

    def create_val_loss_fnc(self):

        dice_reduction = (
            LossReduction.NONE
        )  # unreduced loss is outputted for logging and will be reduced manually
        self.loss_fnc_val_dce = DiceLoss(
            include_background=False,
            to_onehot_y=False,  # already one-hot
            softmax=True,  # applies softmax to logits,
            reduction=dice_reduction,
        )
        self.loss_fnc_val_ce = CrossEntropyLoss(weight=None, reduction="mean")
        self.lambda_dice = 0.5
        self.lambda_ce = 0.5

    def create_loss_fnc(self):
        fg_classes = max(self.model_params["out_channels"] - 1, 1)
        include_background = bool(self.loss_params.get("include_background", False))
        if self.model_params["arch"] in ["nnUNet", "nnUNet_v1", "ResUNet"]:
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
                include_background=include_background,
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
                include_background=include_background,
            )
            self.loss_fnc = loss_func

        else:
            loss_params = dict(self.loss_params)
            loss_params["include_background"] = include_background
            loss_func = CombinedLoss(**loss_params, fg_classes=fg_classes)
            self.loss_fnc = loss_func


class UNetManagerMulti(UNetManager):
    def __init__(self, project_title, configs, lr=None, sync_dist=False):
        super().__init__(project_title, configs, lr, sync_dist)
        self.Discretize = AsDiscrete(argmax=True, dim=1)

    def _common_step(self, batch, batch_idx, dataloader_idx=0):
        if not hasattr(self, "batch_size"):
            self.batch_size = batch["image"].shape[0]
        inputs, target = batch["image"], batch["lm"]
        pred = self.forward(
            inputs
        )  # self.pred so that NeptuneImageGridCallback can use it
        if dataloader_idx == 1:
            pred = pred[0]
            pred = self.Discretize(pred)
            loss = self.loss_fnc.LossFunc.dice(pred, target)
            loss = loss.flatten()
            loss_dict = {}
            for x in range(len(loss)):
                loss_dict["loss_" + str(x)] = loss[x]
            return loss, loss_dict

        else:
            loss = self.loss_fnc(pred, target)
            loss_dict = self.loss_fnc.loss_dict
            self.maybe_store_preds(pred)
            return loss, loss_dict

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, loss_dict = self._common_step(batch, batch_idx, dataloader_idx)
        self.log_losses(loss_dict, prefix="val{}".format(dataloader_idx))
        return loss


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


# %%
if __name__ == "__main__":
# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
    from fran.configs.parser import ConfigMaker
    from fran.utils.common import *

    P = Project(project_title="kits2")
    C = ConfigMaker(P)
    C.setup(6)
    conf = C.configs
    conf["model_params"]["out_channels"] = 3
# %%

    x = torch.rand(1, 1, 192, 192, 96)
    N = UNetManager(project_title="kits2", configs=conf, lr=0.01)
    N.setup()
    N.model.ds_strides

# %%

    image = torch.load(
        "/r/datasets/preprocessed/kits2/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/images/kits21_00002.pt",
        weights_only=False,
    )
    lm = torch.load(
        "/r/datasets/preprocessed/kits2/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/lms/kits21_00002.pt",
        weights_only=False,
    )
    image = image.unsqueeze(0).unsqueeze(0)
    lm = lm.unsqueeze(0).unsqueeze(0)
    batch = {"image": image[:, :, :128, :128, :64], "lm": lm[:, :, :128, :128, :64]}
# %%

    l, d = N._common_step(batch, 0, 1)

    y = N.model(x)
    N.loss_fnc
    loss = N.loss_fnc(y, x)

    use_mask = False
# %%
    if not hasattr(N, "batch_size"):
        N.batch_size = batch["image"].shape[0]
    inputs, target = batch["image"], batch["lm"]
    pred = N.forward(inputs)  # N.pred so that NeptuneImageGridCallback can use it

# %%
    loss = N.loss_fnc(pred, target, use_mask=use_mask)
    # i)
    # default_experiment_planner
# %%
