import  os
from fran.managers.project import Project
import ipdb
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as L

tr = ipdb.set_trace

from ipdb.__main__ import get_ipython
import numpy as np
from typing import Any, Union
from fastcore.basics import store_attr
import torch._dynamo

from fran.evaluation.losses import CombinedLoss, DeepSupervisionLoss

torch._dynamo.config.suppress_errors = True
import itertools as il
import operator
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fran.architectures.create_network import (
    create_model_from_conf,
    pool_op_kernels_nnunet,
)
import torch.nn.functional as F

try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass

class UNetManager(LightningModule):
    def __init__(
        self,
        project_title,
        config,
        lr=None,
        sync_dist=False,
    ):
        super().__init__()

        self.sync_dist = sync_dist
        self.project = Project(project_title)
        self.save_hyperparameters("project_title","config","lr")
        self.plan = config["plan_train"]
        self.model_params = config["model_params"]
        self.loss_params = config['loss_params']
        self.lr = lr if lr else self.model_params["lr"]
        self.model = self.create_model()

    def on_fit_start(self):
        self.create_loss_fnc()
        super().on_fit_start()

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

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self._common_step(batch, batch_idx)
        self.log_losses(loss_dict, prefix="val")
        return loss

    def log_losses(self, loss_dict, prefix):
        metrics = loss_dict.keys()
        metrics = [metric for metric in metrics if "batch" not in metric] # too detailed otherwise
        renamed = [prefix + "_" + nm for nm in metrics]
        logger_dict = {
            neo_key: loss_dict[key] for neo_key, key in zip(renamed, metrics)
        }
        self.log_dict(
            logger_dict,
            logger=True,
            batch_size=self.batch_size,
            sync_dist=self.sync_dist,
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

    def create_model(self):
        model = create_model_from_conf(self.model_params, self.plan)
        return model

    def create_loss_fnc(self):
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
                fg_classes=self.model_params["out_channels"] - 1,
            )
            self.loss_fnc=loss_func

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
                fg_classes=self.model_params["out_channels"] - 1,
            )
            self.loss_fnc=loss_func

        else:
            loss_func = CombinedLoss(
                **self.loss_params, fg_classes=self.model_params["out_channels"] - 1
            )
            self.loss_fnc=loss_func


def update_nep_run_from_config(nep_run, config):
    for key, value in config.items():
        nep_run[key] = value
    return nep_run


def maybe_ddp(devices):
    if devices == 1 or isinstance(devices, Union[list, str, tuple]):
        return "auto"
    ip = get_ipython()
    if ip:
        print("Using interactive-shell ddp strategy")
        return "ddp_notebook"
    else:
        print("Using non-interactive shell ddp strategy")
        return "ddp"



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


    # def training_step(self, batch, batch_idx):
        # output = self._common_step(batch, batch_idx,"train")
        # return output
        # loss = output['loss']
        # return loss

    # def validation_step(self, batch, batch_idx):
    #     output = self._common_step(batch, batch_idx,"val")
    #     return output
        # loss = output['loss']
        # return loss
