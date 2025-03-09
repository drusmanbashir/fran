# %%
import lightning as pl
from torch.utils.data import DataLoader, Subset
from fran.extra.deepcore.deepcore.met.met_utils import (
    submodular_optimizer,
)
from fran.extra.deepcore.deepcore.met.met_utils.euclidean import euclidean_dist_pair_np
from fastcore.net import contextlib
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import shutil
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.utilities.types import STEP_OUTPUT
from monai.transforms.io.dictionary import LoadImaged
from torchinfo import summary
from fran.transforms.imageio import TorchReader
from fran.transforms.misc_transforms import LoadTorchDict, MetaToDict
import ipdb

from utilz.helpers import pp

tr = ipdb.set_trace

import numpy as np
from typing import Any, Union
from pathlib import Path
from fastcore.basics import store_attr
from monai.transforms.croppad.dictionary import (
    RandCropByPosNegLabeld,
    ResizeWithPadOrCropd,
)
from monai.transforms.utility.dictionary import EnsureChannelFirstd

import psutil
import torch._dynamo
from fran.callback.nep import NeptuneImageGridCallback

from fran.evaluation.losses import CombinedLoss, DeepSupervisionLoss
from fran.managers.data import (
    DataManagerLBD,
    DataManagerWID,
    DataManagerPatch,
    DataManagerSource,
    DataManagerWhole,
)
from utilz.imageviewers import ImageMaskViewer

torch._dynamo.config.suppress_errors = True
from fran.managers.nep import NeptuneManager
import itertools as il
import operator
import warnings
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer as TrainerL
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    TQDMProgressBar,
    DeviceStatsMonitor,
)
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

import torch
from lightning.pytorch import LightningModule, Trainer


def fix_dict_keys(input_dict, old_string, new_string):
    output_dict = {}
    for key in input_dict.keys():
        neo_key = key.replace(old_string, new_string)
        output_dict[neo_key] = input_dict[key]
    return output_dict




def resolve_datamanager(mode: str):
        if mode == "patch":
            DMClass = DataManagerPatch
        elif mode == "source":
            DMClass = DataManagerSource
        elif mode == "whole":
            DMClass = DataManagerWhole
        elif mode == "lbd":
            DMClass = DataManagerLBD
        elif mode == "pbd":
            DMClass = DataManagerWID
        else:
            raise NotImplementedError(
                "Mode {} is not supported for datamanager".format(mode)
            )
        return DMClass

class CRAIGCallback(Callback):

    def __init__(self, selection_batch, num_workers=6):
        super().__init__()
        self.selection_batch = selection_batch
        self.num_workers = num_workers

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Runs CRAIG-based subset selection at the start of each training epoch.
        The model is temporarily set to eval mode during gradient computation,
        and Lightning ensures it is set back to train mode automatically.
        """
        # Temporarily set the model to evaluation mode to compute gradients
        pl_module.model.eval()
        # Perform subset selection using CRAIG
        selected_subset = pl_module.select_subset_with_craig()
        # Replace the train dataloader with the subset dataloader
        trainer.train_dataloader = DataLoader(
            dataset=Subset(trainer.train_dataloader.datasetq, selected_subset),
            batch_size=self.selection_batch,
            shuffle=True,
            num_workers=self.num_workers,
        )


class StoreInfo(Callback):
    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.dicis = []
        return super().on_train_epoch_start(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        model = pl_module.model

        tr()
        grad_z_normed = pl_module.loss_fnc.grad_L_z_normed
        # Gi_inside = model.grad_L_x * model.grad_sigma_z[0]
        # Gi_inside_normed_batch = [torch.linalg.norm(G) for G in Gi_inside]
        # # Gi_inside_normed = torch.stack(Gi_inside_normed_batch)
        # Gi_inside_normed.shape
        L_rho = 5

        Gi = Gi_inside_normed_batch * L_rho
        ks = batch["image"].meta["filename_or_obj"]
        dici = {k: G.item() for k, G in zip(ks, Gi)}
        self.dicis.append(dici)
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


# from fran.managers.base import *


# class NeptuneCallback(Callback):
# def on_train_epoch_start(self, trainer, pl_module):
#     trainer.logger.experiment["training/epoch"] = trainer.current_epoch


class UNetTrainer2(LightningModule):
    def __init__(
        self,
        project,
        dataset_params,
        model_params,
        loss_params,
        lr=None,
    ):
        super().__init__()
        self.lr = lr if lr else model_params["lr"]
        store_attr()
        self.save_hyperparameters("model_params", "loss_params", "lr")
        self.model = self.create_model()
        self.loss_fnc = self.create_loss_fnc()

    # def on_fit_start(self):
    #     super().on_fit_start()

    def _common_step(self, batch, batch_idx):
        if not hasattr(self, "batch_size"):
            self.batch_size = batch["image"].shape[0]
        inputs, target = batch["image"], batch["lm"]
        pred = self.forward(
            inputs
        )  # self.pred so that NeptuneImageGridCallback can use it

        loss_dict = self.loss_fnc(pred, target)
        # loss = loss_dict['loss_mean']
        # loss_logging= loss_dict ['ds_tuple0']
        self.maybe_store_preds(pred)
        return loss_dict

    def on_after_backward(self) -> None:
        self.grad_norm = self.compute_gradient_norm()

        return super().on_after_backward()

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
        loss_dict = self._common_step(batch, batch_idx)
        self.log_losses(loss_dict, prefix="train")
        return  loss_dict

    def validation_step(self, batch, batch_idx):
        loss, loss_dict= self._common_step(batch, batch_idx)
        self.log_losses(loss_dict, prefix="val")
        return loss_dict

    def log_losses(self, loss_dict, prefix):
        losses_logging = loss_dict['losses_for_logging']
        metrics = [
            "loss",
            "loss_ce",
            "loss_dice",
            "loss_dice_label1",
            "loss_dice_label2",
        ]
        metrics = [me for me in metrics if me in losses_logging.keys()]
        renamed = [prefix + "_" + nm for nm in metrics]
        logger_dict = {
            neo_key: losses_logging[key] for neo_key, key in zip(renamed, metrics)
        }
        self.log_dict(
            logger_dict,
            logger=True,
            batch_size=self.batch_size,
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
                "interval":"epoch",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        return output

    def forward(self, inputs):
        preds = self.model(inputs)
        return preds

    def create_model(self):
        model = create_model_from_conf(self.model_params, self.dataset_params)
        return model

    def create_loss_fnc(self):
        if self.model_params["arch"] == "nnUNet":
            num_pool = 5
            self.net_num_pool_op_kernel_sizes = pool_op_kernels_nnunet(
                self.dataset_params["patch_size"]
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
                softmax=True,
            )
            return loss_func

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
            return loss_func

        else:
            loss_func = CombinedLoss(
                **self.loss_params, fg_classes=self.model_params["out_channels"] - 1
            )
            return loss_func


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

class Trainer:
    def __init__(self, project, config, run_name=None):
        store_attr()
        self.ckpt = None if run_name is None else checkpoint_from_model_id(run_name)
        self.qc_config(config, project)

    def setup(
        self,
        batch_size=None,
        logging_freq=25,
        lr=None,
        devices=1,
        compiled=None,
        cbs=[],
        neptune=True,
        profiler=False,
        tags=[],
        description="",
        epochs=600,
        batchsize_finder=False,
    ):
        self.maybe_alter_config(batch_size, batchsize_finder, compiled)
        self.set_lr(lr)
        self.set_strategy(devices)
        self.init_dm_unet(epochs)
        cbs, logger, profiler = self.init_cbs(neptune, profiler, cbs, tags, description)
        self.D.prepare_data()

        if self.config["model_params"]["compiled"] == True:
            self.N = torch.compile(self.N)

        self.trainer = TrainerL(
            callbacks=cbs,
            accelerator="gpu",
            devices=devices,
            precision="bf16-mixed",
            profiler=profiler,
            logger=logger,
            max_epochs=epochs,
            log_every_n_steps=logging_freq,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            default_root_dir=self.project.checkpoints_parent_folder,
            strategy=self.strategy,
            # strategy='ddp_find_unused_parameters_true'
        )

    def init_dm_unet(self, epochs):
        if self.ckpt:
            self.D = self.load_dm()
            self.config["dataset_params"] = self.D.dataset_params
            self.N = self.load_trainer()

        else:
            self.D = self.init_dm()
            self.N = self.init_unet_trainer(epochs)

    def set_lr(self, lr):
        if lr and not self.ckpt:
            self.lr = lr
        elif lr and self.ckpt:
            self.lr = lr
            self.state_dict = torch.load(self.ckpt)
            self.state_dict["lr_schedulers"][0]["_last_lr"][0] = lr
            torch.save(self.state_dict, self.ckpt)

        elif lr is None and self.ckpt:
            self.state_dict = torch.load(self.ckpt)
            self.lr = self.state_dict["lr_schedulers"][0]["_last_lr"][0]
        else:
            self.lr = self.config["model_params"]["lr"]

    def init_cbs(self, neptune, profiler, cbs, tags, description):
        cbs += [
            ModelCheckpoint(
                save_last=True,
                monitor="val_loss",
                every_n_epochs=10,
                # mode="min",
                filename="{epoch}-{val_loss:.2f}",
                enable_version_counter=True,
                auto_insert_metric_name=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=3),
        ]
        if neptune == True:
            logger = NeptuneManager(
                project=self.project,
                run_id=self.run_name,
                log_model_checkpoints=False,  # Update to True to log model checkpoints
                tags=tags,
                description=description,
                capture_stdout=True,
                capture_stderr=True,
                capture_traceback=True,
                capture_hardware_metrics=True,
            )
            N = NeptuneImageGridCallback(
                classes=self.config["model_params"]["out_channels"],
                patch_size=self.config["dataset_params"]["patch_size"],
            )

            cbs += [N]
        else:
            logger = None

        if profiler == True:
            profiler = AdvancedProfiler(
                dirpath=self.project.log_folder, filename="profiler"
            )
            # profiler ='simple'
            cbs += [DeviceStatsMonitor(cpu_stats=True)]
        else:
            profiler = None

        return cbs, logger, profiler

    def set_strategy(self, devices):
        self.strategy = maybe_ddp(devices)
        if type(devices) == int and devices > 1:
            self.sync_dist = True
        else:
            self.sync_dist = False

    def maybe_alter_config(self, batch_size, batchsize_finder, compiled):
        if batch_size:
            self.config["dataset_params"]["batch_size"] = batch_size
            # batch_size = self.config["dataset_params"]["batch_size"]
        if (
            batchsize_finder == True
        ):  # note even if you set a batchsize, that will be overridden by this.
            batch_size = self.heuristic_batch_size()
            self.config["dataset_params"]["batch_size"] = batch_size
        if compiled:
            self.config["model_params"]["compiled"] = compiled

    def qc_config(self, config, project):
        ratios = config["dataset_params"]["fgbg_ratio"]
        labels_fg = project.global_properties["labels_all"]
        labels = [0] + labels_fg
        if isinstance(ratios, list):
            assert (
                a := (len(ratios)) == (b := len(labels))
            ), "Class ratios {0} do not match number of labels in dataset {1}".format(
                a, b
            )
        else:
            assert isinstance(
                ratios, int
            ), "If no list is provided, fgbg_ratio must be an integer"

    def heuristic_batch_size(self):
        ram = psutil.virtual_memory()[3] / 1e9
        if ram < 15:
            return 6
        elif ram > 15 and ram < 32:
            return 8
        elif ram > 32 and ram < 48:
            return 20
        else:
            return 48

    def init_dm(self):
        cache_rate = self.config["dataset_params"]["cache_rate"]
        ds_type = self.config["dataset_params"]["ds_type"]
        DMClass = resolve_datamanager(self.config["plan"]["mode"])
        D = DMClass(
            self.project,
            dataset_params=self.config["dataset_params"],
            config=self.config,
            transform_factors=self.config["transform_factors"],
            affine3d=self.config["affine3d"],
            batch_size=self.config["dataset_params"]["batch_size"],
            cache_rate=cache_rate,
            ds_type=ds_type,
        )

        return D

    def init_unet_trainer(self, epochs):
        N = UNetTrainerCraig(
            self.project,
            self.config["dataset_params"],
            self.config["model_params"],
            self.config["loss_params"],
            lr=self.lr,
        )
        return N

    def load_trainer(self, **kwargs):
        try:
            N = UNetTrainerCraig.load_from_checkpoint(
                self.ckpt,
                project=self.project,
                dataset_params=self.config["dataset_params"],
                lr=self.lr,
                **kwargs,
            )
            print("Model loaded from checkpoint: ", self.ckpt)
        except:
            tr()
            ckpt_state = self.state_dict["state_dict"]
            ckpt_state_updated = fix_dict_keys(ckpt_state, "model", "model._orig_mod")
            # print(ckpt_state_updated.keys())
            state_dict_neo = self.state_dict.copy()
            state_dict_neo["state_dict"] = ckpt_state_updated
            ckpt_old = self.ckpt.str_replace("_bkp", "")
            ckpt_old = self.ckpt.str_replace(".ckpt", ".ckpt_bkp")
            torch.save(state_dict_neo, self.ckpt)
            shutil.move(self.ckpt, ckpt_old)

            N = UNetTrainer.load_from_checkpoint(
                self.ckpt,
                project=self.project,
                dataset_params=self.config["dataset_params"],
                lr=self.lr,
                **kwargs,
            )
        return N

    def load_dm(self):
        DMClass = resolve_datamanager(
            self.state_dict["datamodule_hyper_parameters"]["config"]["plan"]["mode"]
        )
        D = DMClass.load_from_checkpoint(self.ckpt, project=self.project)
        return D

    def resolve_datamanager(self, mode: str):
        if mode == "patch":
            DMClass = DataManagerPatch
        elif mode == "source":
            DMClass = DataManagerSource
        elif mode == "whole":
            DMClass = DataManagerWhole
        elif mode == "lbd":
            DMClass = DataManagerLBD
        elif mode == "pbd":
            DMClass = DataManagerWID
        else:
            raise NotImplementedError(
                "Mode {} is not supported for datamanager".format(mode)
            )
        return DMClass

    def fit(self):
        self.trainer.fit(model=self.N, datamodule=self.D, ckpt_path=self.ckpt)

    def fix_state_dict_keys(self, bad_str="model", good_str="model._orig_mod"):
        state_dict = torch.load(self.ckpt)
        ckpt_state = state_dict["state_dict"]
        ckpt_state_updated = fix_dict_keys(ckpt_state, bad_str, good_str)
        state_dict_neo = state_dict.copy()
        state_dict_neo["state_dict"] = ckpt_state_updated

        ckpt_old = self.ckpt.str_replace(".ckpt", ".ckpt_bkp")
        shutil.move(self.ckpt, ckpt_old)

        torch.save(state_dict_neo, self.ckpt)
        return ckpt_old


# %%
if __name__ == "__main__":
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>
    # from fran.utils.common import *

    import torch

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")

    from fran.utils.common import *

    project_title = "litsmc"
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_config_wholeimage.xlsx"
    )
    configuration_filename = "/s/fran_storage/projects/litsmc/experiment_config.xlsx"
    configuration_filename = None

    conf = ConfigMaker(proj, raytune=False).config

    # conf['model_params']['lr']=1e-3

    # conf['dataset_params']['plan']=5
# %%
    device_id = 1
    # run_name = "LITS-1007"
    # device_id = 0
    run_totalseg = "LITS-1025"
    run_litsmc = "LITS-1018"
    run_name = None
    bs = 2  # 5 is good if LBD with 2 samples per case
    # run_name ='LITS-1003'
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = False
    tags = []
    cbs = [StoreInfo()]
    cbs = []
    description = f""
# %%
# SECTION:-------------------- IMPORTANCE SAMPLING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR>

# %%
    Tm = Trainer(proj, conf, run_name)
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=50 if profiler == False else 1,
        batchsize_finder=batch_finder,
        profiler=profiler,
        cbs=cbs,
        neptune=neptune,
        tags=tags,
        description=description,
    )
# %%
    # Tm.D.setup()

    Tm.N.compiled = compiled
    Tm.fit()

    # model(inputs)
# %%

# %%
    Tm.trainer.callbacks[0].dicis

    import pandas as pd

    df = pd.DataFrame(Tm.trainer.callbacks[0].dicis)
    df.to_csv("df.csv")
# %%

    patch_size = [128, 96, 96]
    summ = summary(
        Tm.N,
        input_size=tuple([1, 1] + patch_size),
        col_names=["input_size", "output_size", "kernel_size"],
        depth=4,
        verbose=0,
        device="cpu",
    )

    # Redirect the output of summary() to a file
# %%
    output_file_path = "model_summary.txt"
    with open(output_file_path, "w") as f:
        with contextlib.redirect_stdout(f):
            print(summ)

# %%
    #     pred = torch.load("pred.pt")
    #     target = torch.load("target.pt")
# %%
    #
    #     Tm.trainer.model.to('cpu')
    #     pred = [a.cpu() for a in pred]
    #     loss = Tm.trainer.model.loss_fnc(pred.cpu(), target.cpu())
    #     loss_dict = Tm.trainer.loss_fnc.loss_dict
    #     Tm.trainer.maybe_store_preds(pred)
    #     # preds = [pred.tensor() if hasattr(pred, 'tensor') else pred for pred in preds]
    #     torch.save(preds, 'new_pred.pt')
    #     torch.save(targ.tensor(),'new_target.pt')
    #
    #     tt = torch.tensor(targ.clone().detach())
    #     torch.save(tt,"new_target.pt")
    #
# %%
# %%
# SECTION:-------------------- HOOKS-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR>

# %%
    def c_hook(grad):
        print(grad)
        return grad + 2

# %%
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    c = a * b

    j = a**2
    j.retain_grad()
    j.backward()
    # c.register_hook(c_hook)
    c.register_hook(lambda grad: print(grad))
    # c.retain_grad()

    d = torch.tensor(4.0, requires_grad=True)
    e = d * c
    e.register_hook(lambda grad: grad * 1)
    e.retain_grad()

# %%
    e.backward()
    print(c.grad)

# %%
# %%
# SECTION:-------------------- Backward hooks-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR>

    def c_hook(module, inp, outp):
        print("=" * 50)
        print("Module:", module)
        print(module.mult)
        print("inp:", inp)
        print("outp:", outp)

    from torch import nn

    class Func(nn.Module):
        def __init__(self, mult: int) -> None:
            super().__init__()
            self.mult = mult

        def forward(self, x):
            return self.mult * x

    class Pow(nn.Module):
        def __init__(self, mult: int) -> None:
            super().__init__()
            self.mult = mult

        def forward(self, x):
            return x**self.mult

# %%

    # def f2(x):
    #     return x * 3  # Multiplication by 3

    j1 = Func(10)
    f2 = Func(2)
    f3 = Func(3)
    f4 = Pow(2)
# %%
    h2 = f2.register_full_backward_hook(c_hook)
    h3 = f3.register_full_backward_hook(c_hook)
    h4 = f4.register_full_backward_hook(c_hook)
    j1.register_full_backward_hook(c_hook)
    x = torch.tensor(2.0, requires_grad=True)
    y2 = f2(x)
    j2 = j1(y2)
    y3 = f3(y2)
    y4 = f4(y3)
# %%
    j2.backward(retain_graph=True)
    y4.backward()
# %%

    y2.backward()
    # Define a tensor with requires_grad=True to track operations on it

    # Forward pass through the composed functions

    # Define a backward hook function
    def backward_hook(module, grad_input, grad_output):
        print(f"Grad Input: {grad_input}")
        print(f"Grad Output: {grad_output}")

    # Register backward hook on y1 (the output of f1)
    y2.retain_grad()  # Ensure we keep the gradient for intermediate tensor y1
    y2_hook = y2.register_hook(backward_hook)

    # Perform backward pass (compute gradients)
    y2.backward()

    # Check gradients
    print(f"x.grad: {x.grad}")

    # Remove the hook after use
    y1_hook.remove()
# %%
    d = torch.tensor(4.0, requires_grad=True)
    d.register_hook(lambda grad: print(grad))
    e = d * c
    e.register_hook(lambda grad: print(grad))
    e.retain_grad()
    e.backward()

# %%
# %%
# SECTION:-------------------- TROUBLESHOOTING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>

    Tm.D.setup()
    D = Tm.D
    ds = Tm.D.valid_ds
# %%

    dl = Tm.D.train_dataloader()
    iteri2 = iter(dl)
    # iteri = iter(dl)
    batch = next(iteri2)
# %%
# %%
    while iteri2:
        bb = next(iteri2)
        # pred = Tm.trainer.model(bb['image'].cuda())
        print(bb["lm"].unique())

    dicis = []
# %%
    for i, id in enumerate(ds):

        lm = id["lm"]
        vals = lm.unique()
        print(vals)
        # print(vals)
        if vals.max() > 8:
            tr()
            # print("Rat")
            dici = {"lm": lm.meta["filename_or_obj"], "vals": vals}
            dicis.append(dici)
    #
    # vals = [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8., 118.]
# %%
    dici = ds[7]
    dici = ds.data[7]
    dici = ds.transform(dici)

# %%
    L = LoadImaged(
        keys=["image", "lm"],
        image_only=True,
        ensure_channel_first=False,
        simple_keys=True,
    )
    L.register(TorchReader())

    E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
    Rtr = RandCropByPosNegLabeld(
        keys=["image", "lm"],
        label_key="lm",
        image_key="image",
        fg_indices_key="lm_fg_indices",
        bg_indices_key="lm_bg_indices",
        image_threshold=-2600,
        spatial_size=D.src_dims,
        pos=3,
        neg=1,
        num_samples=D.plan["samples_per_file"],
        lazy=True,
        allow_smaller=False,
    )
    Ld = LoadTorchDict(keys=["indices"], select_keys=["lm_fg_indices", "lm_bg_indices"])

    Rva = RandCropByPosNegLabeld(
        keys=["image", "lm"],
        label_key="lm",
        image_key="image",
        image_threshold=-2600,
        fg_indices_key="lm_fg_indices",
        bg_indices_key="lm_bg_indices",
        spatial_size=D.dataset_params["patch_size"],
        pos=3,
        neg=1,
        num_samples=D.plan["samples_per_file"],
        lazy=False,
        allow_smaller=True,
    )
    Re = ResizeWithPadOrCropd(
        keys=["image", "lm"],
        spatial_size=D.dataset_params["patch_size"],
        lazy=False,
    )

    Ind = MetaToDict(keys=["lm"], meta_keys=["lm_fg_indices", "lm_bg_indices"])
# %%
    D.prepare_data()
    D.setup(None)
# %%
    dici = D.data_train[0]
    D.valid_ds.data[0]

# %%
    dici = D.valid_ds.data[7]
    dici = L(dici)
    dici = Ind(dici)
    dici = Ld(dici)
    dici = D.transforms_dict["E"](dici)
    dici = D.transforms_dict["Rva"](dici)
    dici = Re(dici[1])

# %%
    ImageMaskViewer([dici[0]["image"][0], dici[0]["lm"][0]])

# %%
    Ld = LoadTorchDict(keys=["indices"], select_keys=["lm_fg_indices", "lm_bg_indices"])
    dici = Ld(dici)
# %%

# %%

    fn = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/images/lits_115.pt"
    fn2 = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/lms/lits_115.pt"
    tt = torch.load(fn)
    tt2 = torch.load(fn2)
    ImageMaskViewer([tt, tt2])
# %%

    Re = ResizeWithPadOrCropd(
        keys=["image", "lm"],
        spatial_size=D.dataset_params["patch_size"],
        lazy=False,
    )
# %%
    dici = Re(dici)
# %%
    dici = ds[1]
    dici = ds.data[0]
    keys_tr = "L,E,Ind,Rtr,F1,F2,A,Re,N,I"
    keys_val = "L,E,Ind,Rva,Re,N"
    keys_tr = keys_tr.split(",")
    keys_val = keys_val.split(",")

# %%
    dici = ds.data[5].copy()
    for k in keys_val[:3]:
        tfm = D.transforms_dict[k]
        dici = tfm(dici)
# %%

    ind = 0
    dici = ds.data[ind]
    ImageMaskViewer([dici["image"][0], dici["lm"][0]])
    ImageMaskViewer([dici[ind]["image"][0], dici[ind]["lm"][0]])
# %%
    tfm2 = D.transforms_dict[keys_tr[5]]

# %%
    for didi in dici:
        dd = tfm2(didi)
# %%
    idx = 0
    ds.set_bboxes_labels(idx)
    if ds.enforce_ratios == True:
        ds.mandatory_label = ds.randomize_label()
        ds.maybe_randomize_idx()

    filename, bbox = ds.get_filename_bbox()
    img, lm = ds.load_tensors(filename)
    dici = {"image": img, "lm": lm, "bbox": bbox}
    dici = ds.transform(dici)

    E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
    dici = E(dici)
# %%
    # img = ds.create_metatensor(img_fn)
    # label = ds.create_metatensor(label_fn)
    dici = ds.data[3]
    dici = ds[3]
    dici[0]["image"]
    dat = ds.data[5]
    dici = ds.transform(dat)
    type(dici)
    dici = ds[4]
    dici.keys()
    dat
    dici = {"image": img_fn, "lm": label_fn}
    im = torch.load(img_fn)

    im.shape

# %%

# %%
    b = next(iteri2)

    b["image"].shape
    m = Tm.N.model
    N = Tm.N

# %%
    for x in range(len(ds)):
        casei = ds[x]
        for a in range(len(casei)):
            print(casei[a]["image"].shape)
# %%
    for i, b in enumerate(dl):
        print("----------------------------")
        print(b["image"].shape)
        print(b["label"].shape)
# %%
    # b2 = next(iter(dl2))
    batch = b
    inputs, target, bbox = batch["image"], batch["lm"], batch["bbox"]

    [pp(a["filename"]) for a in bbox]
# %%
    preds = N.model(inputs.cuda())
    pred = preds[0]
    pred = pred.detach().cpu()
    pp(pred.shape)
# %%
    n = 1
    img = inputs[n, 0]
    mask = target[n, 0]
# %%
    ImageMaskViewer([img.permute(2, 1, 0), mask.permute(2, 1, 0)])
# %%
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacings/litsmall/spc_080_080_150/images/lits_4.pt"
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacings/litstp/spc_080_080_150/images/lits_4.pt"
    fn2 = "/home/ub/datasets/preprocessed/lits32/patches/spc_080_080_150/dim_192_192_128/masks/lits_4_1.pt"
    img = torch.load(fn)
    mask = torch.load(fn2)
    pp(img.shape)
# %%

    ImageMaskViewer([img, mask])
# %%
# %%

    Tm.trainer.callback_metrics
# %%
    ckpt = Path(
        "/s/fran_storage/checkpoints/litsmc/Untitled/LITS-709/checkpoints/epoch=81-step=1886.ckpt"
    )
    kk = torch.load(self.ckpt)
    kk["datamodule_hyper_parameters"].keys()
    kk.keys()
    kk["datamodule_hyper_parameters"]

# %%
    model = trainer.model.model
    model.grad_L_x.shape
    model.grad_sigma_z.shape

# %%

    # Example logits (input to softmax)
    z = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)

    # Example logits (larger and more complex)
    z = torch.tensor([2.5, 1.0, -0.5, 3.0, 4.5], requires_grad=True)

    # Compute softmax using PyTorch
    softmax = torch.nn.functional.softmax(z, dim=0)

    # Method 1: Compute gradient using autograd
    dummy_output = softmax.sum()  # Dummy output for backward pass
    dummy_output.backward()

    # Gradient from autograd
    grad_autograd = z.grad.clone()  # Clone to keep for comparison

    # Reset gradients for manual calculation
    z.grad.zero_()

    # Method 2: Manual softmax gradient calculation
    softmax_manual_grad = torch.zeros(len(z), len(z))

    for i in range(len(z)):
        for j in range(len(z)):
            if i == j:
                softmax_manual_grad[i, j] = softmax[i] * (1 - softmax[i])
            else:
                softmax_manual_grad[i, j] = -softmax[i] * softmax[j]

    # Calculate the gradient w.r.t. logits manually
    grad_manual = softmax_manual_grad @ torch.ones(len(z))

    # Print results
    print("Softmax:", softmax)
    print("Autograd Gradient:", grad_autograd)
    print("Manual Gradient:", grad_manual)
    print("Difference:", torch.abs(grad_autograd - grad_manual))
# %%
