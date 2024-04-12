
# %%
import ipdb
tr = ipdb.set_trace
from fran.utils.common import common_vars_filename

import numpy as np
from typing import Union
from pathlib import Path
from fastcore.basics import store_attr
from label_analysis.merge import load_dict
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropd
from monai.transforms.utility.dictionary import EnsureChannelFirstd

import psutil
import random
import torch._dynamo
from fran.callback.nep import NeptuneImageGridCallback

from fran.evaluation.losses import CombinedLoss, DeepSupervisionLoss
from fran.managers.data import ( DataManagerLBD, DataManagerPatch,
                                 DataManagerSource)
from fran.utils.fileio import load_yaml
from fran.utils.imageviewers import ImageMaskViewer

torch._dynamo.config.suppress_errors = True
from fran.managers.nep import NeptuneManager
import itertools as il
import operator
import warnings
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ( LearningRateMonitor,
                                         TQDMProgressBar, DeviceStatsMonitor)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fran.architectures.create_network import (create_model_from_conf, 
                                               pool_op_kernels_nnunet)
import torch.nn.functional as F
from fran.transforms.spatialtransforms import one_hot
try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass

import torch
import torch
from fastcore.basics import store_attr
from fran.managers.training import TrainingManager, UNetTrainer, checkpoint_from_model_id


class TrainingManagerTransfer(TrainingManager):
    def __init__(self, project, configs, run_name):
        assert run_name is not None, "Please specificy a run to transfer learning from"
        super().__init__(project, configs, run_name)
        self.run_name = None
    def init_dm_unet(self,epochs):
            self.N = self.load_trainer(max_epochs= epochs)
            self.D = self.init_dm(cache_rate)

    def fit(self):
        self.trainer.fit(model=self.N, datamodule=self.D, ckpt_path=None)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *
    from torch.profiler import profile, record_function, ProfilerActivity
    project_title = "litsmc"
    mnemonic = "liver"
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    )
    configuration_filename = "/s/fran_storage/projects/litsmc/experiment_configs.xlsx"
    configuration_filename = None

    conf = ConfigMaker(
        proj, raytune=False, configuration_mnemonic=mnemonic
    ).config

    global_props = load_dict(proj.global_properties_filename)
    # conf['model_params']['lr']=1e-3

# %%
    device_id = 0
    run_name = None
    bs =1# if none, will get it from the conf file 
    run_name = "LITS-811"
    run_name ='LITS-919'
    run_name = "LITS-913"
    # run_name ='LITS-836'
    compiled = False
    profiler=False

    batch_finder = False
    neptune = True
    tags = []
    cache_rate=0.0
    description = f""
    Tm = TrainingManagerTransfer(project= proj, configs =conf, run_name= run_name)
# %%
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=21,
        batchsize_finder=batch_finder,
        profiler=profiler,
        neptune=neptune,
        tags=tags,
        description=description,
        cache_rate=cache_rate
    )
    # Tm.D.batch_size=8
    Tm.N.compiled = compiled
# %%
    Tm.fit()
 

# %%
    Tm.D.setup()
    D = Tm.D
    ds = Tm.D.train_ds
    ds = Tm.D.valid_ds
# %%
    dl = Tm.D.train_dataloader()
    dl2 = Tm.D.val_dataloader()
    iteri = iter(dl)
    iteri2 = iter(dl2)
    batch = next(iteri2)
    pred = Tm.N(batch['image'].to(0))
#

# %%
