from label_analysis.merge import pbar
from lightning.pytorch import Trainer as TrainerL
from lightning.pytorch.callbacks import ModelCheckpoint
import shutil
from lightning.pytorch.profilers import AdvancedProfiler
from monai.transforms.io.dictionary import LoadImaged
from fran.transforms.misc_transforms import MetaToDict
from fran.managers import UNetManager, Project
from fran.managers.unet import maybe_ddp
from fran.transforms.imageio import TorchReader
import ipdb

from fran.configs.parser import ConfigMaker
from utilz.helpers import pp

tr = ipdb.set_trace

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

from fran.managers.data import (
    DataManagerBaseline,
    DataManagerLBD,
    DataManagerWID,
    DataManagerPatch,
    DataManagerSource,
    DataManagerWhole,
)
from utilz.imageviewers import ImageMaskViewer

torch._dynamo.config.suppress_errors = True
from fran.managers.nep import NeptuneManager
import warnings
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    TQDMProgressBar,
    DeviceStatsMonitor,
)

from fran.utils.common import COMMON_PATHS

try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass

import torch

if __name__ == "__main__":
    print("Imported main modules. This works")
