import os
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Model Checkpointing
===================

Automatically save model checkpoints during training.
"""

import logging
import os

import lightning.pytorch as pl
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.utilities.rank_zero import WarningCache
from typing_extensions import override

log = logging.getLogger(__name__)
warning_cache = WarningCache()



from lightning.pytorch.callbacks.model_checkpoint import *
from lightning.pytorch.callbacks.model_checkpoint import _PATH
import lightning.pytorch as pl
from typing import override

import ipdb
tr = ipdb.set_trace


class ModelCheckpointUB(ModelCheckpoint):

    @override
    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        super().setup(trainer, pl_module, stage)

    def __warn_if_dir_not_empty(self, dirpath: _PATH) -> None:
        if (
            self.save_top_k != 0
            and _is_dir(self._fs, dirpath, strict=True)
            and len(self._fs.ls(dirpath)) > 0
        ):
            rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
    def __resolve_ckpt_dir(self, trainer: "pl.Trainer") -> _PATH:
        """Determines model checkpoint save directory at runtime. Reference attributes from the trainer's logger to
        determine where to save checkpoints. The path for saving weights is set in this priority:

        1.  The ``ModelCheckpoint``'s ``dirpath`` if passed in
        2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
        3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

        The path gets extended with subdirectory "checkpoints".

        """
        tr()
        if self.dirpath is not None:
            # short circuit if dirpath was passed to ModelCheckpoint
            return self.dirpath

        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

        return ckpt_path


