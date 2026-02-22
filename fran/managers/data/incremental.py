# %%
from __future__ import annotations
import os

from pathlib import Path
from typing import Optional, Tuple



from pathlib import Path
from typing import Optional

import ipdb
import torch
from fastcore.basics import warnings
from utilz.cprint import cprint
from utilz.fileio import load_yaml
from utilz.stringz import headline

from fran.configs.parser import ConfigMaker
from fran.managers.data.training import (
    DataManager,
    DataManagerBaseline,
    DataManagerDual,
    DataManagerLBD,
    DataManagerMulti,
    DataManagerPatch,
    DataManagerShort,
    DataManagerSource,
    DataManagerWID,
    DataManagerWhole,
)
from fran.managers.project import Project

common_vars_filename = os.environ["FRAN_CONF"] + "/config.yaml"
COMMON_PATHS = load_yaml(common_vars_filename)

tr = ipdb.set_trace


class DataManagerDualI(DataManagerDual):
    def __init__(
        self,
        project_title,
        configs: dict,
        batch_size: int,
        cache_rate=0.0,
        device="cuda",
        ds_type=None,
        save_hyperparameters=True,
        keys_tr="L,Remap,Ld,E,N,Rtr,F1,F2,Affine,ResizePC,IntensityTfms",
        keys_val="L,Remap,Ld,E,N,Rva, ResizePC",
        data_folder: Optional[str | Path] = None,
        start_n: int = 40,
        loss_threshold: float = 0.6,
    ):
        super().__init__(
            project_title=project_title,
            configs=configs,
            batch_size=batch_size,
            cache_rate=cache_rate,
            device=device,
            ds_type=ds_type,
            save_hyperparameters=save_hyperparameters,
            keys_tr=keys_tr,
            keys_val=keys_val,
            data_folder=data_folder,
        )
        self.start_n = start_n
        self.loss_threshold = loss_threshold

    def _build_managers(self):
        super()._build_managers()
        if hasattr(self.train_manager, "start_n"):
            self.train_manager.start_n = self.start_n

    def infer_manager_classes(self, configs) -> Tuple[type, type]:
        train_mode = configs["plan_train"]["mode"]
        valid_mode = configs["plan_valid"]["mode"]
        mode_to_class = _mode_to_class()
        for mode in (train_mode, valid_mode):
            if mode not in mode_to_class:
                raise ValueError(
                    f"Unrecognized mode: {mode}. Must be one of {list(mode_to_class.keys())}"
                )
        return mode_to_class[train_mode], mode_to_class[valid_mode]


class DataManagerMultiI(DataManagerMulti):
    def __init__(
        self,
        project_title,
        configs: dict,
        batch_size: int,
        cache_rate=0.0,
        device="cuda",
        ds_type=None,
        save_hyperparameters=True,
        keys_tr="L,Remap,Ld,E,N,Rtr,F1,F2,Affine,ResizePC,IntensityTfms",
        keys_val="L,Remap,Ld,E,N,Rva, ResizePC",
        keys_test="L,E,N,Remap,ResizeP",
        data_folder: Optional[str | Path] = None,
        start_n: int = 40,
    ):
        super().__init__(
            project_title=project_title,
            configs=configs,
            batch_size=batch_size,
            cache_rate=cache_rate,
            device=device,
            ds_type=ds_type,
            save_hyperparameters=save_hyperparameters,
            keys_tr=keys_tr,
            keys_val=keys_val,
            keys_test=keys_test,
            data_folder=data_folder,
        )
        self.start_n = start_n

    def _build_managers(self):
        super()._build_managers()
        if hasattr(self.train_manager, "start_n"):
            self.train_manager.start_n = self.start_n

    def infer_manager_classes(self, configs) -> Tuple[type, type, type]:
        train_mode = configs["plan_train"]["mode"]
        valid_mode = configs["plan_valid"]["mode"]
        test_mode = configs["plan_test"]["mode"]
        mode_to_class = _mode_to_class()
        for mode in (train_mode, valid_mode, test_mode):
            if mode not in mode_to_class:
                raise ValueError(
                    f"Unrecognized mode: {mode}. Must be one of {list(mode_to_class.keys())}"
                )
        return (
            mode_to_class[train_mode],
            mode_to_class[valid_mode],
            mode_to_class[test_mode],
        )


class DataManagerI(DataManager):
    def __init__(
        self,
        project,
        configs: dict,
        batch_size=8,
        cache_rate=0.0,
        device="cuda:0",
        ds_type=None,
        split="train",
        save_hyperparameters=False,
        keys=None,
        data_folder: Optional[str | Path] = None,
        start_n: int = 40,
    ):
        super().__init__(
            project=project,
            configs=configs,
            batch_size=batch_size,
            cache_rate=cache_rate,
            device=device,
            ds_type=ds_type,
            split=split,
            save_hyperparameters=save_hyperparameters,
            keys=keys,
            data_folder=data_folder,
        )
        self.start_n = start_n

    def setup(self, stage: str = None) -> None:
        # Create transforms for this split

        headline(f"Setting up {self.split} dataset. DS type is: {self.ds_type}")
        print("Src Dims: ", self.configs["dataset_params"]["src_dims"])
        print("Patch Size: ", self.plan["patch_size"])
        print("Using fg indices: ", self.plan["use_fg_indices"])

        self.create_transforms()
        self.set_transforms(self.keys)
        print("Transforms are set up: ", self.keys)

        self.create_dataset(self.start_n)
        self.create_dataloader()


    def create_dataset(self, num_train_samples: int):
        if not hasattr(self, "data") or len(self.data) == 0:
            print("No data. DS is not being created at this point.")
            return 0
        if self.split == "train":
            data = self.data[:num_train_samples]
        else:
            data = self.data
        cprint("Size of {0} dataset: {1}".format(self.split,len(data)), color="green")
        print(f"[DEBUG] Example case: {self.cases[0] if self.cases else 'None'}")
        original_data = self.data
        self.data = data
        try:
            if self.split in ("train", "valid"):
                self.ds = super()._create_modal_ds()
            else:
                self.ds = super()._create_test_ds()
            return self.ds
        finally:
            self.data = original_data


    def add_new_cases(self,n_samples_to_add):
        total_cases = self.start_n + n_samples_to_add
        cprint("Size of {0} dataset: {1}".format(self.split, total_cases), color="green")
        self.create_dataset(total_cases)
        self.create_dataloader()


class DataManagerSourceI(DataManagerSource, DataManagerI):
    pass


class DataManagerWholeI(DataManagerWhole, DataManagerI):
    pass


class DataManagerLBDI(DataManagerLBD, DataManagerI):
    pass


class DataManagerWIDI(DataManagerWID, DataManagerI):
    pass


class DataManagerShortI(DataManagerShort, DataManagerI):
    pass


class DataManagerPatchI(DataManagerPatch, DataManagerI):
    pass


class DataManagerBaselineI(DataManagerBaseline, DataManagerI):
    pass


def _mode_to_class():
    return {
        "source": DataManagerSourceI,
        "sourcepatch": DataManagerPatchI,
        "whole": DataManagerWholeI,
        "patch": DataManagerPatchI,
        "lbd": DataManagerLBDI,
        "baseline": DataManagerBaselineI,
        "pbd": DataManagerWIDI,
    }


# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>
if __name__ == "__main__":

    import torch

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")


# %%
    proj_nodes = Project(project_title="nodes")
    CN = ConfigMaker(
        proj_nodes,
    )
    CN.setup(5)
    config_nodes = CN.configs

# %%
#SECTION:-------------------- NODESJ--------------------------------------------------------------------------------------
    batch_size = 2
    ds_type = "lmdb"
    ds_type = None
    config_nodes["dataset_params"]["mode"] = "lbd"
    config_nodes["dataset_params"]["cache_rate"] = 0
    D = DataManagerDualI(
        project_title=proj_nodes.project_title,
        configs=config_nodes,
        batch_size=batch_size,
        ds_type=ds_type,
    )

# %%
    D.prepare_data()
    D.setup()
    tmv = D.valid_manager
    tmt = D.train_manager
    tmv.transforms_dict

# %%
