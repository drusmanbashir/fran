# %%
from __future__ import annotations

from torch.utils.data import DataLoader
from fran.data.collate import grid_collated
from fran.managers.data.training import PatchIterdWithPaddingFlag

from fran.utils.common import PAD_VALUE
from monai.data import Dataset
from monai.data.dataset import PersistentDataset
from monai.data.grid_dataset import GridPatchDataset, PatchIterd
import pandas as pd
import os
import numpy as np

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

    # def train_dataloader(self):
    #     return self.train_manager.dl

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


    def prepare_data(self):
        super().prepare_data()
        self.create_data_df()


    def setup(self, stage: str = None) -> None:
        # Create transforms for this split

        print(stage)
        if stage=="fit":
            headline(f"Setting up {self.split} dataset. DS type is: {self.ds_type}")
            print("Src Dims: ", self.configs["dataset_params"]["src_dims"])
            print("Patch Size: ", self.plan["patch_size"])
            print("Using fg indices: ", self.plan["use_fg_indices"])

            self.create_transforms()
            self.set_transforms(self.keys)
            print("Transforms are set up: ", self.keys)

            cprint("Creating {0} dataset".format(self.split) , color="magenta", italic=True)
            if self.split=="valid":
                self.create_dataset()
                self.create_dataloader()
            elif self.split=="train":
                self.create_incremental_datasets(self.start_n)
                self.create_incremental_dataloaders()
            else:
                raise NotImplementedError
            cprint("Size of dataset: {0}, size of dataloader: {1}".format(len(self.ds), len(self.dl)), color="yellow")

    def create_data_df(self):
        self.data_df = pd.DataFrame(self.data)
        self.data_df["used_in_training"] = False


    def create_indices(self,indices):
        if isinstance(indices,int):
            indices = list(range(indices))
        else:
            indices = indices
        return indices

    
    def create_incremental_datasets(self, train_samples: int|list):
        if not hasattr(self, "data") or len(self.data) == 0:
            print("No data. DS is not being created at this point.")
            return 0

        indices = self.create_indices(train_samples)
        self.data_df.loc[indices,"used_in_training"] = True

        data1 = self.data_df.iloc[indices]
        data1= data1.to_dict(orient="records")

        data2 = self.data_df[~self.data_df["used_in_training"]]
        data2 = data2.to_dict(orient="records")
        cprint("Size of training dataset: {0}".format(len(data1)), color="green")
        cprint("Size of leftover dataset: {0}".format(len(data2)), color="green")
        print(f"[DEBUG] Example case: {self.cases[0] if self.cases else 'None'}")

        #BUG: self.transforms should be same as valid transforms however
        ds1 = Dataset(data=data2, transform=self.transforms)

        dstmp = Dataset(
                    data=data1,
                    transform=self.transforms,
                )
        patch_iter = PatchIterd(
                    keys=["image", "lm"], patch_size=self.plan["patch_size"], mode="constant", constant_values=PAD_VALUE
                )
        patch_iter = PatchIterdWithPaddingFlag(patch_iter)
        ds2 = GridPatchDataset(data=dstmp, patch_iter=patch_iter, with_coordinates=False)
        self.ds = [ds1,ds2]

    def create_incremental_dataloaders(self):
        num_workers1 = min(8,self.effective_batch_size * 2)
        persistent_workers1 = False
        collate_fn1 =self.collate_fn
        shuffle1 = True


        num_workers2 = 0
        persistent_workers2 = False
        collate_fn2 = grid_collated
        shuffle2= False

        self.dl1 =  DataLoader(
            self.ds[0],
            batch_size=self.effective_batch_size,
            num_workers=num_workers1,
            collate_fn=collate_fn1,
            persistent_workers=persistent_workers1,
            pin_memory=True,
            shuffle=shuffle1,
        )

        self.dl2 =  DataLoader(
            self.ds[1],
            batch_size=self.effective_batch_size,
            num_workers=num_workers2,
            collate_fn=collate_fn2,
            persistent_workers=persistent_workers2,
            pin_memory=False,
            shuffle=shuffle2,
        )
        self.dl = self.dl1


    def add_new_cases(self,n_samples_to_add):
        ids = self.data_df.index[self.data_df["used_in_training"] == False]
        next_samples = ids[:int(n_samples_to_add)]
        existing_samples = self.data_df.index[self.data_df["used_in_training"] == True]
        cprint("Samples to start {0}, samples to add {1}".format(len(existing_samples), len(next_samples)), color="yellow", bold=True)
        all_samples = np.append(existing_samples, next_samples)
        self.start_n = all_samples
        self.create_incremental_datasets(all_samples)
        self.create_incremental_dataloaders()

    @staticmethod
    def _norm_name(value) -> str:
        return Path(str(value)).name

    def add_cases_by_filenames(self, filenames: list[str]) -> list[int]:
        if filenames is None or len(filenames) == 0:
            return []

        target = {self._norm_name(fn) for fn in filenames}
        if len(target) == 0:
            return []

        unused = self.data_df[self.data_df["used_in_training"] == False].copy()
        if len(unused) == 0:
            return []

        image_names = unused["image"].map(self._norm_name) if "image" in unused.columns else pd.Series("", index=unused.index)
        lm_names = unused["lm"].map(self._norm_name) if "lm" in unused.columns else pd.Series("", index=unused.index)
        mask = image_names.isin(target) | lm_names.isin(target)
        selected_idx = list(unused.index[mask])
        if len(selected_idx) == 0:
            return []

        existing_samples = self.data_df.index[self.data_df["used_in_training"] == True]
        all_samples = np.append(existing_samples, selected_idx)
        self.start_n = all_samples
        self.create_incremental_datasets(all_samples)
        self.create_incremental_dataloaders()
        cprint(
            "Added {0} scanned cases by filename. Training set now {1}.".format(
                len(selected_idx), len(all_samples)
            ),
            color="yellow",
            bold=True,
        )
        return selected_idx


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
    df = pd.DataFrame(tmt.data)
    df["in_use"]=False
    indices = [0,20]

    df.loc[indices, "in_use"] = True

    data = tmt.data_df.iloc[tmt.indices]
    data[:3].to_dict(orient="records")
# %%
    df.loc["in_use"]==True
    bads = df.index[df["in_use"] == False]

    df.loc[bads]
# %%
    df = tmt.data_df
    ids = df.index[df["used_in_training"] == False]

    df.loc[ids]["in_use"] = False

# %%

    df = pd.DataFrame(

      {"case": ["a", "b", "c", "d"], "used_in_training": [False, False, False, False]}
    )
    indices = [0, 2]

# %%
    print("Initial:")
    print(df)

# BUGGY pattern (chained assignment) -> often does NOT update original df
    df_bug = df.copy()
    df_bug.loc[indices]["used_in_training"] = True
    print("\nAfter chained assignment df_bug.loc[indices]['used_in_training'] = True:")
    print(df_bug)
    print("num_true:", int(df_bug["used_in_training"].sum()))

# CORRECT pattern (single-step assignment) -> updates original df
    df_ok = df.copy()
    df_ok.loc[indices, "used_in_training"] = True
    print("\nAfter direct assignment df_ok.loc[indices, 'used_in_training'] = True:")
    print(df_ok)
    print("num_true:", int(df_ok["used_in_training"].sum()))
# %%
        
# %%
