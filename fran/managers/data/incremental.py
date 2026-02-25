# %%
from __future__ import annotations
import os
import re
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from lightning.pytorch import LightningDataModule


from functools import reduce
from operator import add
from pathlib import Path
from typing import Optional

import ipdb
import numpy as np
import torch
from fastcore.basics import listify, operator, warnings
from lightning import LightningDataModule
from monai.data import DataLoader, Dataset, GridPatchDataset, PatchIterd
from monai.data.dataset import CacheDataset, LMDBDataset, PersistentDataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (RandCropByPosNegLabeld,
                                                 RandSpatialCropSamplesD,
                                                 ResizeWithPadOrCropd, SpatialPadd)
from monai.transforms.intensity.dictionary import (RandAdjustContrastd,
                                                   RandScaleIntensityd,
                                                   RandShiftIntensityd)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, Resized
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.utility.dictionary import (EnsureChannelFirstd,
                                                 MapLabelValued, ToDeviceD)
from tqdm.auto import tqdm as pbar
from utilz.fileio import load_dict, load_yaml
from utilz.helpers import (find_matching_fn, project_title_from_folder, resolve_device, set_autoreload)
from utilz.imageviewers import ImageMaskViewer
from utilz.stringz import ast_literal_eval, headline, info_from_filename, strip_extension

from fran.configs.parser import ConfigMaker, is_excel_None
from fran.data.collate import as_is_collated, grid_collated, patch_collated, source_collated, whole_collated
from fran.data.dataset import NormaliseClipd, fg_in_bboxes
from fran.managers.project import Project
from fran.preprocessing.helpers import bbox_bg_only
from fran.transforms.imageio import LoadTorchd, TorchReader
from fran.transforms.intensitytransforms import RandRandGaussianNoised
from fran.transforms.misc_transforms import (DummyTransform, LoadTorchDict,
                                             MetaToDict)
from fran.utils.folder_names import folder_names_from_plan
from fran.utils.misc import convert_remapping

from fran.utils.common import PAD_VALUE
common_vars_filename = os.environ["FRAN_CONF"] + "/config.yaml"
COMMON_PATHS = load_yaml(common_vars_filename)

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
from dataclasses import dataclass, replace

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
set_autoreload()

common_vars_filename = os.environ["FRAN_CONF"] + "/config.yaml"
COMMON_PATHS = load_yaml(common_vars_filename)

tr = ipdb.set_trace

def create_pd_indices( indices:int|list):
        if isinstance(indices,pd.Index):
            return indices
        if isinstance(indices,int):
            indices = range(indices)
        train1_indices = pd.Index(indices)
        return train1_indices

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
        data_folder_train: Optional[str | Path] = None,
        data_folder_valid: Optional[str | Path] = None,
        train1_indices: int|list = 40,
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

        )

        self.train1_indices = train1_indices
        self.loss_threshold = loss_threshold
        self.plan_train = self.configs["plan_train"]
        self.plan_valid = self.configs["plan_valid"]
        if data_folder_train is None:
            self.data_folder_train =  self.derive_data_folder(self.plan_train)
        else: 
            self.data_folder_train = Path(data_folder_train)
        if data_folder_valid is None:
            self.data_folder_valid =  self.derive_data_folder(self.plan_valid)
        else: 
            self.data_folder_valid = Path(data_folder_valid)

    def derive_data_folder(self, plan):
        mode = plan["mode"]
        key = "data_folder_{}".format(mode)
        folders = folder_names_from_plan(self.project, plan)
        data_folder = folders[key]
        data_folder = Path(data_folder)
        if not data_folder.exists() or len(list(data_folder.rglob("*.pt"))) == 0:
            raise Exception(f"Data folder {data_folder} does not exist")
        return data_folder


    def prepare_data(self):
        if not hasattr(self,"train_df"):
            train1_dicts, train2_dicts, valid_dicts = self.create_dataframes()
        else:
            train1_dicts, train2_dicts, valid_dicts = self.access_dataframes()
        self.data_train1, self.data_train2, self.data_valid = train1_dicts, train2_dicts, valid_dicts
        cprint("Number of files in train1: {}".format(len(self.data_train1)), color="yellow")
        cprint("Number of files in train2: {}".format(len(self.data_train2)), color="yellow")
        cprint("Number of files in valid: {}".format(len(self.data_valid)), color="yellow")
        self._build_managers()

    def create_dataframes(self):
        train_cases, valid_cases = self.project.get_train_val_files(
            self.configs["dataset_params"]["fold"], self.configs["plan_train"]["datasources"]
        )
        train_dicts = self.create_data_dicts(train_cases,self.plan_train,self.data_folder_train)
        self.train_df = pd.DataFrame(train_dicts)
        self.train_df["case_id"] = self.train_df["image"].apply(case_id_from_col)
        self.train_df["used_in_training"] = False
        self.train1_indices = create_pd_indices(self.train1_indices)
        self.update_dataframe_indices(self.train1_indices)

        train1_dicts = self.train_df.iloc[self.train1_indices].to_dict("records")
        train2_dicts = self.train_df[self.train_df["used_in_training"] == False].to_dict("records")
        

        valid_dicts = self.create_data_dicts(valid_cases,self.plan_valid, self.data_folder_valid )
        self.valid_df = pd.DataFrame(valid_dicts)
        return train1_dicts, train2_dicts, valid_dicts

    def update_dataframe_indices(self, indices):
        self.train_df.iloc[indices, self.train_df.columns.get_loc("used_in_training")] = True

    def access_dataframes(self):
        existing_ = self.train_df.index[self.train_df["used_in_training"] == True]

        print("Already used in training {}".format(len(existing_)))
        self.train1_indices = create_pd_indices(self.train1_indices)
        if existing_.equals(self.train1_indices):
            cprint("Using same indices as before", color="yellow", italic=True)
        else:
            cprint("Using different indices", color="yellow", italic=True)
            self.update_dataframe_indices(self.train1_indices)

        train1_dicts = self.train_df[self.train_df["used_in_training"] == True].to_dict("records")
        train2_dicts = self.train_df[self.train_df["used_in_training"] == False].to_dict("records")
        valid_dicts = self.valid_df.to_dict("records")
        return train1_dicts, train2_dicts, valid_dicts



    def _build_managers(self):

        train_mode = self.configs["plan_train"]["mode"]
        valid_mode = self.configs["plan_valid"]["mode"]
        tm1_spec  = DataManagerModes.by_mode(train_mode, split="train1")
        tm2_spec  = DataManagerModes.by_mode(train_mode, split="train2")
        val_spec  = DataManagerModes.by_mode(valid_mode,split="valid")


        self.train_manager1 = tm1_spec.manager_cls(
            project=self.project,
            configs=self.configs,
            collate_fn = tm1_spec.collate_fn,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            data = self.data_train1,
            split="train1",
            device=self.device,
            ds_type=self.ds_type,
            keys=self.keys_tr,
            data_folder=self.data_folder_train,
        )
        self.train_manager2 = tm2_spec.manager_cls(
            project=self.project,
            collate_fn = tm2_spec.collate_fn,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            data = self.data_train2,
            split="train2",
            device=self.device,
            ds_type=self.ds_type,
            keys=self.keys_val,
            data_folder=self.data_folder_train,
            configs=self.configs,
        )
        self.valid_manager = val_spec.manager_cls(
            project=self.project,
            configs=self.configs,
            collate_fn = val_spec.collate_fn,
            data = self.data_valid,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="valid",
            device=self.device,
            ds_type=self.ds_type,
            keys=self.keys_val,
            data_folder=self.data_folder_valid,
        )


    def create_data_dicts(self, fnames,plan, data_folder):
        fnames = [strip_extension(fn) for fn in fnames]
        fnames = [fn + ".pt" for fn in fnames]
        images_fldr = data_folder / ("images")
        lms_fldr = data_folder / ("lms")
        inds_fldr = self.infer_inds_fldr(plan, data_folder)
        images = list(images_fldr.glob("*.pt"))

        data = []

        for fn in pbar(fnames):
            fn = Path(fn)
            img_fn = find_matching_fn(fn.name, images, ["all"])[0]
            lm_fn = find_matching_fn(fn.name, lms_fldr, ["all"])[0]
            indices_fn = inds_fldr / img_fn.name
            assert img_fn.exists(), "Missing image {}".format(img_fn)
            assert lm_fn.exists(), "Missing labelmap fn {}".format(lm_fn)
            dici = {"image": img_fn, "lm": lm_fn, "indices": indices_fn}
            data.append(dici)
        return data

    def infer_inds_fldr(self, plan, data_folder):
        fg_indices_exclude = plan["fg_indices_exclude"]
        if is_excel_None(fg_indices_exclude):
            fg_indices_exclude = None
            indices_subfolder = "indices"
        else:
            if isinstance(fg_indices_exclude, str):
                fg_indices_exclude = ast_literal_eval(fg_indices_exclude)
            fg_indices_exclude = listify(fg_indices_exclude)
            indices_subfolder = "indices_fg_exclude_{}".format(
                "".join([str(x) for x in fg_indices_exclude])
            )
        return data_folder / (indices_subfolder)

    def _iter_managers(self):
        # Dual managers only
        return (self.train_manager1, self.train_manager2, self.valid_manager)

    def train_dataloader(self): # not using train1, as lightning expects it this way
        return self.train_manager1.dl

    def train2_dataloader(self):
        return self.train_manager2.dl


class DataManagerI(DataManager):
    def __init__(
        self,
        project,
        configs: dict,
        collate_fn,
        data,
        batch_size=8,
        cache_rate=0.0,
        device="cuda:0",
        ds_type=None,
        split="train",
        save_hyperparameters=False,
        keys=None,
        
        data_folder: Optional[str | Path] = None,
    ):
        super().__init__(
            project=project,
            configs=configs,
            batch_size=batch_size,
            cache_rate=cache_rate,
            collate_fn = collate_fn,
            device=device,
            ds_type=ds_type,
            split=split,
            save_hyperparameters=save_hyperparameters,
            keys=keys,
            data_folder=data_folder,
        )

        self.data= data


    def prepare_data(self):
        pass

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
            self.create_dataset()
            self.create_dataloader()
            try:
                cprint("Size of dataset: {0}, size of dataloader: {1}".format(len(self.ds), len(self.dl)), color="yellow")
            except:
                cprint("Size of dataset not available for {}".format(self.split), color="yellow")


    def create_dataset(self):
        """Create a single dataset based on split type"""
        if self.split == "train1" or self.split == "valid":
            self.ds = self._create_modal_ds()
        else:
            self.ds = self._create_test_ds()

    # 
    # def create_incremental_datasets(self, train_samples: int|list):
    #     if not hasattr(self, "data") or len(self.data) == 0:
    #         print("No data. DS is not being created at this point.")
    #         return 0
    #
    #     indices = self.create_indices(train_samples)
    #     self.data_df.loc[indices,"used_in_training"] = True
    #
    #     data1 = self.data_df.iloc[indices]
    #     data1= data1.to_dict(orient="records")
    #
    #     data2 = self.data_df[~self.data_df["used_in_training"]]
    #     data2 = data2.to_dict(orient="records")
    #     cprint("Size of training dataset: {0}".format(len(data1)), color="green")
    #     cprint("Size of leftover dataset: {0}".format(len(data2)), color="green")
    #     print(f"[DEBUG] Example case: {self.cases[0] if self.cases else 'None'}")
    #
    #     #BUG: self.transforms should be same as valid transforms however
    #     ds1 = Dataset(data=data2, transform=self.transforms)
    #
    #     dstmp = Dataset(
    #                 data=data1,
    #                 transform=self.transforms,
    #             )
    #     patch_iter = PatchIterd(
    #                 keys=["image", "lm"], patch_size=self.plan["patch_size"], mode="constant", constant_values=PAD_VALUE
    #             )
    #     patch_iter = PatchIterdWithPaddingFlag(patch_iter)
    #     ds2 = GridPatchDataset(data=dstmp, patch_iter=patch_iter, with_coordinates=False)
    #     self.ds = [ds1,ds2]
    #
    # def create_incremental_dataloaders(self):
    #     num_workers1 = min(8,self.effective_batch_size * 2)
    #     persistent_workers1 = False
    #     collate_fn1 =self.collate_fn
    #     shuffle1 = True
    #
    #
    #     num_workers2 = 0
    #     persistent_workers2 = False
    #     collate_fn2 = grid_collated
    #     shuffle2= False
    #
    #     self.dl1 =  DataLoader(
    #         self.ds[0],
    #         batch_size=self.effective_batch_size,
    #         num_workers=num_workers1,
    #         collate_fn=collate_fn1,
    #         persistent_workers=persistent_workers1,
    #         pin_memory=True,
    #         shuffle=shuffle1,
    #     )
    #
    #     self.dl2 =  DataLoader(
    #         self.ds[1],
    #         batch_size=self.effective_batch_size,
    #         num_workers=num_workers2,
    #         collate_fn=collate_fn2,
    #         persistent_workers=persistent_workers2,
    #         pin_memory=False,
    #         shuffle=shuffle2,
    #     )
    #     self.dl = self.dl1
    #

    # def add_new_cases(self,n_samples_to_add):
    #     ids = self.data_df.index[self.data_df["used_in_training"] == False]
    #     next_samples = ids[:int(n_samples_to_add)]
    #     existing_samples = self.data_df.index[self.data_df["used_in_training"] == True]
    #     cprint("Samples to start {0}, samples to add {1}".format(len(existing_samples), len(next_samples)), color="yellow", bold=True)
    #     all_samples = np.append(existing_samples, next_samples)
    #     self.start_n = all_samples
    #     self.create_incremental_datasets(all_samples)
    #     self.create_incremental_dataloaders()
    #
    # @staticmethod
    # def _norm_name(value) -> str:
    #     return Path(str(value)).name
    #
    # def add_cases_by_filenames(self, filenames: list[str]) -> list[int]:
    #     if filenames is None or len(filenames) == 0:
    #         return []
    #
    #     target = {self._norm_name(fn) for fn in filenames}
    #     if len(target) == 0:
    #         return []
    #
    #     unused = self.data_df[self.data_df["used_in_training"] == False].copy()
    #     if len(unused) == 0:
    #         return []
    #
    #     image_names = unused["image"].map(self._norm_name) if "image" in unused.columns else pd.Series("", index=unused.index)
    #     lm_names = unused["lm"].map(self._norm_name) if "lm" in unused.columns else pd.Series("", index=unused.index)
    #     mask = image_names.isin(target) | lm_names.isin(target)
    #     selected_idx = list(unused.index[mask])
    #     if len(selected_idx) == 0:
    #         return []
    #
    #     existing_samples = self.data_df.index[self.data_df["used_in_training"] == True]
    #     all_samples = np.append(existing_samples, selected_idx)
    #     self.start_n = all_samples
    #     self.create_incremental_datasets(all_samples)
    #     self.create_incremental_dataloaders()
    #     cprint(
    #         "Added {0} scanned cases by filename. Training set now {1}.".format(
    #             len(selected_idx), len(all_samples)
    #         ),
    #         color="yellow",
    #         bold=True,
    #     )
    #     return selected_idx
    #

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


@dataclass(frozen=True)
class DataManagerModeSpec:
    mode: str
    manager_cls: type
    collate_fn: type


class DataManagerModes:
    SOURCE = DataManagerModeSpec(
        mode="source",
        manager_cls=DataManagerSourceI,
        collate_fn=source_collated,
    )
    SOURCEPATCH = DataManagerModeSpec(
        mode="sourcepatch",
        manager_cls=DataManagerPatchI,
        collate_fn=patch_collated,
    )
    WHOLE = DataManagerModeSpec(
        mode="whole",
        manager_cls=DataManagerWholeI,
        collate_fn=whole_collated,
    )
    PATCH = DataManagerModeSpec(
        mode="patch",
        manager_cls=DataManagerPatchI,
        collate_fn=patch_collated,
    )
    LBD = DataManagerModeSpec(
        mode="lbd",
        manager_cls=DataManagerLBDI,
        collate_fn=source_collated,
    )
    BASELINE = DataManagerModeSpec(
        mode="baseline",
        manager_cls=DataManagerBaselineI,
        collate_fn=whole_collated,
    )
    # PBD = DataManagerModeSpec(
    #     mode="pbd",
    #     manager_cls=DataManagerWIDI,
    #     collate_fn=source_collated,
    # )
    #
    _BY_MODE = {
        SOURCE.mode: SOURCE,
        SOURCEPATCH.mode: SOURCEPATCH,
        WHOLE.mode: WHOLE,
        PATCH.mode: PATCH,
        LBD.mode: LBD,
        BASELINE.mode: BASELINE,
        # PBD.mode: PBD,
    }

    @classmethod
    def by_mode(cls, mode: str, split: Optional[str] = None) -> DataManagerModeSpec:
        try:
            spec = cls._BY_MODE[mode]
        except KeyError as exc:
            raise ValueError(
                f"Unrecognized mode: {mode}. Must be one of {list(cls._BY_MODE.keys())}"
            ) from exc
        if split in {"test", "train2"}:
            return replace(spec, collate_fn=grid_collated)
        return spec


def _mode_to_spec():
    return DataManagerModes._BY_MODE.copy()


def _mode_to_class():
    return {mode: spec.manager_cls for mode, spec in _mode_to_spec().items()}

def case_id_from_col(filename):
    if isinstance(filename, str):
        filename_name = filename.split("/")[-1]
    else:
        filename_name= filename.name
    inf = info_from_filename(filename_name)
    return inf["case_id"]

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
    D.setup("fit")
# %%
    D.train1_indices
    D.prepare_data()

    tm = D.train_manager

# %%
    train1_indices= 40
# %%
    train_cases, valid_cases = tm.project.get_train_val_files(
            tm.dataset_params["fold"], tm.plan["datasources"]
        )


# %%
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
    spec = DataManagerModes.by_mode("patch", split="train1")
    print(spec.collate_fn)  # patch_collated
# %%
