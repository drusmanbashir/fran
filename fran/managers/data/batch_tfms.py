from __future__ import annotations

from typing import Optional

from fran.managers.data.main import (
    DataManager,
    DataManagerBaseline,
    DataManagerDual,
    DataManagerLBD,
    DataManagerMulti,
    DataManagerPatch,
    DataManagerRBD,
    DataManagerSource,
    DataManagerWhole,
)
from fran.transforms.batch_affine import BatchRandAffined3D


class DataManagerBTfms(DataManager):
    pass


class DataManagerSourceBTfms(DataManagerSource):
    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        provided_keys = kwargs["keys"] if "keys" in kwargs else None
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        self.keys_tr = "Ld,Rtr,L2,E,F1,F2,ResizePC,N,IntensityTfms"
        self.keys_val = "L,E,N,Remap,ResizeP"
        if provided_keys is None:
            if self.uses_train_keys():
                self.keys = self.keys_tr
            elif self.is_eval_split():
                self.keys = self.keys_val


class DataManagerWholeBTfms(DataManagerWhole):
    def __init__(self, project, configs: dict, batch_size=8, **kwargs):
        provided_keys = kwargs["keys"] if "keys" in kwargs else None
        super().__init__(project, configs, batch_size, **kwargs)
        self.keys_tr = "L,E,F1,F2,ResizeW,N,IntensityTfms"
        self.keys_val = "L,E,ResizeW,N"
        if provided_keys is None:
            if self.uses_train_keys():
                self.keys = self.keys_tr
            elif self.is_eval_split():
                self.keys = self.keys_val


class DataManagerLBDBTfms(DataManagerSourceBTfms, DataManagerLBD):
    pass


class DataManagerRBDBTfms(DataManagerLBDBTfms, DataManagerRBD):
    pass


class DataManagerPatchBTfms(DataManagerPatch):
    def set_tfm_keys(self):
        self.keys_tr = "RP, L,Remap,E,N,F1,F2,ResizePC,IntensityTfms"
        self.keys_val = "RP,L,Remap,E,ResizePC,N "
        self.keys_test = "L,E,N,Remap,ResizeP"
        if self.uses_train_keys():
            self.keys = self.keys_tr
        elif self.is_eval_split():
            self.keys = self.keys_val
        elif self.split == "test":
            self.keys = self.keys_test
        else:
            raise ValueError


class DataManagerBaselineBTfms(DataManagerBaseline):
    def set_tfm_keys(self):
        self.keys_val = "L,Ld,E,ResizePC,N"
        self.keys_tr = self.keys_val


class DataManagerDualBTfms(DataManagerDual):
    def __init__(
        self,
        project_title,
        configs: dict,
        batch_size: int,
        cache_rate=0.0,
        device="cuda",
        ds_type=None,
        save_hyperparameters=True,
        data_folder: Optional[str] = None,
        manager_class_train: Optional[type] = None,
        manager_class_valid: Optional[type] = None,
        train_indices=None,
        val_indices=None,
        val_sampling=1.0,
        debug=False,
        batch_tfms: bool = True,
    ):
        super().__init__(
            project_title=project_title,
            configs=configs,
            batch_size=batch_size,
            cache_rate=cache_rate,
            device=device,
            ds_type=ds_type,
            save_hyperparameters=save_hyperparameters,
            data_folder=data_folder,
            manager_class_train=manager_class_train,
            manager_class_valid=manager_class_valid,
            train_indices=train_indices,
            val_indices=val_indices,
            val_sampling=val_sampling,
            debug=debug,
            batch_tfms=batch_tfms,
        )

    def _create_batch_affine(self):
        affine3d = self.configs["affine3d"]
        return BatchRandAffined3D(
            keys=["image", "lm"],
            mode=["bilinear", "nearest"],
            prob=affine3d["p"],
            rotate_range=affine3d["rotate_range"],
            scale_range=affine3d["scale_range"],
        )

    def infer_manager_classes(self, configs):
        train_mode = configs["plan_train"]["mode"]
        valid_mode = configs["plan_valid"]["mode"]
        mode_to_class = {
            "source": DataManagerSourceBTfms,
            "whole": DataManagerWholeBTfms,
            "pbd": DataManagerPatchBTfms,
            "sourcepbd": DataManagerPatchBTfms,
            "lbd": DataManagerLBDBTfms,
            "rbd": DataManagerRBDBTfms,
            "baseline": DataManagerBaselineBTfms,
        }
        for mode in (train_mode, valid_mode):
            if mode not in mode_to_class:
                raise ValueError(
                    f"Unrecognized mode: {mode}. Must be one of {list(mode_to_class.keys())}"
                )
        return mode_to_class[train_mode], mode_to_class[valid_mode]


class DataManagerMultiBTfms(DataManagerMulti):
    def __init__(
        self,
        project_title,
        configs: dict,
        batch_size: int,
        cache_rate=0.0,
        device="cuda",
        ds_type=None,
        save_hyperparameters=True,
        data_folder: Optional[str] = None,
        manager_class_train: Optional[type] = None,
        manager_class_valid: Optional[type] = None,
        manager_class_test: Optional[type] = None,
        train_indices=None,
        val_indices=None,
        val_sampling=1.0,
        debug=False,
        batch_tfms: bool = True,
    ):
        super().__init__(
            project_title=project_title,
            configs=configs,
            batch_size=batch_size,
            cache_rate=cache_rate,
            device=device,
            ds_type=ds_type,
            save_hyperparameters=save_hyperparameters,
            data_folder=data_folder,
            manager_class_train=manager_class_train,
            manager_class_valid=manager_class_valid,
            manager_class_test=manager_class_test,
            train_indices=train_indices,
            val_indices=val_indices,
            val_sampling=val_sampling,
            debug=debug,
            batch_tfms=batch_tfms,
        )

    def _create_batch_affine(self):
        affine3d = self.configs["affine3d"]
        return BatchRandAffined3D(
            keys=["image", "lm"],
            mode=["bilinear", "nearest"],
            prob=affine3d["p"],
            rotate_range=affine3d["rotate_range"],
            scale_range=affine3d["scale_range"],
        )

    def infer_manager_classes(self, configs):
        train_mode = configs["plan_train"]["mode"]
        valid_mode = configs["plan_valid"]["mode"]
        mode_to_class = {
            "source": DataManagerSourceBTfms,
            "whole": DataManagerWholeBTfms,
            "pbd": DataManagerPatchBTfms,
            "sourcepbd": DataManagerPatchBTfms,
            "lbd": DataManagerLBDBTfms,
            "rbd": DataManagerRBDBTfms,
            "baseline": DataManagerBaselineBTfms,
        }
        for mode in (train_mode, valid_mode):
            if mode not in mode_to_class:
                raise ValueError(
                    f"Unrecognized mode: {mode}. Must be one of {list(mode_to_class.keys())}"
                )
        return mode_to_class[train_mode], mode_to_class[valid_mode]

    def infer_test_manager_class(self, configs):
        test_mode = configs["plan_test"]["mode"]
        mode_to_class = {
            "source": DataManagerSourceBTfms,
            "whole": DataManagerWholeBTfms,
            "pbd": DataManagerPatchBTfms,
            "sourcepbd": DataManagerPatchBTfms,
            "lbd": DataManagerLBDBTfms,
            "rbd": DataManagerRBDBTfms,
            "baseline": DataManagerBaselineBTfms,
        }
        if test_mode not in mode_to_class:
            raise ValueError(
                f"Unrecognized mode: {test_mode}. Must be one of {list(mode_to_class.keys())}"
            )
        return mode_to_class[test_mode]


__all__ = [
    "DataManagerBTfms",
    "DataManagerBaselineBTfms",
    "DataManagerDualBTfms",
    "DataManagerLBDBTfms",
    "DataManagerMultiBTfms",
    "DataManagerPatchBTfms",
    "DataManagerRBDBTfms",
    "DataManagerSourceBTfms",
    "DataManagerWholeBTfms",
]
