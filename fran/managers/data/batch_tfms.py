from __future__ import annotations
import torch
import warnings
from typing import Optional

from fran.managers.data.main import (
    DataManager,
    DataManagerDual,
    DataManagerLBD,
    DataManagerMulti,
    DataManagerPatch,
    DataManagerRBD,
    DataManagerSource,
    DataManagerWhole,
)
from fran.transforms.batch_affine import BatchRandAffined3D
from fran.transforms.batch_spatial import (
    BatchCenterCropOrPadd,
    BatchRandFlipd,
    BatchResized,
    BatchSpatialPadd,
)


class DataManagerBTfms(DataManager):
    def create_transforms(self):
        super().create_transforms()
        affine3d = self.configs["affine3d"]
        patch_size = self.plan["patch_size"]
        self.transforms_dict["F1"] = BatchRandFlipd(
            keys=["image", "lm"],
            prob=self.flip["prob"],
            spatial_axis=0,
        )
        self.transforms_dict["F2"] = BatchRandFlipd(
            keys=["image", "lm"],
            prob=self.flip["prob"],
            spatial_axis=1,
        )
        self.transforms_dict["Affine"] = BatchRandAffined3D(
            keys=["image", "lm"],
            mode=["bilinear", "nearest"],
            prob=affine3d["p"],
            rotate_range=affine3d["rotate_range"],
            scale_range=affine3d["scale_range"],
        )
        self.transforms_dict["ResizePC"] = BatchCenterCropOrPadd(
            keys=["image", "lm"],
            spatial_size=patch_size,
        )
        self.transforms_dict["ResizeP"] = BatchSpatialPadd(
            keys=["image", "lm"],
            spatial_size=patch_size,
        )
        self.transforms_dict["ResizeW"] = BatchResized(
            keys=["image", "lm"],
            spatial_size=patch_size,
            mode=["linear", "nearest"],
        )


class DataManagerSourceBTfms(DataManagerBTfms, DataManagerSource):
    keys_tr = "Ld,Rtr,L2,E,N,IntensityTfms"
    keys_tr_batch = "F1,F2,Affine,ResizePC"
    keys_val = "L,E,N"
    keys_val_batch = "Remap,ResizeP"
    keys_test = None
    keys_test_batch = None

    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        provided_keys = kwargs["keys"] if "keys" in kwargs else None
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        if provided_keys is None:
            if self.uses_train_keys():
                self.keys = self.keys_tr
            elif self.is_eval_split():
                self.keys = self.keys_val


class DataManagerWholeBTfms(DataManagerBTfms, DataManagerWhole):
    keys_tr = "L,E,N,IntensityTfms"
    keys_tr_batch = "Affine,ResizeW"
    keys_val = "L,E,N"
    keys_val_batch = "ResizeW"
    keys_test = None
    keys_test_batch = None

    def __init__(self, project, configs: dict, batch_size=8, **kwargs):
        provided_keys = kwargs["keys"] if "keys" in kwargs else None
        super().__init__(project, configs, batch_size, **kwargs)
        assert "F1" not in self.keys_tr_batch and "F2" not in self.keys_tr_batch
        if provided_keys is None:
            if self.uses_train_keys():
                self.keys = self.keys_tr
                assert "F1" not in self.keys and "F2" not in self.keys
            elif self.is_eval_split():
                self.keys = self.keys_val


class DataManagerLBDBTfms(DataManagerSourceBTfms, DataManagerLBD):
    pass


class DataManagerRBDBTfms(DataManagerLBDBTfms, DataManagerRBD):
    pass


class DataManagerPatchBTfms(DataManagerBTfms, DataManagerPatch):
    keys_tr = "RP,L,Remap,E,N,IntensityTfms"
    keys_tr_batch = "F1,F2,Affine,ResizePC"
    keys_val = "RP,L,Remap,E,N"
    keys_val_batch = "ResizePC"
    keys_test = "L,E,N,Remap"
    keys_test_batch = "ResizeP"

    def set_tfm_keys(self):
        if self.uses_train_keys():
            self.keys = self.keys_tr
        elif self.is_eval_split():
            self.keys = self.keys_val
        elif self.split == "test":
            self.keys = self.keys_test
        else:
            raise ValueError



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
        }
        if test_mode not in mode_to_class:
            raise ValueError(
                f"Unrecognized mode: {test_mode}. Must be one of {list(mode_to_class.keys())}"
            )
        return mode_to_class[test_mode]


__all__ = [
    "DataManagerBTfms",
    "DataManagerDualBTfms",
    "DataManagerLBDBTfms",
    "DataManagerMultiBTfms",
    "DataManagerPatchBTfms",
    "DataManagerRBDBTfms",
    "DataManagerSourceBTfms",
    "DataManagerWholeBTfms",
]

# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
if __name__ == '__main__':
    from pprint import pp

    from fran.configs.parser import ConfigMaker
    from fran.transforms.imageio import LoadTorchd
    from utilz.imageviewers import ImageMaskViewer

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *

    project_title = "kits23"
    proj = Project(project_title=project_title)

    CL = ConfigMaker(proj)
    CL.setup(2)
    conf = CL.configs
# %%
# SECTION:-------------------- LIDC-------------------------------------------------------------------------------------- <CR>
    batch_size = 4
    ds_type = "lmdb"
    ds_type = None
    proj_tit = proj.project_title
    conf["dataset_params"]["cache_rate"] = 0.0

    D = DataManagerDualBTfms(
        project_title=proj_tit,
        configs=conf,
        batch_size=batch_size,
        ds_type=ds_type,
    )

# %%
    D.prepare_data()
    D.setup("fit")
    tmv = D.valid_manager
    tmt = D.train_manager
    tmv.transforms_dict
# %%
# %%
    dl = tmt.dl
    iteri = iter(dl)
    for x, batch in enumerate(iteri):
        batch = next(iteri)
        print(batch["image"].shape)
#


