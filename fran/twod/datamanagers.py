from __future__ import annotations

from typing import Any, Dict, Hashable, Mapping, Sequence

import torch
from monai.transforms.transform import MapTransform, Randomizable
from fran.managers.data.training import DataManagerDual, DataManager

import torch
from monai.transforms import MapTransform




class RandZWindowd(Randomizable, MapTransform):
    def __init__(self, keys, T, z_dim=-1):
        super().__init__(keys)
        self.T = T
        self.z_dim = z_dim

    def randomize(self, Z: int):
        self.start = int(self.R.randint(0, Z - self.T + 1))

    def __call__(self, data):
        d = dict(data)
        Z = d[self.keys[0]].shape[self.z_dim]
        self.randomize(Z)

        for k in self.keys:
            x = d[k]
            slc = [slice(None)] * x.ndim
            slc[self.z_dim] = slice(self.start, self.start + self.T)
            d[k] = x[tuple(slc)]

        d["start_z"] = self.start
        return d

class DataManager2(DataManager):
    def create_transforms(self):
        super().create_transforms()
        self.transforms_dict["Z"] = RandZWindowd

class DataManagerDual2(DataManagerDual):

    def __init__(
        self,
        project_title,
        configs: dict,
        batch_size: int,
        cache_rate=0.0,
        device="cuda",
        ds_type=None,
        save_hyperparameters=True,
        keys_tr="L,Z,Sa,Rva,Affine,F1,F2 ,Re,N,IntensityTfms",
        keys_val="L,Z,Sa, Rva,Re, N",
        # keys_tr = "L,Remap,Ld,E,N,Rtr,F1,F2,Affine,ResizePC,IntensityTfms",
        # keys_val = "L,N,Remap,Ld,E,ResizePC",
    ):
        super().__init__(
            project_title,
            configs,
            batch_size,
            cache_rate,
            device,
            ds_type,
            save_hyperparameters,
            keys_tr,
            keys_val,
        )

    def infer_manager_classes(self,configs)->tuple:
        return (DataManager2, DataManager2)
