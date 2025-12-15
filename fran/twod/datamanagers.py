from __future__ import annotations

import warnings
from typing import Optional

import torch
from monai.transforms.transform import MapTransform, Randomizable

from fran.configs.parser import ConfigMaker
from fran.managers import Project
from fran.managers.data.training import DataManagerDual, DataManagerLBD


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


class DataManager2(DataManagerLBD):
    def create_transforms(self):
        self.sequence_length=16
        super().create_transforms()
        self.transforms_dict["Z"] = RandZWindowd(keys = ["image", "lm"], T = self.sequence_length)


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
        keys_tr = "L,Z,Remap,Ld,E,N,Rtr,F1,F2,ResizePC,IntensityTfms",
        keys_val = "L,Z,N,Remap,Ld,E,ResizePC",
        data_folder: Optional[str | Path] = None,
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
            data_folder,
        )

    def infer_manager_classes(self, configs) -> tuple:
        return (DataManager2, DataManager2)


# %%
if __name__ == "__main__":

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")
    project_title = "litsmc"
    torch.set_float32_matmul_precision("medium")
    proj_litsmc = Project(project_title=project_title)

    C = ConfigMaker(proj_litsmc)
    C.setup(1)
    conf_litsmc = C.configs
# %%

    conf_litsmc["plan_train"]["patch_size"] = [256, 256]
    batch_size = 8
    ds_type = "lmdb"
    from pathlib import Path

    data_fldr = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_ex070/slices")
# %%

    D = DataManagerDual2(
        project_title=proj_litsmc.project_title,
        configs=conf_litsmc,
        batch_size=batch_size,
        ds_type=ds_type,
        data_folder=None,
    )
# %%
    D.prepare_data()
    D.configs["plan_train"]
    D.setup()
# %%
    dl =D.train_dataloader()

    iteri = iter(dl)
    batch = next(iteri)
# %%
    batch['image'].shape
# %%


    dm = D.train_manager
    dici = dm.data[0]
    dici = dm.ds[0]
# %%
    dici = dm.transforms_dict["L"](dici)
    dici = dm.transforms_dict["Z"](dici)
    dici = dm.transforms_dict["Remap"](dici)
    dici = dm.transforms_dict["Ld"](dici)
    dici=dm.transforms_dict["E"](dici)
    print(dici['image'].shape)
    dici=dm.transforms_dict["N"](dici)
    dici=dm.transforms_dict["Rtr"](dici)
    print(dici['image'].shape)
    dici=dm.transforms_dict["F1"](dici)
    dici=dm.transforms_dict["F2"](dici)
    dici=dm.transforms_dict["ResizePC"](dici)
    dici=dm.transforms_dict["IntensityTfms"](dici)

# %%

    dici['image'][0][0].shape

    aa = dici[0]

    aa.keys()
# %%
