# %%
import ast
import math
from functools import reduce
from operator import add
from pathlib import Path

import ipdb
import numpy as np
import torch
from fastcore.basics import listify, store_attr, warnings
from lightning.pytorch import LightningDataModule
from monai.data import DataLoader, Dataset
from monai.data.dataset import CacheDataset, LMDBDataset, PersistentDataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (RandCropByPosNegLabeld,
                                                 ResizeWithPadOrCropd)
from monai.transforms.intensity.dictionary import (RandAdjustContrastd,
                                                   RandScaleIntensityd,
                                                   RandShiftIntensityd)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd
from monai.transforms.utility.dictionary import (EnsureChannelFirstd,
                                                 FgBgToIndicesd, ToDeviced)

from fran.data.dataloader import img_lm_bbox_collated
from fran.data.dataset import (ImageMaskBBoxDatasetd, MaskLabelRemapd,
                               NormaliseClipd)
from fran.transforms.imageio import LoadTorchd, TorchReader
from fran.transforms.intensitytransforms import RandRandGaussianNoised
from fran.transforms.misc_transforms import LoadDict, MetaToDict
from fran.utils.config_parsers import is_excel_None
from fran.utils.fileio import load_dict
from fran.utils.helpers import find_matching_fn, folder_name_from_list
from fran.utils.imageviewers import ImageMaskViewer
from fran.utils.string import strip_extension

tr = ipdb.set_trace


def int_to_ratios(n_fg_labels, fgbg_ratio=3):
    ratios = [1] + [fgbg_ratio / n_fg_labels] * n_fg_labels
    return ratios


def list_to_fgbg(class_ratios):
    bg = class_ratios[0]
    fg = class_ratios[1:]
    fg = reduce(add, fg)
    return fg, bg


def simple_collated(batch):
    imgs = []
    labels = []
    for i, item in enumerate(batch):
        for ita in item:
            imgs.append(ita["image"])
            labels.append(ita["lm"])
    output = {"image": torch.stack(imgs, 0), "lm": torch.stack(labels, 0)}
    return output


class DataManager(LightningDataModule):
    def __init__(
        self,
        project,
        dataset_params: dict,
        config: dict,
        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
        cache_rate=0.0,
        ds_type=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.plan = config['plan']
        store_attr(but="transform_factors")
        global_properties = load_dict(project.global_properties_filename)
        self.dataset_params["intensity_clip_range"] = global_properties[
            "intensity_clip_range"
        ]
        self.dataset_params["mean_fg"] = global_properties["mean_fg"]
        self.dataset_params["std_fg"] = global_properties["std_fg"]
        self.batch_size = batch_size
        self.cache_rate = cache_rate
        self.ds_type=ds_type
        self.assimilate_tfm_factors(transform_factors)

    def assimilate_tfm_factors(self, transform_factors):
        for key, value in transform_factors.items():
            dici = {"value": value[0], "prob": value[1]}
            setattr(self, key, dici)

    def create_transforms(self):
        E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )
        P = MaskLabelRemapd(
            keys=["lm"], src_dest_labels=self.dataset_params["src_dest_labels"]
        )

        F1 = RandFlipd(
            keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=0, lazy=True
        )
        F2 = RandFlipd(
            keys=["image", "lm"], prob=self.flip["prob"], spatial_axis=1, lazy=True
        )
        int_augs = [
            RandScaleIntensityd(
                keys="image", factors=self.scale["value"], prob=self.scale["prob"]
            ),
            RandRandGaussianNoised(
                keys=["image"], std_limits=self.noise["value"], prob=self.noise["prob"]
            ),
            # RandGaussianNoised(
            #     keys=["image"], std=self.noise["value"], prob=self.noise["prob"]
            # ),
            RandShiftIntensityd(
                keys="image", offsets=self.shift["value"], prob=self.shift["prob"]
            ),
            RandAdjustContrastd(
                ["image"], gamma=self.contrast["value"], prob=self.contrast["prob"]
            ),
            self.create_affine_tfm(),
        ]

        A = self.create_affine_tfm()
        Re = ResizeWithPadOrCropd(
            keys=["image", "lm"],
            spatial_size=self.dataset_params["patch_size"],
            lazy=True,
        )
        self.transforms_dict = {
            "A": A,
            "E": E,
            "N": N,
            "F1": F1,
            "F2": F2,
            "I": int_augs,
            "Re": Re,
            "P": P,
        }

    def set_transforms(self, keys_tr: str, keys_val: str):
        self.tfms_train = self.tfms_from_dict(keys_tr)
        self.tfms_valid = self.tfms_from_dict(keys_val)

    def tfms_from_dict(self, keys: str):
        keys = keys.split(",")
        tfms = []
        for key in keys:
            tfm = self.transforms_dict[key]
            if key == "I":
                tfms.extend(tfm)
            else:
                tfms.append(tfm)
        tfms = Compose(tfms)
        return tfms

    def prepare_data(self):
        # getting the right folders
        dataset_mode = self.plan["mode"]
        assert dataset_mode in [
            "whole",
            "patch",
            "source",
            "lbd",
        ], "Set a value for mode in 'whole', 'patch' or 'source' "
        self.train_nii, self.valid_nii = self.project.get_train_val_files(
            self.dataset_params["fold"]
        )
        self.data_folder = self.derive_data_folder()

    def create_data_dicts(self, fnames):
        fnames = [strip_extension(fn) for fn in fnames]
        fnames = [fn + ".pt" for fn in fnames]
        fnames = fnames
        images_fldr = self.data_folder / ("images")
        lms_fldr = self.data_folder / ("lms")
        inds_fldr = self.infer_inds_fldr(self.plan)
        images = list(images_fldr.glob("*.pt"))
        data = []
        for fn in fnames:
            fn = Path(fn)
            img_fn = find_matching_fn(fn.name, images,True)
            lm_fn =  find_matching_fn(fn.name, lms_fldr, True)
            indices_fn = inds_fldr / img_fn.name
            assert img_fn.exists(), "Missing image {}".format(img_fn)
            assert lm_fn.exists(), "Missing labelmap fn {}".format(lm_fn)
            dici = {"image": img_fn, "lm": lm_fn, "indices":indices_fn}
            data.append(dici)
        return data

    def infer_inds_fldr(self, plan):
        fg_indices_exclude = plan["fg_indices_exclude"]
        if is_excel_None(fg_indices_exclude):
            fg_indices_exclude=None
            indices_subfolder = "indices"
        else:
            if isinstance(fg_indices_exclude, str):
                fg_indices_exclude = ast.literal_eval(fg_indices_exclude)
            fg_indices_exclude = listify(fg_indices_exclude)
            indices_subfolder = "indices_fg_exclude_{}".format(
                "".join([str(x) for x in fg_indices_exclude])
            )
        return self.data_folder / (indices_subfolder)

    def derive_data_folder(self):
        raise NotImplementedError

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.batch_size * 2,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )
        return train_dl

    def val_dataloader(self):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.batch_size * 2,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )
        return valid_dl

    def create_affine_tfm(self):
        affine = RandAffined(
            keys=["image", "lm"],
            mode=["bilinear", "nearest"],
            prob=self.affine3d["p"],
            # spatial_size=self.dataset_params['src_dims'],
            rotate_range=self.affine3d["rotate_range"],
            scale_range=self.affine3d["scale_range"],
        )
        return affine

    def forward(self, inputs, target):
        return self.model(inputs)

    def setup(self, stage: str) -> None:
        raise NotImplementedError

    @property
    def src_dims(self):
        if self.dataset_params["zoom"] == True:
            src_dims = self.dataset_params["src_dims"]
        else:
            src_dims = self.dataset_params["patch_size"]
        return src_dims


class DataManagerSource(DataManager):
    def __init__(
        self,
        project,
        dataset_params: dict,
        config: dict,
        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
        **kwargs
    ):
        super().__init__(
            project,
            dataset_params,
            config,
            transform_factors,
            affine3d,
            batch_size,
            **kwargs
        )
        self.collate_fn = simple_collated

    def derive_data_folder(self):
        prefix = "spc"
        spacing = ast.literal_eval(self.plan["spacing"])
        parent_folder = self.project.fixed_spacing_folder
        data_folder = folder_name_from_list(prefix, parent_folder, spacing)
        return data_folder

    def prepare_data(self):
        super().prepare_data()
        self.data_train = self.create_data_dicts(self.train_nii)
        self.data_valid = self.create_data_dicts(self.valid_nii)

    def create_transforms(self):
        super().create_transforms()
        fgbg_ratio = self.dataset_params["fgbg_ratio"]
        if isinstance(fgbg_ratio, list):
            fg, bg = list_to_fgbg(fgbg_ratio)
        else:
            fg = fgbg_ratio
            bg = 1

        L = LoadImaged(
            keys=["image", "lm" ],
            image_only=True,
            ensure_channel_first=False,
            simple_keys=True,
        )
        L.register(TorchReader())
        Ld= LoadDict(keys= ["indices"],select_keys =  ["lm_fg_indices","lm_bg_indices"])
        Ind = MetaToDict(keys=["lm"], meta_keys=["lm_fg_indices", "lm_bg_indices"])
        Rtr = RandCropByPosNegLabeld(
            keys=["image", "lm"],
            label_key="lm",
            image_key="image",
            fg_indices_key="lm_fg_indices",
            bg_indices_key="lm_bg_indices",
            image_threshold=-2600,
            spatial_size=self.src_dims,
            pos=fg,
            neg=bg,
            num_samples=self.dataset_params["samples_per_file"],
            lazy=True,
            allow_smaller=True,
        )
        Rva = RandCropByPosNegLabeld(
            keys=["image", "lm"],
            label_key="lm",
            image_key="image",
            image_threshold=-2600,
            fg_indices_key="lm_fg_indices",
            bg_indices_key="lm_bg_indices",
            spatial_size=self.dataset_params["patch_size"],
            pos=1,
            neg=1,
            num_samples=self.dataset_params["samples_per_file"],
            lazy=True,
            allow_smaller=True,
        )
        self.transforms_dict.update(
            {
                "Ld":Ld,
                "L": L,
                "Ind": Ind,
                "Rtr": Rtr,
                "Rva": Rva,
            }
        )

    def setup(self, stage: str = None):
        self.create_transforms()
        self.set_transforms(
            keys_tr="L,Ld,E,Rtr,F1,F2,A,Re,N,I", keys_val="L,Ld,E,Rva,Re,N"
        )
        print("Setting up datasets. Training ds type is: ",self.ds_type)
        if is_excel_None(self.ds_type):
            self.train_ds = Dataset(data=self.data_train, transform=self.tfms_train)
        elif self.ds_type == "cache":
            self.train_ds = CacheDataset(
                data=self.data_train,
                transform=self.tfms_train,
                cache_rate=self.cache_rate,
            )
        elif self.ds_type == "lmdb":
            self.train_ds = LMDBDataset(
                data=self.data_train,
                transform=self.tfms_train,
                cache_dir=self.project.cache_folder,
                db_name="training_cache"
            )
        self.valid_ds = PersistentDataset(
            data=self.data_valid,
            transform=self.tfms_valid,
            cache_dir=self.project.cache_folder,
        )


class DataManagerLBD(DataManagerSource):
    def derive_data_folder(self, dataset_mode=None):
        spacing = ast.literal_eval(self.plan["spacing"])
        parent_folder = self.project.lbd_folder
        folder_suffix = "plan"+str(self.dataset_params["plan"])
        data_folder = folder_name_from_list(
            prefix="spc", parent_folder=parent_folder, values_list=spacing,suffix=folder_suffix
        )
        assert data_folder.exists(), "Dataset folder {} does not exists".format(
            data_folder
        )
        return data_folder

    def prepare_data(self):
        super().prepare_data()
        self.data_train = self.create_data_dicts(self.train_nii)
        self.data_valid = self.create_data_dicts(self.valid_nii)


class DataManagerShort(DataManager):
    def prepare_data(self):
        super().prepare_data()
        self.train_list = self.train_list[:32]

    def train_dataloader(self, num_workers=4, **kwargs):
        return super().train_dataloader(num_workers, **kwargs)


class DataManagerPatchLegacy(DataManager):
    '''
    Uses bboxes to randonly select fg bg labels. New version(below) uses monai fgbgindices instead
    '''
    
    def __init__(
        self,
        project,
        dataset_params: dict,
        config: dict,
        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
        **kwargs
    ):
        super().__init__(
            project,
            dataset_params,
            config,
            transform_factors,
            affine3d,
            batch_size,
            **kwargs
        )
        self.collate_fn = img_lm_bbox_collated

    def derive_data_folder(self):
        parent_folder = self.project.patches_folder
        plan_name = "plan"+str(self.dataset_params["plan"])
        source_plan_name = self.plan['source_plan']
        source_plan = self.config[source_plan_name]
        spacing = ast.literal_eval(source_plan['spacing'])
        # spacing = self.dataset_params["spacing"]
        subfldr1 = folder_name_from_list("spc", parent_folder, spacing)
        src_dims = self.src_dims
        subfldr2 = folder_name_from_list("dim", subfldr1, src_dims,plan_name)
        return subfldr2

    def setup(self, stage: str = None):
        self.create_transforms()
        if not math.isnan(self.dataset_params["src_dest_labels"]):
            keys_tr = "P,E,F1,F2,A,Re,N,I"
        else:
            keys_tr = "E,F1,F2,A,Re,N,I"
        keys_val = "E,Re,N"
        self.set_transforms(keys_tr=keys_tr, keys_val=keys_val)
        fgbg_ratio = self.dataset_params["fgbg_ratio"]
        if isinstance(fgbg_ratio, int):
            n_fg_labels = len(self.project.global_properties["labels_all"])
            class_ratios = int_to_ratios(n_fg_labels=n_fg_labels, fgbg_ratio=fgbg_ratio)
        else:
            class_ratios = fgbg_ratio

        bboxes_fname = self.data_folder / "bboxes_info"
        self.train_ds = ImageMaskBBoxDatasetd(
            self.train_nii,
            bboxes_fname,
            class_ratios,
            transform=self.tfms_train,
        )
        self.valid_ds = ImageMaskBBoxDatasetd(
            self.valid_nii, bboxes_fname, transform=self.tfms_valid
        )

    @property
    def src_dims(self):
        return ast.literal_eval(self.plan["patch_size"])

class DataManagerPatch(DataManager):
    def __init__(
        self,
        project,
        dataset_params: dict,
        config: dict,
        transform_factors: dict,
        affine3d: dict,
        batch_size=8,
        **kwargs
    ):
        super().__init__(
            project,
            dataset_params,
            config,
            transform_factors,
            affine3d,
            batch_size,
            **kwargs
        )
        self.collate_fn = img_lm_bbox_collated

    def prepare_data(self):
        # getting the right folders
        dataset_mode = self.plan["mode"]
        self.train_cids, self.valid_cids = self.project.get_train_val_cids(
            self.dataset_params["fold"]
        )
        self.data_folder = self.derive_data_folder()

    def derive_data_folder(self):
        parent_folder = self.project.patches_folder
        plan_name = "plan"+str(self.dataset_params["plan"])
        source_plan_name = self.plan['source_plan']
        source_plan = self.config[source_plan_name]
        spacing = ast.literal_eval(source_plan['spacing'])
        # spacing = self.dataset_params["spacing"]
        subfldr1 = folder_name_from_list("spc", parent_folder, spacing)
        src_dims = self.src_dims
        subfldr2 = folder_name_from_list("dim", subfldr1, src_dims,plan_name)
        return subfldr2

    def setup(self, stage: str = None):
        self.create_transforms()
        if not math.isnan(self.dataset_params["src_dest_labels"]):
            keys_tr = "P,E,F1,F2,A,Re,N,I"
        else:
            keys_tr = "E,F1,F2,A,Re,N,I"
        keys_val = "E,Re,N"
        self.set_transforms(keys_tr=keys_tr, keys_val=keys_val)
        fgbg_ratio = self.dataset_params["fgbg_ratio"]
        if isinstance(fgbg_ratio, int):
            n_fg_labels = len(self.project.global_properties["labels_all"])
            class_ratios = int_to_ratios(n_fg_labels=n_fg_labels, fgbg_ratio=fgbg_ratio)
        else:
            class_ratios = fgbg_ratio

        bboxes_fname = self.data_folder / "bboxes_info"
        self.train_ds = ImageMaskBBoxDatasetd(
            self.train_nii,
            bboxes_fname,
            class_ratios,
            transform=self.tfms_train,
        )
        self.valid_ds = ImageMaskBBoxDatasetd(
            self.valid_nii, bboxes_fname, transform=self.tfms_valid
        )

    @property
    def src_dims(self):
        return ast.literal_eval(self.plan["patch_size"])


class DataManagerShort(DataManagerPatch):
    def prepare_data(self):
        super().prepare_data()
        self.train_list = self.train_list[:32]

    def train_dataloader(self, num_workers=4, **kwargs):
        return super().train_dataloader(num_workers, **kwargs)


# %%
if __name__ == "__main__":

    import torch

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *

<<<<<<< HEAD
    project_title = "nodes"
=======
    project_title = "litsmc"
>>>>>>> efc2e4fb (jj)
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    )
    configuration_filename = None

    config = ConfigMaker(
        proj, raytune=False, configuration_filename=configuration_filename
    ).config

    global_props = load_dict(proj.global_properties_filename)
# %%
# %%
#SECTION:-------------------- DataManagerSource ------------------------------------------------------------------------------------------------------

    batch_size = 2
    D = DataManagerSource(
        proj,
        config = config,
        dataset_params=config["dataset_params"],
        transform_factors=config["transform_factors"],
        affine3d=config["affine3d"],
        batch_size=batch_size,
        plan = configs["plan2"]
    )
# %%
    D.prepare_data()

    D.setup()
<<<<<<< HEAD
    D.dataset_folder
=======
    D.data_folder
>>>>>>> efc2e4fb (jj)
# %%

# %%
#SECTION:-------------------- Patch--------------------------------------------------------------------------------------


    batch_size = 2
    D = DataManagerPatch(
        proj,
        config = config,
        dataset_params=config["dataset_params"],
        transform_factors=config["transform_factors"],
        affine3d=config["affine3d"],
        batch_size=batch_size,
    )

# %%

    D.prepare_data()

    D.setup()
    D.data_folder
# %%

    E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
    Rtr = RandCropByPosNegLabeld(
        keys=["image", "lm"],
        label_key="lm",
        image_key="image",
        fg_indices_key="lm_fg_indices",
        bg_indices_key="lm_bg_indices",
        image_threshold=-2600,
        spatial_size=D.src_dims,
        pos=1,
        neg=1,
        num_samples=D.dataset_params["samples_per_file"],
        lazy=True,
        allow_smaller=False,
    )
    Ld= LoadDict(keys= ["indices"],select_keys =  ["lm_fg_indices","lm_bg_indices"])

    Rva = RandCropByPosNegLabeld(
            keys=["image", "lm"],
            label_key="lm",
            image_key="image",
            image_threshold=-2600,
            fg_indices_key="lm_fg_indices",
            bg_indices_key="lm_bg_indices",
            spatial_size=D.dataset_params["patch_size"],
            pos=1,
            neg=1,
            num_samples=D.dataset_params["samples_per_file"],
            lazy=True,
            allow_smaller=True,
        )
    Re = ResizeWithPadOrCropd(
            keys=["image", "lm"],
            spatial_size=D.dataset_params["patch_size"],
            lazy=True,
        )

    L = LoadTorchd(keys=["image", "lm"])
# %%
    D.prepare_data()
    D.setup(None)
# %%
    D.valid_ds[7]

    keys_val="L,Ld,E,Rva,Re,N"
    dici = D.valid_ds.data[7]
    dici = L(dici)
    dici =Ld(dici)
    dici = E(dici)
    dici = Rva(dici)
    dici = Re(dici)

# %%
    dl = D.train_dataloader()
    iteri = iter(dl)
    aa = D.Train_ds[0]
    b = next(iteri)
    print(b["image"].shape)
# %%
    im_fn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/nodes/spc_080_080_300/images/nodes_20_20190611_Neck1p0I30f3_thick.pt"
    label_fn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/nodes/spc_080_080_300/masks/nodes_20_20190611_Neck1p0I30f3_thick.pt"
    dici = {"image": im_fn, "lm": label_fn}
    D.setup()
# %%
    ind = 1
    img = b["image"][ind][0]
    lab = b["lm"][ind][0]
    ImageMaskViewer([img, lab])
# %


# %%
