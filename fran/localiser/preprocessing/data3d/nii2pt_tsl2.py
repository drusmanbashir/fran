# %%
from pathlib import Path

import ipdb
import numpy as np
import pandas as pd
import ray
import torch
from fran.localiser.transforms import (
    MultiRemapsTSL,
    NormaliseZeroToOne,
    TSLRegions,
    WindowTensor3Channeld,
    tfms_from_dict,
)
from fran.localiser.transforms.tsl import MultiRemapsTSLMonai
from fran.transforms.imageio import LoadSITKd
from fran.transforms.spatialtransforms import Project2D

from monai.transforms.spatial.dictionary import Orientationd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToDeviced
from utilz.fileio import maybe_makedirs
from utilz.helpers import create_df_from_folder
from utilz.imageviewers import ImageMaskViewer
from utilz.stringz import strip_extension


import lightning as L
import numpy as np
import pandas as pd
import ray
import torch
from fran.localiser.preprocessing.data.nii2pt import _PreprocessorNII2PTWorkerBase
from fran.localiser.preprocessing.data.pt2jpg import write_list_to_txt
from fran.transforms.imageio import LoadTorchd
from fran.transforms.intensitytransforms import MakeBinary
from fran.transforms.misc_transforms import BoundingBoxesYOLOd
from monai.apps.detection.transforms.dictionary import ConvertBoxToStandardModed
from monai.data.dataset import Dataset
from monai.transforms.croppad.dictionary import BoundingRectd
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.utility.dictionary import (
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
)
from torch.utils.data import random_split
from torchvision.utils import save_image
from tqdm.auto import tqdm
from utilz.fileio import maybe_makedirs
from utilz.helpers import create_df_from_folder
from utilz.stringz import strip_extension


class DynamicResized(Resized):
    def __init__(self, keys, mode,min_size, lazy=False):
        assert len(min_size) == 3, "min_size should be a tuple of (min_height, min_width, min_depth)"
        self.min_size = min_size
        super().__init__(keys=keys, spatial_size=None, mode=mode, lazy=lazy)

    def _get_spatial_size(self, img):
        if img.ndim == 4:
            shpe = img.shape[1:]  # (C, H, W, D)
        else: raise ValueError(f"Unsupported image dimensions: {img.ndim}. Expected 4")
        return shpe

    def __call__(self, data):
        img = data[self.keys[0]]
        spatial_size = self._get_spatial_size(img)
        new_size = [min(s, m) for s, m in zip(spatial_size, self.min_size)]
        self.resizer.spatial_size = new_size
        return super().__call__(data)

class _PreprocessorNII2JPGWorkerBase:
    def __init__(self, output_folder,output_size, device="cpu", debug=False):
        self.output_folder = Path(output_folder)
        self.device = device
        self.debug = debug
        self.image_key = "image"
        self.label_key = "lm"
        self.lm_key = self.label_key
        self.tfms_keys = self.worker_tfms_keys()
        self.create_transforms(device=device)
        self.transforms = tfms_from_dict(self.tfms_keys, self.transforms_dict)
        self.min_output_size = output_size


    def worker_tfms_keys(self):
        return "L,E,O,DynRes,Remap"

    def create_transforms(self, device="cpu"):
        box_key = "lm_bbox"
        self.L = LoadSITKd(keys=[self.image_key, self.lm_key])
        self.E = EnsureChannelFirstd(
            keys=[self.image_key, self.lm_key], channel_dim="no_channel"
        )
        self.O = Orientationd(keys=[self.image_key, self.lm_key], axcodes="RAS")
        self.N = NormaliseZeroToOne(keys=[self.image_key])
        self.Dev = ToDeviced(keys=[self.image_key, self.lm_key], device=device)
        self.P1 = Project2D(
            keys=[self.lm_key, self.image_key],
            operations=["mean", "sum"],
            dim=1,
            output_keys=["lm1", "image1"],
        )
        self.P2 = Project2D(
            keys=[self.lm_key, self.image_key],
            operations=["sum", "mean"],
            dim=2,
            output_keys=["lm2", "image2"],
        )
        self.P3 = Project2D(
            keys=[self.lm_key, self.image_key],
            operations=["sum", "mean"],
            dim=3,
            output_keys=["lm3", "image3"],
        )
        self.Win = WindowTensor3Channeld(image_key=self.image_key)
        # Axial XY projection is kept for reference but not used:
        # it is a top-down/bottom-up view after RAS orientation and is not informative for YOLO.
        self.Remap = MultiRemapsTSL(lm_key=self.lm_key)
        self.Ld = LoadTorchd([self.image_key, self.label_key])
        self.ToBinary = MakeBinary([self.label_key])
        self.Et = EnsureTyped(keys=[self.image_key], dtype=torch.float32)
        self.Et2 = EnsureTyped(keys=[self.label_key], dtype=torch.long)
        self.E2 = EnsureTyped(keys=[self.image_key], dtype=torch.float16)
        self.ExtractBbox = BoundingRectd(keys=[self.label_key])
        self.DynResize = DynamicResized(keys=[self.image_key, self.label_key], mode=["trilinear", "nearest"], min_size=self.min_output_size)
        self.CB = ConvertBoxToStandardModed(mode="xxyy", box_keys=[box_key])
        self.YoloBboxes = BoundingBoxesYOLOd(
            [box_key], 2, key_template_tensor=self.label_key, output_keys=["bbox_yolo"]
        )
        self.Resize = Resized(
            keys=[self.image_key, self.label_key],
            spatial_size=self.min_output_size,
            mode=["bilinear", "nearest"],
            lazy=True,
        )
        self.DelI = DeleteItemsd(keys=[self.label_key])

        self.transforms_dict = {
            "L": self.L,
            "Ld": self.Ld,
            "DynRes": self.DynResize,
            "E": self.E,
            "Et": self.Et,
            "ToBinary": self.ToBinary,
            "Et2": self.Et2,
            "E2": self.E2,
            "ExtractBbox": self.ExtractBbox,
            "YoloBboxes": self.YoloBboxes,
            "DelI": self.DelI,
            "CB": self.CB,
            "Resize": self.Resize,
            "O": self.O,
            "N": self.N,
            "Dev": self.Dev,
            "P1": self.P1,
            "P2": self.P2,
            "P3": self.P3,
            "Win": self.Win,
            "Remap": self.Remap,
        }

    def save_pt(self, tnsr, subfolder, suffix):
        fn = Path(tnsr.meta["filename_or_obj"])
        fn_name = strip_extension(fn.name) + "_" + str(suffix) + ".pt"
        out_fn = self.output_folder / subfolder / fn_name
        torch.save(tnsr.contiguous(), out_fn)

    def image_suffixes(self):
        if "Win" in self.tfms_keys:
            suffixes = []
            for window in self.Win.windows.keys():
                for projection in [1, 2]:
                    suffixes.append(f"{window}{projection}")
            return suffixes
        return [1, 2]

    def _process_row(self, row):
        dici = {"image": row["image"], "lm": row["lm"]}
        dici = self.transforms(dici)
        for projection in [1, 2]:
            image = dici["image" + str(projection)]
            lm = dici["lm" + str(projection)]
            if "Win" in self.tfms_keys:
                for window_ind, window in enumerate(self.Win.windows.keys()):
                    suffix = f"{window}{projection}"
                    self.save_pt(image[[window_ind]], "images", suffix)
                    self.save_pt(lm, "lms", suffix)
            else:
                self.save_pt(image, "images", projection)
                self.save_pt(lm, "lms", projection)
        return {"case_id": row["case_id"], "ok": True}

    def process(self, df):
        outputs = []
        for ind in range(len(df)):
            row = df.iloc[ind]
            outputs.append(self._process_row(row))
        return outputs

@ray.remote(num_cpus=1)
class TSLWorkere2e(_PreprocessorNII2JPGWorkerBase):
    pass


class TSLWorkerLocale2e(_PreprocessorNII2JPGWorkerBase):
    pass


class PreprocessorNII2JPG_TSL(L.LightningDataModule):
    keys_tr = "L,ToBinary,Et,Et2,N,Resize,ExtractBbox,CB,YoloBboxes,DelI"
    keys_val = "L,ToBinary,Et,Et2,N,Resize,ExtractBbox,CB,YoloBboxes,DelI"

    def __init__(self, data_folder, output_folder, batch_size: int = 4):
        super().__init__()
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.fldr_imgs = self.data_folder / "images"
        self.fldr_lms = self.data_folder / "lms"
        self.out_pt = self.output_folder / "pt"
        self.output_fldr_imgs = self.out_pt / "images"
        self.output_fldr_lms = self.out_pt / "lms"
        self.out_jpg = self.output_folder / "jpg"
        self.actor_cls = TSLWorkere2e
        self.local_worker_cls = TSLWorkerLocale2e
        self.batch_size = batch_size
        self.data_dir = self.out_pt
        self.image_key = "image"
        self.label_key = "lm"
        self.tsl_regions = TSLRegions()
        self.data_yaml = self.tsl_regions.data_yaml

    def _df_from_folder(self):
        return create_df_from_folder(self.data_folder)

    def create_data_df(self):
        self.df = self._df_from_folder()
        assert len(self.df) > 0, "No valid case files found in {}".format(
            self.data_folder
        )
        self.case_ids = self.df["case_id"].tolist()
        print("Total number of cases: ", len(self.df))

    def create_output_folders(self):
        maybe_makedirs([self.output_fldr_imgs, self.output_fldr_lms])

    def register_existing_files(self):
        existing = []
        for suffix in self.local_worker_cls(self.out_pt).image_suffixes():
            suffix = str(suffix)
            imgs = {p.name for p in self.output_fldr_imgs.glob("*_" + suffix + ".pt")}
            lms = {p.name for p in self.output_fldr_lms.glob("*_" + suffix + ".pt")}
            existing.append(imgs.intersection(lms))
        self.existing_output_fnames = set.intersection(*existing) if existing else set()
        print("Output folder: ", self.out_pt)
        print(
            "Image files fully processed in a previous session: ",
            len(self.existing_output_fnames),
        )

    def remove_completed_cases(self):
        if not getattr(self, "existing_output_fnames", None):
            return
        n_before = len(self.df)
        suffix = self.local_worker_cls(self.out_pt).image_suffixes()[0]
        keep_mask = self.df["image"].apply(
            lambda x: (
                strip_extension(Path(x).name) + "_" + str(suffix) + ".pt"
                not in self.existing_output_fnames
            )
        )
        self.df = self.df[keep_mask]
        print("Image files remaining to process:", len(self.df), "/", n_before)

    def should_use_ray(self):
        debug = getattr(self, "debug", False)
        return (self.num_processes > 1) and (debug == False)

    def build_worker_kwargs(self, device, debug):
        return {
            "output_folder": self.out_pt,
            "device": device,
            "debug": debug,
        }

    def setup(self, overwrite=False, num_processes=8, device="cpu", debug=False):
        self.create_output_folders()
        self.num_processes = max(1, int(num_processes))
        self.debug = debug
        self.create_data_df()
        self.register_existing_files()
        print("Overwrite:", overwrite)
        if overwrite == False:
            self.remove_completed_cases()
        self.use_ray = self.should_use_ray()
        worker_kwargs = self.build_worker_kwargs(device=device, debug=debug)
        if self.use_ray:
            n = min(len(self.df), self.num_processes)
            self.mini_dfs = np.array_split(self.df, n)
            self.actors = [self.actor_cls.remote(**worker_kwargs) for _ in range(n)]
        else:
            self.mini_dfs = [self.df]
            self.local_worker = self.local_worker_cls(**worker_kwargs)

    def prepare_data(self):
        imgs = list(self.output_fldr_imgs.glob("*"))
        tot = len(imgs)
        train_len = int(0.8 * tot)
        val_len = tot - train_len
        imgs_train, imgs_val = random_split(
            imgs, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )
        self.train_dicts = self.create_data_dicts(imgs_train)
        self.val_dicts = self.create_data_dicts(imgs_val)

    def create_data_dicts(self, imgs):
        lms = {lm.name: lm for lm in self.output_fldr_lms.glob("*")}
        data_dicts = []
        for img in imgs:
            if img.name in lms:
                dici = {"image": img, "lm": lms[img.name]}
                data_dicts.append(dici)
        return data_dicts

    def create_transforms(self, probs=0.3):
        box_key = "lm_bbox"
        self.DelI = DeleteItemsd(keys=[self.label_key])
        self.transforms_dict = {
            "N": self.N,
            "L": self.Ld,
            "E": self.E,
            "Et": self.Et,
            "ToBinary": self.ToBinary,
            "Et2": self.Et2,
            "E2": self.E2,
            "ExtractBbox": self.ExtractBbox,
            "YoloBboxes": self.YoloBboxes,
            "DelI": self.DelI,
            "CB": self.CB,
            "Resize": self.Resize,
        }

    def set_transforms(self, keys_tr: str, keys_val: str):
        self.tfms_train = tfms_from_dict(keys_tr, self.transforms_dict)
        self.tfms_valid = tfms_from_dict(keys_val, self.transforms_dict)

    def build_datasets(self):
        self.prepare_data()
        self.create_transforms()
        self.set_transforms(keys_tr=self.keys_tr, keys_val=self.keys_val)
        self.ds_train = Dataset(self.train_dicts, self.tfms_train)
        self.ds_val = Dataset(self.val_dicts, self.tfms_valid)

    def format_bbox_rows(self, dici):
        rows = []
        for cls_id, box in enumerate(dici["bbox_yolo"]):
            rows.append([cls_id] + box.tolist())
        return rows

    def export_split(self, ds, split_dir):
        fldr_imgs = split_dir / "images"
        fldr_labels = split_dir / "labels"
        fldr_imgs.mkdir(parents=True, exist_ok=True)
        fldr_labels.mkdir(parents=True, exist_ok=True)
        for dici in tqdm(ds):
            im = dici["image"]
            imv = im.permute(0, 2, 1)
            src_fn = Path(im.meta["filename_or_obj"])
            nm_jpg = src_fn.name.replace("pt", "jpg")
            nm_txt = src_fn.name.replace("pt", "txt")
            rows = self.format_bbox_rows(dici)
            save_image(imv, fldr_imgs / nm_jpg)
            write_list_to_txt(rows, fldr_labels / nm_txt)

    def data_yaml_lines(self):
        return self.tsl_regions.data_yaml_lines()

    def write_data_yaml(self, out_dir):
        with open(out_dir / "data.yaml", "w") as f:
            f.write(self.data_yaml)

    def export_yolo_dataset(self, out_dir):
        out_dir = Path(out_dir)
        self.build_datasets()
        self.export_split(self.ds_train, out_dir / "train")
        self.export_split(self.ds_val, out_dir / "valid")
        self.write_data_yaml(out_dir)

    def process(self):
        if len(self.df) == 0:
            self.results = []
            self.results_df = pd.DataFrame([])
        elif self.use_ray:
            results = ray.get(
                [
                    actor.process.remote(mini_df)
                    for actor, mini_df in zip(self.actors, self.mini_dfs)
                ]
            )
            self.results = results
            self.results_df = pd.DataFrame(
                [item for sublist in self.results for item in sublist]
            )
        else:
            self.results = [self.local_worker.process(self.mini_dfs[0])]
            self.results_df = pd.DataFrame(
                [item for sublist in self.results for item in sublist]
            )
        self.export_yolo_dataset(self.out_jpg)
        return self.results_df


if __name__ == "__main__":
    from fran.data.dataregistry import DS
    from utilz.imageviewers import ImageMaskViewer

# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
    data_folder = DS["totalseg"].folder
    img_fns = sorted((Path(data_folder) / "images").glob("*.nii.gz"))
    img_fn = img_fns[0]
    lm_fn = Path(data_folder) / "lms" / img_fn.name
    output_folder = Path("/s/xnat_shadow/totalseg2d_debug")

# %%
    w = TSLWorkerLocale2e(output_folder=output_folder / "pt")
    mini_df = pd.DataFrame(
        [
            {
                "case_id": "totalseg_0001",
                "image": img_fn,
                "lm": lm_fn,
            }
        ]
    )

# %%
    def print_dici(dici, label):
        print("\n" + "=" * 80)
        print(label)
        for key, value in dici.items():
            if hasattr(value, "shape"):
                print(
                    key,
                    type(value).__name__,
                    "shape=",
                    tuple(value.shape),
                    "dtype=",
                    value.dtype,
                )
                if hasattr(value, "meta") and "filename_or_obj" in value.meta:
                    print("  filename_or_obj:", value.meta["filename_or_obj"])
            else:
                print(key, value)

# %%
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 100)

    print(mini_df)
    row = mini_df.iloc[0]
    dici = {"image": row["image"], "lm": row["lm"]}

# %%
    L1 = w.transforms_dict["L"]
    E1 = w.transforms_dict["E"]
    O1 = w.transforms_dict["O"]
    w.transforms_dict.keys
    Win1 = w.transforms_dict["Win"]
    Remap1 = w.transforms_dict["Remap"]
    Remap2 = MultiRemapsTSLMonai(lm_key=w.lm_key)
    output_size = [256,256,256]
    P1 = w.transforms_dict["P1"]
    P2 = w.transforms_dict["P2"]

# %%
    print_dici(dici, "Before NII transforms")
    dici1 = L1(dici)
    print_dici(dici1, "After L")
    dici2 = E1(dici1)
    print_dici(dici2, "After E")
    dici3 = O1(dici2)
    dici3['image'].shape
    dici3['lm'].shape
    DR = DynamicResized(keys=[w.image_key, w.lm_key], mode=["trilinear", "nearest"], min_size=output_size)
    dici4 = DR(dici3)
          
    print_dici(dici3, "After O")
    print_dici(dici4, "After DynamicResized")
    ImageMaskViewer([dici3["image"][0], dici3["lm"][0]], "im")
# %%
    dici4 = Remap1(dici3)


    dici5b = Remap2(dici4)
    print_dici(dici5, "After Remap")


    ImageMaskViewer([dici5b["image"][0], dici5b["lm"][0]], "im")
    Remap1.regions
    
# %%
    dici6 = P1(dici5)
    print_dici(dici6, "After P1")
    dici7 = P2(dici6)
    print_dici(dici7, "After P2")
    dici3["lm1"].shape

# %%
    suffix = "b1"
    w.save_pt(dici7["image1"][[0]], "images", suffix)
    w.save_pt(dici7["lm1"], "lms", suffix)

# %%
    P = PreprocessorNII2JPG_TSL(data_folder, output_folder)
    P.create_transforms()
    pt_img = (
        output_folder
        / "pt"
        / "images"
        / f"{img_fn.stem.replace('.nii', '')}_{suffix}.pt"
    )
    pt_lm = (
        output_folder / "pt" / "lms" / f"{img_fn.stem.replace('.nii', '')}_{suffix}.pt"
    )
    dici_pt = {"image": pt_img, "lm": pt_lm}

# %%
    print_dici(dici_pt, "Before PT transforms")
    dici_pt1 = L2(dici_pt)
    print_dici(dici_pt1, "After L")
    dici_pt2 = ToBinary2(dici_pt1)
    print_dici(dici_pt2, "After ToBinary")
    dici_pt3 = Et2(dici_pt2)
    print_dici(dici_pt3, "After Et")
    dici_pt4 = Et22(dici_pt3)
    print_dici(dici_pt4, "After Et2")
    dici_pt5 = N2(dici_pt4)
    print_dici(dici_pt5, "After N")
    dici_pt6 = Resize2(dici_pt5)
    print_dici(dici_pt6, "After Resize")
    dici_pt7 = ExtractBbox2(dici_pt6)
    print_dici(dici_pt7, "After ExtractBbox")
    dici_pt8 = CB2(dici_pt7)
    print_dici(dici_pt8, "After CB")
    dici_pt9 = YoloBboxes2(dici_pt8)
    print_dici(dici_pt9, "After YoloBboxes")
    dici_pt10 = DelI2(dici_pt9)
    print_dici(dici_pt10, "After DelI")

# %%
    print(P.format_bbox_rows(dici_pt9))
