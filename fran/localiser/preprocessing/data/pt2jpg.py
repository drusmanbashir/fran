# %%
from pathlib import Path

import ipdb
import lightning as L
import torch
from fran.transforms.imageio import LoadTorchd
from fran.transforms.intensitytransforms import MakeBinary
from fran.transforms.misc_transforms import BoundingBoxYOLOd
from monai.apps.detection.transforms.dictionary import ConvertBoxToStandardModed
from monai.data.dataset import Dataset
from monai.transforms import Compose
from monai.transforms.croppad.dictionary import BoundingRectd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.spatial.dictionary import RandFlipd, RandRotated, RandZoomd, Resized
from monai.transforms.utility.dictionary import (
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
)
from torch.utils.data import random_split
from torchvision.utils import save_image
from tqdm.auto import tqdm

tr = ipdb.set_trace


def write_list_to_txt(data_list, filepath, delimiter=" "):
    with open(filepath, "w") as f:
        if isinstance(data_list[0], (list, tuple)):
            for item in data_list:
                line = delimiter.join(str(x) for x in item)
                f.write(line + "\n")
        else:
            line = delimiter.join(str(x) for x in data_list)
            f.write(line + "\n")


class DetectDS(Dataset):
    def __init__(self, fldr):
        fldr = Path(fldr)
        self.fldr_imgs = fldr / "images"
        self.fldr_lms = fldr / "lms"
        imgs = {img.name: img for img in self.fldr_imgs.glob("*")}
        lms = {lm.name: lm for lm in self.fldr_lms.glob("*")}
        self.data_dicts = []
        for name in sorted(imgs.keys() & lms.keys()):
            dici = {"image": imgs[name], "lm": lms[name]}
            self.data_dicts.append(dici)

    def __len__(self) -> int:
        return len(self.data_dicts)


class PreprocessorPT2JPG(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.fldr_imgs = self.data_dir / "images"
        self.fldr_lms = self.data_dir / "lms"

    def prepare_data(self):
        imgs = list(self.fldr_imgs.glob("*"))
        tot = len(imgs)
        train_len = int(0.8 * tot)
        val_len = tot - train_len
        imgs_train, imgs_val = random_split(
            imgs, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )
        self.train_dicts = self.create_data_dicts(imgs_train)
        self.val_dicts = self.create_data_dicts(imgs_val)

    def create_data_dicts(self, imgs):
        lms = {lm.name: lm for lm in self.fldr_lms.glob("*")}
        data_dicts = []
        for img in imgs:
            if img.name in lms:
                dici = {"image": img, "lm": lms[img.name]}
                data_dicts.append(dici)
        return data_dicts

    def create_transforms(self, probs=0.3):
        image_key = "image"
        label_key = "lm"
        box_key = "lm_bbox"
        outputsize = [512, 512]
        N = NormalizeIntensityd([image_key])
        L = LoadTorchd([image_key, label_key])
        E = EnsureChannelFirstd(keys=[image_key])
        MkB = MakeBinary([label_key])
        Et = EnsureTyped(keys=[image_key], dtype=torch.float32)
        Et2 = EnsureTyped(keys=[label_key], dtype=torch.long)
        E2 = EnsureTyped(keys=[image_key], dtype=torch.float16)
        ExtractBbox = BoundingRectd(keys=[label_key])
        CB = ConvertBoxToStandardModed(mode="xxyy", box_keys=[box_key])
        YoloBbox = BoundingBoxYOLOd(
            [box_key], 2, key_template_tensor=label_key, output_keys=["bbox_yolo"]
        )
        Rotate = RandRotated(
            keys=[image_key, label_key],
            prob=probs,
            keep_size=True,
            mode=["bilinear", "nearest"],
            range_x=[0.4, 0.4],
            lazy=True,
        )
        Zoom = RandZoomd(
            keys=[image_key, label_key],
            mode=["bilinear", "nearest"],
            prob=probs,
            min_zoom=0.7,
            max_zoom=1.4,
            padding_mode="constant",
            keep_size=True,
            lazy=True,
        )
        Flip1 = RandFlipd(
            keys=[image_key, label_key], prob=probs, spatial_axis=0, lazy=True
        )
        Flip2 = RandFlipd(
            keys=[image_key, label_key], prob=probs, spatial_axis=1, lazy=True
        )
        Resize = Resized(
            keys=[image_key, label_key],
            spatial_size=outputsize,
            mode=["bilinear", "nearest"],
            lazy=True,
        )
        DelI = DeleteItemsd(keys=[label_key])
        self.transforms_dict = {
            "N": N,
            "L": L,
            "E": E,
            "Et": Et,
            "MkB": MkB,
            "Et2": Et2,
            "E2": E2,
            "ExtractBbox": ExtractBbox,
            "YoloBbox": YoloBbox,
            "DelI": DelI,
            "CB": CB,
            "Rotate": Rotate,
            "Zoom": Zoom,
            "Flip1": Flip1,
            "Flip2": Flip2,
            "Resize": Resize,
        }

    def set_transforms(self, keys_tr: str, keys_val: str):
        self.tfms_train = self.tfms_from_dict(keys_tr)
        self.tfms_valid = self.tfms_from_dict(keys_val)

    def tfms_from_dict(self, keys: str):
        keys = keys.replace(" ", "").split(",")
        tfms = []
        for key in keys:
            tfm = self.transforms_dict[key]
            tfms.append(tfm)
        return Compose(tfms)

    def build_datasets(self):
        self.prepare_data()
        self.create_transforms()
        self.set_transforms(keys_tr=self.keys_tr, keys_val=self.keys_val)
        self.ds_train = Dataset(self.train_dicts, self.tfms_train)
        self.ds_val = Dataset(self.val_dicts, self.tfms_valid)

    def format_bbox_rows(self, dici):
        bbox = [0] + dici["bbox_yolo"].tolist()
        return bbox

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
        return [
            "names:",
            "- ROI",
            "nc: 1",
            "test: ../test/images",
            "train: ../train/images",
            "val: ../valid/images",
            "",
        ]

    def write_data_yaml(self, out_dir):
        with open(out_dir / "data.yaml", "w") as f:
            f.write("\n".join(self.data_yaml_lines()))

    def export_yolo_dataset(self, out_dir):
        out_dir = Path(out_dir)
        self.build_datasets()
        self.export_split(self.ds_train, out_dir / "train")
        self.export_split(self.ds_val, out_dir / "valid")
        self.write_data_yaml(out_dir)


class DetectDataModule(PreprocessorPT2JPG):
    keys_tr = "L,MkB,Et,Et2,N,Flip1,Flip2,Zoom,Resize,ExtractBbox,CB,YoloBbox"
    keys_val = "L,MkB,Et,Et2,N,Resize,ExtractBbox,CB,YoloBbox,DelI"
