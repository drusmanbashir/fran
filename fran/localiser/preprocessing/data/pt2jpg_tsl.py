# %%
from pathlib import Path

import ipdb
import lightning as L
import torch
from fran.localiser.transforms import NormaliseZeroToOne, TSLRegions
from fran.transforms.imageio import LoadTorchd
from fran.transforms.intensitytransforms import MakeBinary
from fran.transforms.misc_transforms import BoundingBoxesYOLOd
from monai.apps.detection.transforms.dictionary import ConvertBoxToStandardModed
from monai.data.dataset import Dataset
from monai.transforms import Compose
from monai.transforms.croppad.dictionary import BoundingRectd
from monai.transforms.spatial.dictionary import RandFlipd, RandRotated, RandZoomd, Resized
from monai.transforms.utility.dictionary import (
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
)
from torch.utils.data import random_split


tr = ipdb.set_trace
from fran.localiser.preprocessing.data.pt2jpg import PreprocessorPT2JPG


class PreprocessorPT2JPG_TSL(PreprocessorPT2JPG):
    # keys_tr = "L,ToBinary,Et,Et2,N,Flip1,Flip2,Zoom,Resize,ExtractBbox,CB,YoloBboxes"
    keys_tr = "L,ToBinary,Et,Et2,N,Resize,ExtractBbox,CB,YoloBboxes,DelI"
    keys_val = "L,ToBinary,Et,Et2,N,Resize,ExtractBbox,CB,YoloBboxes,DelI"

    def __init__(self, data_dir: str = "./", batch_size: int = 4):
        super().__init__(data_dir=data_dir, batch_size=batch_size)
        self.image_key = "image"
        self.label_key = "lm"
        self.tsl_regions = TSLRegions()
        self.data_yaml = self.tsl_regions.data_yaml

    def create_transforms(self, probs=0.3):
        box_key = "lm_bbox"
        self.outputsize = [512, 512]
        self.L = LoadTorchd([self.image_key, self.label_key])
        self.E = EnsureChannelFirstd(keys=[self.image_key])
        self.ToBinary = MakeBinary([self.label_key])
        self.Et = EnsureTyped(keys=[self.image_key], dtype=torch.float32)
        self.Et2 = EnsureTyped(keys=[self.label_key], dtype=torch.long)
        self.E2 = EnsureTyped(keys=[self.image_key], dtype=torch.float16)
        self.ExtractBbox = BoundingRectd(keys=[self.label_key])
        self.CB = ConvertBoxToStandardModed(mode="xxyy", box_keys=[box_key])
        self.YoloBboxes = BoundingBoxesYOLOd(
            [box_key], 2, key_template_tensor=self.label_key, output_keys=["bbox_yolo"]
        )
        self.Rotate = RandRotated(
            keys=[self.image_key, self.label_key],
            prob=probs,
            keep_size=True,
            mode=["bilinear", "nearest"],
            range_x=[0.4, 0.4],
            lazy=True,
        )
        self.Zoom = RandZoomd(
            keys=[self.image_key, self.label_key],
            mode=["bilinear", "nearest"],
            prob=probs,
            min_zoom=0.7,
            max_zoom=1.4,
            padding_mode="constant",
            keep_size=True,
            lazy=True,
        )
        self.Flip1 = RandFlipd(
            keys=[self.image_key, self.label_key], prob=probs, spatial_axis=0, lazy=True
        )
        self.Flip2 = RandFlipd(
            keys=[self.image_key, self.label_key], prob=probs, spatial_axis=1, lazy=True
        )
        self.Resize = Resized(
            keys=[self.image_key, self.label_key],
            spatial_size=self.outputsize,
            mode=["bilinear", "nearest"],
            lazy=True,
        )
        self.N = NormaliseZeroToOne(keys=[self.image_key])
        self.DelI = DeleteItemsd(keys=[self.label_key])
        self.transforms_dict = {
            "N": self.N,
            "L": self.L,
            "E": self.E,
            "Et": self.Et,
            "ToBinary": self.ToBinary,
            "Et2": self.Et2,
            "E2": self.E2,
            "ExtractBbox": self.ExtractBbox,
            "YoloBboxes": self.YoloBboxes,
            "DelI": self.DelI,
            "CB": self.CB,
            "Rotate": self.Rotate,
            "Zoom": self.Zoom,
            "Flip1": self.Flip1,
            "Flip2": self.Flip2,
            "Resize": self.Resize,
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

    def format_bbox_rows(self, dici):
        rows = []
        for cls_id, box in enumerate(dici["bbox_yolo"]):
            rows.append([cls_id] + box.tolist())
        return rows

    def data_yaml_lines(self):
        return self.tsl_regions.data_yaml_lines()

    def write_data_yaml(self, out_dir):
        with open(out_dir / "data.yaml", "w") as f:
            f.write(self.data_yaml)
