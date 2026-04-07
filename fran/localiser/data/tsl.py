# %%
from pathlib import Path

import ipdb
import lightning as L
import torch
from fran.transforms.imageio import LoadTorchd
from fran.transforms.intensitytransforms import MakeBinary
from fran.transforms.misc_transforms import BoundingBoxesYOLOd
from monai.apps.detection.transforms.dictionary import ConvertBoxToStandardModed
from monai.data.dataset import Dataset
from monai.transforms import Compose
from monai.transforms.croppad.dictionary import BoundingRectd
from monai.transforms.intensity.dictionary import (
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.transforms.spatial.dictionary import RandFlipd, RandRotated, RandZoomd, Resized
from monai.transforms.utility.dictionary import (
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    RepeatChanneld,
)
from torch.utils.data import random_split

from fran.localiser.data.lidc import DetectDS, PreprocessorPT2JPG, write_list_to_txt

tr = ipdb.set_trace


class DetectDataModuleTSL(PreprocessorPT2JPG):
    keys_tr = "L,MkB,Et,Et2,N,IntensityTfms,Flip1,Flip2,Zoom,Resize,ExtractBbox,CB,YoloBboxes,Rp"
    keys_val = "L,MkB,Et,Et2,N,Resize,ExtractBbox,CB,YoloBboxes,DelI,Rp"

    def create_transforms(self, probs=0.3, probs_intensity=0.3):
        image_key = "image"
        label_key = "lm"
        box_key = "lm_bbox"
        outputsize = [512, 512]
        probs_int = probs_intensity

        L = LoadTorchd([image_key, label_key])
        E = EnsureChannelFirstd(keys=[image_key])
        MkB = MakeBinary([label_key])
        Et = EnsureTyped(keys=[image_key], dtype=torch.float32)
        Et2 = EnsureTyped(keys=[label_key], dtype=torch.long)
        E2 = EnsureTyped(keys=[image_key], dtype=torch.float16)
        ExtractBbox = BoundingRectd(keys=[label_key])
        CB = ConvertBoxToStandardModed(mode="xxyy", box_keys=[box_key])
        YoloBboxes = BoundingBoxesYOLOd(
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
        N = NormalizeIntensityd(keys=[image_key])
        int_augs = [
            RandGaussianSmoothd(
                keys=[image_key],
                prob=probs_int,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys="image", factors=0.25, prob=probs_int),
            RandGaussianNoised(keys=["image"], mean=0, prob=probs_int, std=0.1),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=probs_int),
            RandAdjustContrastd(["image"], gamma=(0.7, 1.5)),
        ]
        Rp = RepeatChanneld(keys=[image_key], repeats=3)
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
            "YoloBboxes": YoloBboxes,
            "DelI": DelI,
            "CB": CB,
            "Rotate": Rotate,
            "Zoom": Zoom,
            "IntensityTfms": int_augs,
            "Flip1": Flip1,
            "Flip2": Flip2,
            "Resize": Resize,
            "Rp": Rp,
        }

    def set_transforms(self, keys_tr: str, keys_val: str):
        self.tfms_train = self.tfms_from_dict(keys_tr)
        self.tfms_valid = self.tfms_from_dict(keys_val)

    def tfms_from_dict(self, keys: str):
        keys = keys.replace(" ", "").split(",")
        tfms = []
        for key in keys:
            tfm = self.transforms_dict[key]
            if key == "IntensityTfms":
                tfms.extend(tfm)
            else:
                tfms.append(tfm)
        return Compose(tfms)

    def format_bbox_rows(self, dici):
        rows = []
        for cls_id, box in enumerate(dici["bbox_yolo"]):
            rows.append([cls_id] + box.tolist())
        return rows

    def data_yaml_lines(self):
        lines = ["names:"]
        for class_name in self.class_names:
            lines.append("- " + class_name)
        lines.extend(
            [
                "nc: " + str(len(self.class_names)),
                "test: ../test/images",
                "train: ../train/images",
                "val: ../valid/images",
                "",
            ]
        )
        return lines

    @property
    def class_names(self):
        return [
            "class0",
            "class1",
            "class2",
            "class3",
            "class4",
        ]
