import random

import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import de_parallel


class CTIntensityAugment:
    def __init__(
        self,
        p=0.5,
        contrast=0.15,
        brightness=0.1,
        gamma=0.15,
        noise_std=0.03,
        blur_p=0.15,
    ):
        self.p = p
        self.contrast = contrast
        self.brightness = brightness
        self.gamma = gamma
        self.noise_std = noise_std
        self.blur_p = blur_p

    def __call__(self, labels):
        if random.random() > self.p:
            return labels
        img = labels["img"].astype(np.float32) / 255.0
        img = img * random.uniform(1 - self.contrast, 1 + self.contrast)
        img = img + random.uniform(-self.brightness, self.brightness)
        img = np.clip(img, 0.0, 1.0)
        if self.gamma > 0:
            img = img ** random.uniform(1 - self.gamma, 1 + self.gamma)
        if self.noise_std > 0:
            img = img + np.random.normal(
                0.0, random.uniform(0.0, self.noise_std), img.shape
            ).astype(np.float32)
        if random.random() < self.blur_p:
            img = cv2.GaussianBlur(
                img,
                random.choice([(3, 3), (5, 5)]),
                sigmaX=random.uniform(0.4, 1.0),
            )
        labels["img"] = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        return labels


class CTAugYOLODataset(YOLODataset):
    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            transforms.insert(-1, CTIntensityAugment())
        return transforms


class CTAugDetectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode=mode, batch=batch)
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return CTAugYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
        )
