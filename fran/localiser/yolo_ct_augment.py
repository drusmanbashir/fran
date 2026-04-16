from pathlib import Path
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
from fran.transforms.spatialtransforms import Project2D
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import colorstr


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


WINDOW_PRESETS = {
    "a": [-150.0, 250.0],
    "b": [-450.0, 1050.0],
    "c": [-1350.0, 150.0],
}


def apply_window_tensor(image, window, randomize=False):
    lower, upper = WINDOW_PRESETS[window]
    if randomize:
        width = upper - lower
        centre = 0.5 * (lower + upper)
        width = width * random.uniform(0.9, 1.1)
        centre = centre + random.uniform(-0.1 * width, 0.1 * width)
        lower = centre - 0.5 * width
        upper = centre + 0.5 * width
    image = torch.clamp(image, lower, upper)
    image = (image - lower) / (upper - lower)
    return image


class RandomWindowTensor3Channeld:
    def __init__(self, image_key, randomize=False):
        self.image_key = image_key
        self.randomize = randomize

    def __call__(self, data):
        image = data[self.image_key]
        outs = []
        for window in WINDOW_PRESETS:
            outs.append(apply_window_tensor(image, window, self.randomize))
        data[self.image_key] = torch.cat(outs, dim=0)
        return data


class CTAugYOLODataset(YOLODataset):
    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            transforms.insert(-1, CTIntensityAugment())
        return transforms


class CTAugYOLODataset3D(YOLODataset):
    def __init__(self, *args, **kwargs):
        self.window = RandomWindowTensor3Channeld("image", randomize=kwargs["augment"])
        self.P1 = Project2D(
            keys=["lm", "image"],
            operations=["sum", "mean"],
            dim=1,
            output_keys=["lm1", "image1"],
        )
        self.P2 = Project2D(
            keys=["lm", "image"],
            operations=["sum", "mean"],
            dim=2,
            output_keys=["lm2", "image2"],
        )
        super().__init__(*args, **kwargs)

    def get_img_files(self, img_path):
        p = Path(img_path)
        files = sorted(str(fn) for fn in p.glob("*.pt"))
        assert len(files) > 0, f"{self.prefix}No pt files found in {img_path}"
        if self.fraction < 1:
            n_files = round(len(files) * self.fraction)
            files = files[:n_files]
        return files

    def get_labels(self):
        labels = []
        for im_file in self.im_files:
            im_file = Path(im_file)
            lm_file = im_file.parent.parent / "lms" / im_file.name
            labels.append(
                {
                    "im_file": str(im_file),
                    "lm_file": str(lm_file),
                    "shape": (1, 1),
                    "cls": np.zeros((0, 1), dtype=np.float32),
                    "bboxes": np.zeros((0, 4), dtype=np.float32),
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        return labels

    def load_pt_pair(self, index):
        label = self.labels[index]
        image = torch.load(label["im_file"], weights_only=False)
        lm = torch.load(label["lm_file"], weights_only=False)
        if hasattr(image, "as_tensor"):
            image = image.as_tensor()
        if hasattr(lm, "as_tensor"):
            lm = lm.as_tensor()
        image = torch.as_tensor(image).float()
        lm = torch.as_tensor(lm)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if lm.ndim == 3:
            lm = lm.unsqueeze(0)
        return image, lm

    def create_bbox_array(self, lm):
        if lm.ndim == 2:
            lm = lm.unsqueeze(0)
        h, w = lm.shape[-2:]
        boxes = []
        classes = []
        for cls_id, channel in enumerate(lm):
            coords = torch.nonzero(channel > 0, as_tuple=False)
            if len(coords) == 0:
                continue
            y0 = coords[:, 0].min().item()
            y1 = coords[:, 0].max().item() + 1
            x0 = coords[:, 1].min().item()
            x1 = coords[:, 1].max().item() + 1
            xc = 0.5 * (x0 + x1) / w
            yc = 0.5 * (y0 + y1) / h
            bw = (x1 - x0) / w
            bh = (y1 - y0) / h
            boxes.append([xc, yc, bw, bh])
            classes.append([cls_id])
        if len(boxes) == 0:
            cls = np.zeros((0, 1), dtype=np.float32)
            bboxes = np.zeros((0, 4), dtype=np.float32)
        else:
            cls = np.asarray(classes, dtype=np.float32)
            bboxes = np.asarray(boxes, dtype=np.float32)
        return cls, bboxes

    def make_projection_label(self, base_label, image, lm, index):
        image_np = image.permute(1, 2, 0).contiguous().cpu().numpy()
        image_np = (np.clip(image_np, 0.0, 1.0) * 255).astype(np.uint8)
        cls, bboxes = self.create_bbox_array(lm)
        label = {
            "im_file": base_label["im_file"],
            "shape": image_np.shape[:2],
            "cls": cls,
            "bboxes": bboxes,
            "segments": [],
            "keypoints": None,
            "normalized": True,
            "bbox_format": "xywh",
            "img": image_np,
            "ori_shape": image_np.shape[:2],
            "resized_shape": image_np.shape[:2],
            "ratio_pad": (1.0, 1.0),
        }
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def get_image_and_label(self, index):
        base_label = self.labels[index]
        image, lm = self.load_pt_pair(index)
        data = {"image": image, "lm": lm}
        data = self.window(data)
        data = self.P1(data)
        data = self.P2(data)
        out = []
        for suffix in [1, 2]:
            out.append(
                self.make_projection_label(
                    base_label=base_label,
                    image=data[f"image{suffix}"],
                    lm=data[f"lm{suffix}"],
                    index=index,
                )
            )
        return out

    def __getitem__(self, index):
        labels = self.get_image_and_label(index)
        out = []
        for label in labels:
            out.append(self.transforms(label))
        return out

    def build_transforms(self, hyp=None):
        hyp.mosaic = 0.0
        hyp.mixup = 0.0
        transforms = super().build_transforms(hyp)
        if self.augment:
            transforms.insert(-1, CTIntensityAugment())
        return transforms

    @staticmethod
    def collate_fn(batch):
        flat_batch = []
        for item in batch:
            if isinstance(item, list):
                flat_batch.extend(item)
            else:
                flat_batch.append(item)
        return YOLODataset.collate_fn(flat_batch)


class CTAugDetectionTrainer(DetectionTrainer):
    def use_3d_pt_dataset(self, img_path):
        return len(list(Path(img_path).glob("*.pt"))) > 0

    def _get_stride(self):
        if not self.model:
            return 32
        m = self.model.module if hasattr(self.model, "module") else self.model
        return max(int(m.stride.max()), 32)

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = self._get_stride()

        if self.use_3d_pt_dataset(img_path):
            return CTAugYOLODataset3D(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect,
                cache=None,
                single_cls=self.args.single_cls or False,
                stride=gs,
                pad=0.0,
                prefix=colorstr(f"{mode}: "),
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                fraction=self.args.fraction,
            )

        if mode != "train":
            return super().build_dataset(img_path, mode=mode, batch=batch)

        return CTAugYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
        )
# %%
if __name__ == "__main__":
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------

    image_fn = Path("/tmp/yolo3d_debug/images/case_0000.pt")
    lm_fn = Path("/tmp/yolo3d_debug/lms/case_0000.pt")
    fldr_imgs = image_fn.parent
    fldr_lms = lm_fn.parent

    hyp = deepcopy(DEFAULT_CFG)
    ds = CTAugYOLODataset3D(
        img_path=str(fldr_imgs),
        imgsz=64,
        batch_size=2,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="debug: ",
        task="detect",
        classes=None,
        data={"names": ["ROI"]},
        fraction=1.0,
    )

# %%
    print("n_files", len(ds.im_files))
    print("first_file", ds.im_files[0])
    print("first_lm", ds.labels[0]["lm_file"])
    print("label0", ds.labels[0])

# %%
    raw = ds.get_image_and_label(0)
    print("n_projections", len(raw))
    for i, dici in enumerate(raw):
        print(
            "raw",
            i,
            dici["img"].shape,
            dici["cls"].shape,
            dici["bboxes"].shape,
        )

# %%
    item = ds[0]
    print("item_len", len(item))
    for i, dici in enumerate(item):
        print(
            "tfm",
            i,
            dici["img"].shape,
            dici["cls"].shape,
            dici["bboxes"].shape,
            dici["batch_idx"].shape,
        )

# %%
    batch = ds.collate_fn([ds[0], ds[0]])
    print("batch_img", batch["img"].shape)
    print("batch_cls", batch["cls"].shape)
    print("batch_bboxes", batch["bboxes"].shape)
    print("batch_idx", batch["batch_idx"])
