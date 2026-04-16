# Codex: localiser GPU dataloader replica.
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, colorstr


WINDOW_PRESETS = {
    "b": [-450.0, 1050.0],
    "c": [-1350.0, 150.0],
    "a": [-150.0, 250.0],
}


def resolve_cuda_device(device):
    assert torch.cuda.is_available(), "CUDA is required for localiser GPU training"
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str) and device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


class LocaliserPTDataset(Dataset):
    def __init__(self, data_folder, mode, size_3d=(256, 256, 256)):
        self.data_folder = Path(data_folder)
        self.mode = mode
        self.size_3d = size_3d
        self.data = self.create_data_dicts()
        self.labels = []
        for item in self.data:
            self.labels.append(
                {
                    "im_file": item["image"],
                    "lm_file": item["lm"],
                    "cls": torch.zeros(0, 1),
                    "bboxes": torch.zeros(0, 4),
                    "segments": [],
                    "keypoints": None,
                }
            )

    def create_data_dicts(self):
        images_folder = self.data_folder / "images"
        lms_folder = self.data_folder / "lms"
        data = []
        for image_fn in sorted(images_folder.glob("*.pt")):
            lm_fn = lms_folder / image_fn.name
            assert lm_fn.exists(), f"Missing labelmap {lm_fn}"
            data.append({"image": str(image_fn), "lm": str(lm_fn)})
        return data

    def __len__(self):
        return len(self.data)

    def load_tensor(self, fn):
        tensor = torch.load(fn, weights_only=False)
        if hasattr(tensor, "as_tensor"):
            tensor = tensor.as_tensor()
        tensor = torch.as_tensor(tensor)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, index):
        item = self.data[index]
        image = self.resize_3d(self.load_tensor(item["image"]).float(), "trilinear")
        lm = self.resize_3d(self.load_tensor(item["lm"]).float(), "nearest")
        return {
            "image": image,
            "lm": lm,
            "im_file": item["image"],
            "lm_file": item["lm"],
            "mode": self.mode,
        }

    def resize_3d(self, tensor, mode):
        tensor = tensor.unsqueeze(0)
        if mode == "nearest":
            tensor = F.interpolate(tensor, size=self.size_3d, mode=mode)
        else:
            tensor = F.interpolate(
                tensor, size=self.size_3d, mode=mode, align_corners=False
            )
        return tensor[0]


def localiser_raw_collate(batch):
    images = []
    lms = []
    im_files = []
    lm_files = []
    for item in batch:
        images.append(item["image"])
        lms.append(item["lm"])
        im_files.append(item["im_file"])
        lm_files.append(item["lm_file"])
    return {
        "image": torch.stack(images),
        "lm": torch.stack(lms),
        "im_file": im_files,
        "lm_file": lm_files,
        "mode": batch[0]["mode"],
    }


class DictCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class ToDeviced:
    def __init__(self, keys, device):
        self.keys = keys
        self.device = device

    def __call__(self, data):
        for key in self.keys:
            data[key] = data[key].to(self.device, non_blocking=True)
        return data


class MakeBinaryd:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            data[key] = data[key] > 0
        return data


class Window3Channeld:
    def __init__(self, key, randomize=False):
        self.key = key
        self.randomize = randomize

    def __call__(self, data):
        image = data[self.key]
        outs = []
        for lower, upper in WINDOW_PRESETS.values():
            if self.randomize:
                width = upper - lower
                centre = 0.5 * (lower + upper)
                width = width * torch.empty((), device=image.device).uniform_(0.9, 1.1)
                centre = centre + torch.empty((), device=image.device).uniform_(
                    -0.1 * width, 0.1 * width
                )
                lower = centre - 0.5 * width
                upper = centre + 0.5 * width
            out = image.clamp(lower, upper)
            outs.append((out - lower) / (upper - lower))
        data[self.key] = torch.cat(outs, dim=1)
        return data


class Project2DLocaliserd:
    def __init__(self, dim, image_out, lm_out, image_key="image", lm_key="lm"):
        self.dim = dim
        self.image_out = image_out
        self.lm_out = lm_out
        self.image_key = image_key
        self.lm_key = lm_key

    def __call__(self, data):
        image = data[self.image_key]
        lm = data[self.lm_key].float()
        data[self.image_out] = image.mean(dim=self.dim)
        data[self.lm_out] = lm.sum(dim=self.dim) > 0
        return data


class Resize2DLocaliserd:
    def __init__(self, image_keys, lm_keys, size):
        self.image_keys = image_keys
        self.lm_keys = lm_keys
        self.size = size

    def __call__(self, data):
        for key in self.image_keys:
            data[key] = F.interpolate(
                data[key], size=self.size, mode="bilinear", align_corners=False
            )
        for key in self.lm_keys:
            data[key] = F.interpolate(data[key].float(), size=self.size, mode="nearest")
        return data


class IntensityAugLocaliserd:
    def __init__(self, keys, prob=0.5):
        self.keys = keys
        self.prob = prob

    def __call__(self, data):
        for key in self.keys:
            image = data[key]
            if torch.rand((), device=image.device) < self.prob:
                scale = torch.empty((), device=image.device).uniform_(0.7, 1.3)
                shift = torch.empty((), device=image.device).uniform_(-0.1, 0.1)
                image = image * scale + shift
            data[key] = image.clamp(0, 1)
        return data


class Flip2DLocaliserd:
    def __init__(self, image_keys, lm_keys, prob=0.3):
        self.image_keys = image_keys
        self.lm_keys = lm_keys
        self.prob = prob

    def __call__(self, data):
        for dim in [-1, -2]:
            if torch.rand((), device=data[self.image_keys[0]].device) < self.prob:
                for key in self.image_keys:
                    data[key] = torch.flip(data[key], dims=[dim])
                for key in self.lm_keys:
                    data[key] = torch.flip(data[key], dims=[dim])
        return data


class PackYOLOLocaliserd:
    def __call__(self, data):
        images = torch.cat([data["image1"], data["image2"]], dim=0).clamp(0, 1)
        lms = torch.cat([data["lm1"], data["lm2"]], dim=0)
        cls, bboxes, batch_idx = self.bboxes_from_lms(lms)
        n = data["image1"].shape[0]
        data["img"] = images
        data["cls"] = cls
        data["bboxes"] = bboxes
        data["batch_idx"] = batch_idx
        data["im_file"] = data["im_file"] + data["im_file"]
        data["ori_shape"] = [(images.shape[-2], images.shape[-1])] * (2 * n)
        data["resized_shape"] = data["ori_shape"]
        return data

    def bboxes_from_lms(self, lms):
        n_images, n_classes, height, width = lms.shape
        cls_out = []
        boxes_out = []
        batch_out = []
        for batch_i in range(n_images):
            projection_offset = 0 if batch_i < n_images // 2 else n_classes
            for cls_i in range(n_classes):
                coords = torch.nonzero(lms[batch_i, cls_i] > 0, as_tuple=False)
                if coords.numel() == 0:
                    continue
                y0 = coords[:, 0].min()
                y1 = coords[:, 0].max() + 1
                x0 = coords[:, 1].min()
                x1 = coords[:, 1].max() + 1
                xc = (x0 + x1).float() * 0.5 / width
                yc = (y0 + y1).float() * 0.5 / height
                bw = (x1 - x0).float() / width
                bh = (y1 - y0).float() / height
                cls_out.append(torch.tensor([cls_i + projection_offset], device=lms.device))
                boxes_out.append(torch.stack([xc, yc, bw, bh]))
                batch_out.append(torch.tensor(batch_i, device=lms.device))
        if len(boxes_out) == 0:
            return (
                torch.zeros(0, 1, device=lms.device),
                torch.zeros(0, 4, device=lms.device),
                torch.zeros(0, device=lms.device),
            )
        return (
            torch.stack(cls_out).float(),
            torch.stack(boxes_out).float(),
            torch.stack(batch_out).float(),
        )


class LocaliserGPUDataManager(LightningDataModule):
    def __init__(self, data_folder, batch_size, mode, imgsz=256, device=0, workers=0):
        super().__init__()
        self.data_folder = Path(data_folder)
        self.batch_size = batch_size
        self.mode = mode
        self.imgsz = imgsz
        self.device = resolve_cuda_device(device)
        self.workers = workers
        self.keys_tr = "Dev,ToBinary,WinR,P1,P2,Flip2D,Resize2D,IntAugs,PackYOLO"
        self.keys_val = "Dev,ToBinary,Win,P1,P2,Resize2D,PackYOLO"

    def setup(self, stage=None):
        self.ds = LocaliserPTDataset(self.data_folder, self.mode)
        self.create_transforms()
        self.gpu_tfms = self.tfms_from_dict(
            self.keys_tr if self.mode == "train" else self.keys_val
        )
        self.dl = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=self.mode == "train",
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=localiser_raw_collate,
        )

    def create_transforms(self):
        image_keys = ["image1", "image2"]
        lm_keys = ["lm1", "lm2"]
        self.transforms_dict = {
            "Dev": ToDeviced(["image", "lm"], self.device),
            "ToBinary": MakeBinaryd(["lm"]),
            "WinR": Window3Channeld("image", randomize=True),
            "Win": Window3Channeld("image", randomize=False),
            "P1": Project2DLocaliserd(dim=2, image_out="image1", lm_out="lm1"),
            "P2": Project2DLocaliserd(dim=3, image_out="image2", lm_out="lm2"),
            "Flip2D": Flip2DLocaliserd(image_keys, lm_keys),
            "Resize2D": Resize2DLocaliserd(image_keys, lm_keys, (self.imgsz, self.imgsz)),
            "IntAugs": IntensityAugLocaliserd(image_keys),
            "PackYOLO": PackYOLOLocaliserd(),
        }

    def tfms_from_dict(self, keys):
        transforms = []
        for key in keys.replace(" ", "").split(","):
            transforms.append(self.transforms_dict[key])
        return DictCompose(transforms)


class CTAugDetectionTrainerGPU(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.fran_data_folder = overrides.pop("fran_data_folder")
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dm = LocaliserGPUDataManager(
            data_folder=self.fran_data_folder,
            batch_size=batch_size,
            mode=mode,
            imgsz=self.args.imgsz,
            device=self.device,
            workers=self.args.workers,
        )
        dm.setup()
        if mode == "train":
            self.localiser_train_dm = dm
        else:
            self.localiser_val_dm = dm
        return dm.dl

    def preprocess_batch(self, batch):
        dm = self.localiser_train_dm if batch["mode"] == "train" else self.localiser_val_dm
        return dm.gpu_tfms(batch)

    def plot_training_labels(self):
        return None


if __name__ == "__main__":
# %%
# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
    from fran.configs.parser import ConfigMaker
    from fran.managers.project import Project

    P = Project(project_title="totalseg")
    C = ConfigMaker(P)
    _ = C

    device = 0
    data_folder = Path("/s/tmp/nii2pt_tsl3d_debug")
    dm = LocaliserGPUDataManager(
        data_folder=data_folder,
        batch_size=2,
        mode="train",
        imgsz=256,
        device=device,
        workers=0,
    )
    dm.setup()

# %%
    torch.cuda.synchronize(device)
    start = time.time()
    for batch_i, batch in enumerate(dm.dl):
        batch = dm.gpu_tfms(batch)
        assert batch["img"].is_cuda
        assert batch["img"].ndim == 4
        assert batch["bboxes"].shape[1] == 4
        print(
            batch_i,
            tuple(batch["img"].shape),
            tuple(batch["cls"].shape),
            tuple(batch["bboxes"].shape),
            batch["img"].device,
        )
        if batch_i == 9:
            break
    torch.cuda.synchronize(device)
    print("10 batch seconds:", round(time.time() - start, 3))
# %%
    batch.keys()


# %%
