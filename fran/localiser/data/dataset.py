import random
from pathlib import Path

import ipdb
import numpy as np
import torch
from fran.localiser.helpers import draw_image_lm_bbox
from fran.managers.project import Project
from matplotlib.transforms import Transform
from ultralytics.data.dataset import YOLODataset
from utilz.cprint import cprint
from utilz.stringz import info_from_filename

tr = ipdb.set_trace

import cv2
import numpy as np
import torch
from fran.localiser.helpers import draw_image_lm_bbox
from fran.localiser.transforms.transforms import (
    NormaliseZeroTo255,
    WindowTensor3Channeld,
    WindowTensor3ChannkjeldRand,
)
from fran.managers.project import Project
from fran.transforms.imageio import TorchReader
from fran.transforms.intensitytransforms import MakeBinary, RandRandGaussianNoised
from fran.transforms.misc_transforms import BoundingBoxesYOLOd
from fran.transforms.spatialtransforms import Project2D
from monai.apps.detection.transforms.dictionary import ConvertBoxToStandardModed
from monai.transforms import Compose as MonaiCompose
from monai.transforms.croppad.dictionary import BoundingRectd
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import (
    RandFlipd,
    RandRotated,
    RandZoomd,
    Resized,
)
from monai.transforms.utility.dictionary import DeleteItemsd, EnsureTyped
from ultralytics.data.augment import Compose as YOLOCompose
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, colorstr
from ultralytics.utils.torch_utils import de_parallel


class Fix2DBBox(Transform):
    def __init__(self, bbox_keys):
        self.bbox_keys = bbox_keys

    def __call__(self, data):
        for key in self.bbox_keys:
            data[key] = self.yyxx_to_xxyy(data[key])
        return data

    def yyxx_to_xxyy(self, bb):
        return bb[:, [2, 3, 0, 1]]


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


WINDOW_TO_IND_MAP = {"a": 0, "b": 1, "c": 2}
WINDOW_PRESETS = {
    "b": [-450.0, 1050.0],
    "c": [-1350.0, 150.0],
    "a": [-150.0, 250.0],
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
    keys_tr = "LT,ToBinary,Et,Et2,WinR,P1,P2,IntAugs,Flip1,Flip2,Zoom,Resize2D,N255,LM2YoloBBox,DelI"
    keys_val = "LT,ToBinary,Et,Et2,Win,P1,P2,Resize2D,N255,LM2YoloBBox,DelI"

    def __init__(
        self,
        project: Project,
        configs: dict,
        data_folder: str,
        mode: str,  # "train" | "val"
        imgsz: int,
        *args,
        **kwargs,
    ):

        augment = mode == "train"
        self.debug = kwargs.pop("debug", False)
        self.data_folder = Path(data_folder)
        self.project = project
        self.configs = configs
        self.mode = mode
        self.keys = self.keys_tr if augment else self.keys_val
        self.create_data_dicts()

        super().__init__(
            img_path=str(Path(data_folder) / "images"),
            imgsz=imgsz,
            augment=augment,
            *args,
            **kwargs,
        )

    def create_data_dicts(self):
        case_ids = None
        # Put fold/split selection here. Set case_ids to the selected case_id list.
        # train, val = self.project.get_train_val_case_ids(fold=...)
        if self.mode == "train":
            pass

        elif self.mode == "val":
            pass
        else:
            raise ValueError(f"Invalid mode {self.mode}")

        imgs_folder = self.data_folder / "images"
        img_fns = list(imgs_folder.glob("*.pt"))
        lms_folder = self.data_folder / "lms"
        self.data_dicts = []
        for fn in img_fns:
            case_id = info_from_filename(fn.name, full_caseid=True)["case_id"]
            if case_ids is None or case_id in case_ids:
                lm_fn = lms_folder / fn.name
                assert lm_fn.exists(), (
                    f"Landmark file {lm_fn} does not exist for image {fn}"
                )
                dici = {"img": fn, "lm": lm_fn}
                self.data_dicts.append(dici)

    def tfms_from_dict(self, keys: str):
        keys2 = keys.replace(" ", "")
        keys_list = keys2.split(",")
        tfms = []
        for key in keys_list:
            try:
                tfm = self.transforms_dict[key]
                tfms.append(tfm)
            except KeyError as e:
                print("All keys are: ", self.transforms_dict.keys())
                print(f"Transform {key} not found.")
                raise e

        tfms = MonaiCompose(tfms)
        return tfms

    def apply_monai_transforms(self, data: dict):
        if self.debug == False:
            return self.monai_tfms(data)
        else:
            return self.apply_monai_transforms_debug(data)

    def apply_monai_transforms_debug(self, data: dict):
        keys = self.keys
        keys = keys.replace(" ", "").split(",")
        for key in keys:
            cprint(f"{key}", color="yellow")
            tr()
            tfm = self.transforms_dict[key]
            if isinstance(data, list | tuple):
                data = data[0]
            data = tfm(data)
        return data

    def apply_yolo_transforms(self, label: dict):
        if self.debug == False:
            return self.transforms(label)
        else:
            return self.apply_yolo_transforms_debug(label)

    def apply_yolo_transforms_debug(self, label: dict):
        for tfm in self.transforms.transforms:
            cprint(tfm.__class__.__name__, color="yellow")
            tr()
            label = tfm(label)
        return label

    def get_img_files(self, img_path):
        files = [str(dici["img"]) for dici in self.data_dicts]
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

    def extract_classes_yolobbox(self, data):
        lm1bb = data["bbox_yolo1"]
        valid1 = lm1bb.sum(axis=1) > 0
        n1 = lm1bb.shape[0]
        cls1 = torch.arange(n1).reshape(-1, 1)
        data["bbox_yolo1"] = lm1bb[valid1]
        data["cls1"] = cls1[valid1]

        lm2bb = data["bbox_yolo2"]
        valid2 = lm2bb.sum(axis=1) > 0
        n2 = lm2bb.shape[0]
        cls2 = torch.arange(n1, n1 + n2).reshape(-1, 1)
        data["bbox_yolo2"] = lm2bb[valid2]
        data["cls2"] = cls2[valid2]
        return data

    def make_projection_label(self, base_label, image, bboxes, cls, index):
        if hasattr(image, "as_tensor"):
            image = image.as_tensor()
        image_np = image.permute(1, 2, 0).contiguous().cpu().numpy()
        image_np = np.clip(image_np, 0.0, 255.0).astype(np.uint8)
        bboxes = torch.as_tensor(bboxes).float().cpu().numpy()
        if bboxes.ndim == 1:
            bboxes = bboxes[None]
        cls = torch.as_tensor(cls).float().cpu().numpy()
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
        data = {"img": image, "lm": lm}
        data = self.window(data)
        data = self.P1(data)
        data = self.P2(data)
        out = []
        for suffix in [1, 2]:
            out.append(
                self.make_projection_label(
                    base_label=base_label,
                    image=data[f"image{suffix}"],
                    bboxes=data[f"bbox_yolo{suffix}"],
                    cls=data[f"cls{suffix}"],
                    index=index,
                )
            )
        return out

    def __getitem__(self, index):
        dici = self.data_dicts[index]
        dici_out = self.apply_monai_transforms(dici)
        labels = self.get_image_and_label_from_dict(index, dici_out)
        out = []
        for label in labels:
            out.append(self.apply_yolo_transforms(label))
        return out

    def get_image_and_label_from_dict(self, index, data):
        data = self.extract_classes_yolobbox(data)
        base_label = self.labels[index]
        out = []
        for suffix in [1, 2]:
            out.append(
                self.make_projection_label(
                    base_label=base_label,
                    image=data[f"image{suffix}"],
                    bboxes=data[f"bbox_yolo{suffix}"],
                    cls=data[f"cls{suffix}"],
                    index=index,
                )
            )
        return out

    def trace_item_shapes(self, index):
        traces = []
        data = self.apply_monai_transforms(self.data_dicts[index])
        for key in [
            "img",
            "lm",
            "image1",
            "lm1",
            "bbox_yolo1",
            "image2",
            "lm2",
            "bbox_yolo2",
        ]:
            if key in data:
                value = data[key]
                traces.append((key, tuple(value.shape)))
        labels = self.get_image_and_label_from_dict(index, data)
        for i, label in enumerate(labels):
            traces.append((f"label{i}_img", label["img"].shape))
            traces.append((f"label{i}_cls", label["cls"].shape))
            traces.append((f"label{i}_boxes", label["instances"].bboxes.shape))
            formatted = self.apply_yolo_transforms(label)
            traces.append((f"formatted{i}_img", tuple(formatted["img"].shape)))
            traces.append((f"formatted{i}_cls", tuple(formatted["cls"].shape)))
            traces.append((f"formatted{i}_boxes", tuple(formatted["bboxes"].shape)))
        return traces

    def save_bbox_review_images(self, folder, n_images=16, channel=0, batch_size=4):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        saved = 0
        for start in range(0, len(self.data_dicts), batch_size):
            for index in range(start, min(start + batch_size, len(self.data_dicts))):
                data = self.apply_monai_transforms(self.data_dicts[index])
                for suffix in [1, 2]:
                    img = data[f"image{suffix}"][channel]
                    lm = data[f"lm{suffix}"]
                    boxes = data[f"lm{suffix}_bbox"]
                    valid = boxes.sum(axis=1) > 0
                    for cls, box in zip(
                        torch.arange(boxes.shape[0])[valid], boxes[valid]
                    ):
                        lm_cls = lm[int(cls)]
                        filename = (
                            folder
                            / f"{saved:03d}_case{index}_p{suffix}_c{int(cls)}.jpg"
                        )
                        draw_image_lm_bbox(
                            img, lm_cls, *box.tolist(), filename=filename
                        )
                        saved += 1
                        if saved == n_images:
                            return

    @staticmethod
    def collate_fn(batch):
        flat_batch = []
        for item in batch:
            flat_batch.extend(item)
        return YOLODataset.collate_fn(flat_batch)

    def build_transforms(self, hyp=None):
        image_key = "img"
        lm_key = "lm"
        box_keys = ["lm1_bbox", "lm2_bbox"]

        imgs_proj_keys = ["image1", "image2"]
        lms_proj_keys = ["lm1", "lm2"]
        proj_keys_all = imgs_proj_keys + lms_proj_keys
        int_augs = {
            "contrast": [0.7, 1.3],
            "shift": [-1.0, 1.0],
            "scale": [-1.0, 1.0],
            "noise_ub": [0.0, 0.25],
            "noise": [0.05, 0.25],
            "brightness": [0.7, 2.0],
            "flip": 1,
        }
        probs_intensity = 0.5
        probs_spatial = 0.3
        output_size = (256, 256)
        LT = LoadImaged(keys=[image_key, lm_key], reader=TorchReader)
        YoloBboxes = BoundingBoxesYOLOd(
            keys=box_keys,
            dim=2,
            key_template_tensor=lms_proj_keys[0],
            output_keys=["bbox_yolo1", "bbox_yolo2"],
        )
        P1 = Project2D(
            keys=[image_key, lm_key],
            operations=["mean", "sum"],
            dim=1,
            output_keys=["image1", "lm1"],
        )
        P2 = Project2D(
            keys=[image_key, lm_key],
            operations=["mean", "sum"],
            dim=2,
            output_keys=["image2", "lm2"],
        )
        WinR = WindowTensor3ChanneldRand(image_key=image_key, prob=0.5, jitter=100)
        Win = WindowTensor3Channeld(image_key=image_key)

        IntensityTfms = [
            RandScaleIntensityd(
                keys=["image1", "image2"],
                factors=int_augs["scale"],
                prob=probs_intensity,
            ),
            RandRandGaussianNoised(
                keys=["image1"],
                std_limits=int_augs["noise"],
                prob=probs_intensity,
            ),
            RandRandGaussianNoised(
                keys=["image2"],
                std_limits=int_augs["noise"],
                prob=probs_intensity,
            ),
            RandShiftIntensityd(
                keys=["image1", "image2"],
                offsets=int_augs["shift"],
                prob=probs_intensity,
            ),
            RandAdjustContrastd(
                ["image1", "image2"], gamma=int_augs["contrast"], prob=probs_intensity
            ),
        ]
        IntAugs = MonaiCompose(IntensityTfms)
        # self.transforms_dict["IntensityTfms"] = IntensityTfms

        ToBinary = MakeBinary([lm_key])
        Et = EnsureTyped(keys=[image_key], dtype=torch.float32)
        Et2 = EnsureTyped(keys=[lm_key], dtype=torch.long)
        ExtractBbox = BoundingRectd(keys=lms_proj_keys)  # returns y0, y1, x0, x1
        FixBB = Fix2DBBox(bbox_keys=box_keys)

        CB = ConvertBoxToStandardModed(
            mode="xxyy", box_keys=box_keys
        )  # mode = current mode
        Rotate = RandRotated(
            keys=[image_key, lm_key],
            prob=probs_spatial,
            keep_size=True,
            mode=["bilinear", "nearest"],
            range_x=[0.4, 0.4],
            lazy=False,
        )
        Zoom = RandZoomd(
            keys=[image_key, lm_key],
            mode=["bilinear", "nearest"],
            prob=probs_spatial,
            min_zoom=0.7,
            max_zoom=1.4,
            padding_mode="constant",
            keep_size=True,
            lazy=False,
        )

        Flip1 = RandFlipd(
            keys=proj_keys_all, prob=probs_spatial, spatial_axis=0, lazy=False
        )
        Flip2 = RandFlipd(
            keys=proj_keys_all, prob=probs_spatial, spatial_axis=1, lazy=False
        )
        Resize2D = Resized(
            keys=proj_keys_all,
            spatial_size=output_size,
            mode=["bilinear", "bilinear", "nearest", "nearest"],
            lazy=False,
        )
        N255 = NormaliseZeroTo255(keys=imgs_proj_keys)
        LM2YoloBBox = MonaiCompose([ExtractBbox, FixBB, CB, YoloBboxes])
        DelI = DeleteItemsd(keys=[lm_key, image_key])
        self.transforms_dict = {
            "LT": LT,
            "ToBinary": ToBinary,
            "Et": Et,
            "Et2": Et2,
            "P1": P1,
            "P2": P2,
            "IntAugs": IntAugs,
            "LM2YoloBBox": LM2YoloBBox,
            "DelI": DelI,
            "CB": CB,
            "Rotate": Rotate,
            "Zoom": Zoom,
            "Flip1": Flip1,
            "Flip2": Flip2,
            "Resize2D": Resize2D,
            "N255": N255,
            "WinR": WinR,
            "Win": Win,
        }

        self.monai_tfms = self.tfms_from_dict(self.keys)
        yolo_tfms = super().build_transforms(hyp)

        # keep only safe YOLO tfms
        yolo_tfms = YOLOCompose(
            [
                t
                for t in yolo_tfms.transforms
                if t.__class__.__name__ in ["LetterBox", "Format"]
            ]
        )

        return yolo_tfms


class CTAugDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.fran_project = overrides.pop("fran_project")
        self.fran_configs = overrides.pop("fran_configs")
        self.fran_data_folder = overrides.pop("fran_data_folder")
        self.fran_debug = overrides.pop("fran_debug", False)
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def preprocess_batch(self, batch):
        device = self.device

        imgs = batch["img"].to(device, non_blocking=True).float()
        lms = batch["lm"].to(device, non_blocking=True)

        # ---- GPU transforms ----
        imgs = self.apply_window_gpu(imgs)
        imgs, lms = self.project_2d_gpu(imgs, lms)
        bboxes, cls = self.compute_bboxes_gpu(lms)

        # normalize
        imgs = imgs.clamp(0, 1)

        batch_out = {
            "img": imgs,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": torch.arange(len(imgs), device=device),
        }

        return batch_out

    def use_3d_pt_dataset(self, img_path):
        return len(list(Path(img_path).glob("*.pt"))) > 0

    def build_dataset(self, img_path, mode="train", batch=None):
        if self.use_3d_pt_dataset(img_path):
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
            data_folder = self.fran_data_folder
            if data_folder is None:
                data_folder = Path(img_path).parent
            return CTAugYOLODataset3D(
                project=self.fran_project,
                configs=self.fran_configs,
                data_folder=data_folder,
                mode=mode,
                debug=self.fran_debug,
                imgsz=self.args.imgsz,
                batch_size=batch,
                hyp=self.args,
                rect=self.args.rect,
                cache=None,
                single_cls=self.args.single_cls or False,
                stride=int(gs),
                pad=0.0,
                prefix=colorstr(f"{mode}: "),
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                fraction=self.args.fraction,
            )
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


class CustomYOLODataset(YOLODataset):
    def __init__(self, data_folder, mode="train", *args, **kwargs):
        """
        mode: "train" | "val" | "test"
        """
        self.data_folder = data_folder
        self.mode = mode
        super().__init__(*args, **kwargs)

    # -----------------------------
    # REQUIRED: image discovery
    # -----------------------------
    def get_img_files(self, img_path):
        """
        Return list[str] of image identifiers (paths or dummy ids)
        """
        files = []  # <-- fill
        return files

    # -----------------------------
    # REQUIRED: label metadata
    # -----------------------------
    def get_labels(self):
        """
        Returns list[dict], one per sample
        """
        labels = []
        for im_file in self.im_files:
            labels.append(
                {
                    "im_file": im_file,  # str (can be dummy)
                    "cls": np.zeros((0, 1), dtype=np.float32),  # (N,1)
                    "bboxes": np.zeros((0, 4), dtype=np.float32),  # (N,4)
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        return labels

    # -----------------------------
    # CORE: data loading
    # -----------------------------
    def __getitem__(self, index):
        """
        MUST return dict after transforms()
        """

        label = self.labels[index].copy()

        # -----------------------------
        # YOU FILL THESE
        # -----------------------------
        img = None  # numpy array (H, W, C)
        cls = None  # np.ndarray (N,1) float32
        bboxes = None  # np.ndarray (N,4) float32

        # -----------------------------
        # REQUIRED STRUCTURE
        # -----------------------------
        label["img"] = img
        label["cls"] = cls
        label["bboxes"] = bboxes

        # optional but commonly present
        label["ori_shape"] = None
        label["resized_shape"] = None
        label["ratio_pad"] = (1.0, 1.0)

        return self.transforms(label)

    # -----------------------------
    # AUGMENTATIONS
    # -----------------------------
    def build_transforms(self, hyp=None):
        """
        hyp: Ultralytics hyperparameters object
        """
        transforms = super().build_transforms(hyp)

        if self.mode == "train":
            # insert your train augmentations
            pass
        else:
            # validation / test transforms
            pass

        return transforms

    # -----------------------------
    # OPTIONAL: custom batching
    # -----------------------------
    @staticmethod
    def collate_fn(batch):
        """
        Only override if you change output structure
        """
        return YOLODataset.collate_fn(batch)


# %%
if __name__ == "__main__":
    from fran.configs.parser import ConfigMaker
    from fran.localiser.helpers import draw_image_bbox
    from fran.localiser.transforms.tsl import TSLRegions

# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------

    P = Project(project_title="totalseg")
    C = ConfigMaker(P)

# %%

    T = TSLRegions()
    names = T.regions
    names_ap = [region + "_ap" for region in names]
    names_lat = [region + "_lat" for region in names]
    names = names_ap + names_lat
    names_dici = {x: i for i, x in enumerate(names)}
    nc = len(names)
    data = {"names": names_dici, "nc": nc}

# %%
    data_folder = "/s/tmp/nii2pt_tsl3d_debug"
    batch = 4
    mode = "train"
    img_size = 256
    classes = 10
    hyp = DEFAULT_CFG
# %%
    C = CTAugYOLODataset3D(
        project=P,
        configs=C,
        data_folder=data_folder,
        mode=mode,
        imgsz=img_size,
        batch_size=batch,
        hyp=hyp,
        rect=False,
        cache=None,
        single_cls=False,
        stride=1,
        pad=0.0,
        prefix=colorstr(f"{mode}: "),
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
    )
# %%
    #
    # n = 6
    # dat  = C[n]
    # dici = C.data_dicts[n]
    # img = dќici['img']
    # lm =dici  dici['lm']
    #
    #
    keys_tr = "LT,ToBinary,Et,Et2,WinR,P1,P2,IntAugs,Flip1,Flip2,Zoom,Resize2D,N255,LM2YoloBBox,DelI"
# %%
    img.shape
    lm.shape

# %%
    index = 0
    dici_out = C.monai_tfms(dici)
    labels = C.get_image_and_label_from_dict(index, dici_out)
    out = []
    for label in labels:
        out.append(C.transforms(label))

    dat = out[0]
# %%
    img = dat["img"]
    bbx = dat["bboxes"]
    n = 0
    bbo = bbx[n]
# %%
    draw_image_bbox(
        img=img[1],
        start_x=bbo[0],
        start_y=bbo[1],
        stop_x=bbo[2],
        stop_y=bbo[3],
    )

# %%
    draw_image_lm_bbox(
        img=input_tensor[0],
        lm=lm[n],
        start_x=bbo[0],
        start_y=bbo[1],
        stop_x=bbo[2],
        stop_y=bbo[3],
    )
# %%

    lmfn = Path("/tmp/lm.pt")
    torch.save(lm, lmfn)
    dici2["bbox1_yolo"]


# %%

