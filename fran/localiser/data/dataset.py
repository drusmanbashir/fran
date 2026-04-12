import random
from pathlib import Path

from pathlib import Path

from ultralytics.data.dataset import YOLODataset
import numpy as np
import ipdb
from matplotlib.transforms import Transform
import torch
from fran.configs.parser import ConfigMaker
from fran.localiser.helpers import draw_image_lm_bbox, show_images_with_boxes
from fran.transforms.imageio import LoadTorchd
from monai.data.dataset import Dataset
from torch.utils.data import random_split
from torchvision.utils import save_image
from tqdm.auto import tqdm

from utilz.stringz import info_from_filename

tr = ipdb.set_trace

import cv2
from monai.transforms.transform import MapTransform, RandomizableTransform
import numpy as np
import torch
from fran.transforms.spatialtransforms import Project2D
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import de_parallel

from utilz.imageviewers import ImageMaskViewer


class Fix2DBBox(Transform):
    def __init__(self, bbox_keys):
        self.bbox_keys = bbox_keys

    def __call__(self, data):
        for key in self.bbox_keys:
                data[key] = self.yyxx_to_xxyy(data[key])
        return data


    def yyxx_to_xxyy(self,bb):
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

    def build_dataset(self, img_path, mode="train", batch=None):
        if self.use_3d_pt_dataset(img_path):
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
            return CTAugYOLODataset3D(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
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
            labels.append({
                "im_file": im_file,                         # str (can be dummy)
                "cls": np.zeros((0, 1), dtype=np.float32),  # (N,1)
                "bboxes": np.zeros((0, 4), dtype=np.float32),  # (N,4)
                "normalized": True,
                "bbox_format": "xywh",
            })
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
        img = None        # numpy array (H, W, C)
        cls = None        # np.ndarray (N,1) float32
        bboxes = None     # np.ndarray (N,4) float32

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
    import lightning as L
    from fran.localiser.transforms import NormaliseZeroToOne
    from fran.transforms.intensitytransforms import MakeBinary
    from fran.transforms.misc_transforms import BoundingBoxYOLOd
    from monai.apps.detection.transforms.dictionary import ConvertBoxToStandardModed
    from monai.transforms import Compose
    from monai.transforms.croppad.dictionary import BoundingRectd
    from monai.transforms.spatial.dictionary import RandFlipd, RandRotated, RandZoomd, Resized
    from monai.transforms.utility.dictionary import DeleteItemsd, EnsureChannelFirstd, EnsureTyped
    from monai.transforms.compose import Compose
    from fran.localiser.transforms import MultiRemapsTSL, NormaliseZeroToOne
    from fran.localiser.transforms.transforms import WindowTensor3ChanneldRand
    from fran.transforms.imageio import TorchReader
    from fran.transforms.intensitytransforms import MakeBinary, RandRandGaussianNoised
    from fran.transforms.misc_transforms import BoundingBoxesYOLOd
    from monai.apps.detection.transforms.dictionary import ConvertBoxToStandardModed
    from monai.transforms.croppad.dictionary import BoundingRectd
    from monai.transforms.intensity.dictionary import RandAdjustContrastd, RandScaleIntensityd, RandShiftIntensityd
    from monai.transforms.io.dictionary import LoadImaged
    from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
    from fran.managers.project import Project

    P = Project(project_title="totalseg")
    C = ConfigMaker(P)

# %%
    mode= "train"
    data_folder = "/s/tmp/nii2pt_tsl3d_debug"
    train,val = P.get_train_val_case_ids(fold=1)
    if mode == "train":
        case_ids = train

    elif mode == "val":
        case_ids = val
    else: 
        raise ValueError(f"Invalid mode {mode}")

    imgs_folder = Path(data_folder) / "images"
    img_fns = list(imgs_folder.glob("*.pt"))
    lms_folder = Path(data_folder) / "lms"
# %%
    img_fns_final = []
    for fn in img_fns:
        case_id = info_from_filename(fn.name,full_caseid=True)["case_id"]
        if case_id in train:
            img_fns_final.append(fn)

# %%

    img_fn = Path("/s/tmp/nii2pt_tsl3d_debug/images/totalseg_s1424.pt")
    lm_fn = Path("/s/tmp/nii2pt_tsl3d_debug/lms/totalseg_s1424.pt")
    img = torch.load(img_fn, weights_only=False)

# %%
    image_key = "image"
    lm_key = "lm"
    box_keys = ["lm1_bbox", "lm2_bbox"]
    max_output_size = (256, 256, 256)
    label_key = "cls"
    label_key = "label"

    imgs_proj_keys = ["image1", "image2"]
    lms_proj_keys= ["lm1", "lm2"]
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

# %%
    probs_intensity = 0.5
    probs_spatial = 0.3
    output_size = (256, 256)
    LT = LoadImaged(keys=[image_key, lm_key], reader=TorchReader)
    E = EnsureChannelFirstd(keys=[image_key])
    ToBinary = MakeBinary([label_key])
    Et = EnsureTyped(keys=[image_key], dtype=torch.float32)
    Et2 = EnsureTyped(keys=[label_key], dtype=torch.long)
    E2 = EnsureTyped(keys=[image_key], dtype=torch.float16)

    # YoloBbox = BoundingBoxYOLOd(
    #     [box_key], 2, key_template_tensor=lm_key, output_keys=["bbox_yolo"]
    # )
    YoloBboxes = BoundingBoxesYOLOd(
        keys = box_keys,
        dim=2,
        key_template_tensor=lms_proj_keys[0],
        output_keys=["bbox_yolo1", "bbox_yolo2"],
    )
    N = NormaliseZeroToOne(keys=[image_key])
    Remap = MultiRemapsTSL(lm_key=lm_key)
# %%
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
    Win = WindowTensor3ChanneldRand(image_key="image", prob=0.5, jitter=100)

    IntensityTfms = [
        RandScaleIntensityd(
            keys=["image1", "image2"], factors=int_augs["scale"], prob=probs_intensity
        ),
        RandRandGaussianNoised(
            keys=["image1", "image2"], std_limits=int_augs["noise"], prob=probs_intensity
        ),
        RandShiftIntensityd(
            keys=["image1", "image2"], offsets=int_augs["shift"], prob=probs_intensity
        ),
        RandAdjustContrastd(
            ["image1", "image2"], gamma=int_augs["contrast"], prob=probs_intensity
        ),
    ]
    IntAugs = Compose(IntensityTfms)
    # self.transforms_dict["IntensityTfms"] = IntensityTfms

    ToBinary = MakeBinary([lm_key])
    Et = EnsureTyped(keys=[image_key], dtype=torch.float32)
    Et2 = EnsureTyped(keys=[lm_key], dtype=torch.long)
    E2 = EnsureTyped(keys=[image_key], dtype=torch.float16)
    ExtractBbox = BoundingRectd(keys=lms_proj_keys) # returns y0, y1, x0, x1
    FixBB = Fix2DBBox(bbox_keys=box_keys)

    CB = ConvertBoxToStandardModed(mode="xxyy", box_keys=box_keys) #mode = current mode
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
        mode=["bilinear","bilinear", "nearest","nearest" ],
        lazy=False,
    )
    LM2YoloBBox = Compose([ExtractBbox, FixBB, CB, YoloBboxes])
    DelI = DeleteItemsd(keys=[lm_key])
    transforms_dict = {
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
        "Win": Win,
    }

    # keys_tr = "LT,ToBinary,Et,Et2,Win,P1,P2,IntAugs,Flip1,Flip2,Zoom,Resize2D"#
    keys_tr = "LT,ToBinary,Et,Et2,Win,P1,P2,IntAugs,Flip1,Flip2,Zoom,Resize2D,LM2YoloBBox,DelI"
    keys_val = "LT,ToBinary,Et,Et2,Win,P1,P2,Resize2D,LM2YoloBBox,DelI"
# %%
    dici2 = {"image": img_fn, "lm": lm_fn}
    for k in keys_tr.split(","):
        dici2 = transforms_dict[k](dici2)
        dici2.keys()

# %%
    dici2['bbox_yolo1']
    lm1bb = dici2['bbox_yolo1']
    valid1 = lm1bb.sum(axis=1) > 0
    N1 = lm1bb.shape[0]
    c1= torch.arange(N1).reshape(-1,1)
    out = torch.cat([c1,lm1bb], dim=1)
    cls_bbox1 = out[valid1]
# %%
    lm2bb = dici2['bbox_yolo2']
    valid2 = lm2bb.sum(axis=1) > 0
    N2 = lm2bb.shape[0]
    c2= torch.arange(N1,N1+N2).reshape(-1,1)
    out2 = torch.cat([c2,lm2bb], dim=1)
    cls_bbox2 = out2[valid2]

# %%
    dici = {"img": [dici2['image1'], dici2['image2']],
            "cls": [c1,c2],
            "bboxes": [lm1bb, lm2bb]
            }
# %%
    print(cls_bbox1)

# %%
    lm2bb = dici2['bbox_yolo2']
    valid2 = lm2bb.sum(axis=1) > 0
    N2 = lm2bb.shape[0]
    c2= np.arange(N1,N1+N2).reshape(-1,1).astype(np.float32)
    out2 = np.hstack([c2,lm2bb])
    cls_bbox2 = out2[valid2]
# %%
                                           



    valid = (lm_for_cls.sum(axis=1) > 0)
    cls_out = torch.nonzero(torch.tensor(valid)).float()


    






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
    dici2['bbox1_yolo']


# %%
