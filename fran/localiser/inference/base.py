import logging
import math
from pathlib import Path

import ipdb
import psutil
import torch
from fastcore.all import listify
from fran.localiser.helpers import draw_tensor_boxes
from fran.localiser.yolo_ct_augment import apply_window_tensor
from fran.transforms.imageio import LoadSITKd
from fran.transforms.spatialtransforms import Project2D
from lightning.fabric import Fabric
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import SpatialPadd
from monai.transforms.spatial.dictionary import Orientationd, Resized, Spacingd
from monai.transforms.transform import MapTransform
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToDeviced
from torch.nn.functional import interpolate
from tqdm.auto import tqdm
from utilz.fileio import maybe_makedirs
from utilz.stringz import headline, strip_extension

tr = ipdb.set_trace


def bboxes_combine(bbo_ap, bbo_lat) -> dict:
    height_start = min(bbo_ap[1], bbo_lat[1])
    height_end = max(bbo_ap[3], bbo_lat[3])
    height = (height_start, height_end)
    slice_props = {
        "width": (bbo_ap[0], bbo_ap[2]),
        "ap": (bbo_lat[0], bbo_lat[2]),
        "height": height,
    }

    return slice_props


def crop_to_bbox(img, slice_props):
    assert img.ndim == 3, f"Image must be 3D, got {img.ndim()}"
    shp3d = img.shape
    height3d = shp3d[2]
    ap3d = shp3d[0]
    width3d = shp3d[1]
    wd = slice_props["width"]
    height = slice_props["height"]
    ap = slice_props["ap"]

    slc_width = slice(math.floor(wd[0] * width3d), math.ceil(wd[1] * width3d))
    slc_height = slice(int(height[0] * height3d), int(height[1] * height3d))
    slc_ap = slice(int(ap[0] * ap3d), int(ap[1] * ap3d))

    img_cropped = img[slc_width, slc_ap, slc_height]
    return img_cropped


def list_to_chunks(items, chunksize):
    return [items[i : i + chunksize] for i in range(0, len(items), chunksize)]


def load_images_nifti(images):
    loader = LoadSITKd(keys=["image"], image_only=True)
    return [loader({"image": image}) for image in listify(images)]


def largest_bbox(box, proj, classes, tol=1e-2):
    pads3tup = proj["letterbox_padded"]
    cls_box = [c for c in classes if c in box.cls]
    # inds = [box.cls == c for c in cls_box]
    want = torch.tensor(cls_box, device=box.cls.device)
    inds = []
    for c in want:
        m = box.cls == c
        if m.any():
            ii = torch.nonzero(m, as_tuple=True)[0]
            inds.append(ii[box.conf[ii].argmax()])

    inds = (
        torch.stack(inds)
        if inds
        else torch.empty(0, dtype=torch.long, device=box.cls.device)
    )
    xyxyn = box.xyxyn[inds]
    shp = box.orig_shape
    pads = pads3tup[1:]
    props = torch.zeros(4, device=xyxyn.device)
    assert len(pads) == 2, "Only works for 2D"
    pads_yolo = (pads[1], pads[0])
    for ind, pad_pair in enumerate(pads_yolo):
        props[ind * 1] = -(pad_pair[0] / shp[ind])
        props[ind * 1 + 2] = pad_pair[1] / shp[ind]
    xyxy_adjust = xyxyn + props
    xyxy_before_bounding = xyxy_adjust.detach().cpu().tolist()
    out_of_bounds = bool(
        ((xyxy_adjust < 0) | (xyxy_adjust > 1)).any().detach().cpu().item()
    )
    xyxy_bounded = xyxy_adjust.clamp(0, 1)
    starts, _ = xyxy_bounded[:, :2].min(dim=0)
    stops, _ = xyxy_bounded[:, 2:].max(dim=0)
    xyxy_unified_padcorrected = torch.cat([starts, stops])
    xyxy_unified_padcorrected = xyxy_unified_padcorrected.detach().tolist()
    message = "OK"
    if out_of_bounds:
        message = (
            "WARNING: tensor contains values outside [0, 1]; "
            f"xyxy_before_bounding={xyxy_before_bounding}"
        )
    return {
        "unified_bbox": xyxy_unified_padcorrected,
        "message": message,
    }


class WindowOned(MapTransform):
    def __init__(self, keys, window="a"):
        super().__init__(keys)
        self.window = window

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key].float()
            image = apply_window_tensor(image, self.window)
            d[key] = image
        return d


class CloneKeyd(MapTransform):
    def __init__(self, keys, output_keys):
        super().__init__(keys)
        self.output_keys = listify(output_keys)

    def __call__(self, data):
        d = dict(data)
        for key, output_key in zip(self.keys, self.output_keys):
            d[output_key] = d[key].clone()
        return d


class PermuteFlip2Dd(MapTransform):
    def __init__(self, keys, permute=(0, 2, 1), flip_direction="vertical"):
        super().__init__(keys)
        self.permute = permute
        self.flip_dims = {
            "vertical": (1,),
            "height": (1,),
            "h": (1,),
            "y": (1,),
            "updown": (1,),
            "horizontal": (2,),
            "width": (2,),
            "w": (2,),
            "x": (2,),
            "leftright": (2,),
        }[flip_direction.lower()]

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key].permute(*self.permute)
            d[key] = torch.flip(image, dims=self.flip_dims).contiguous()
        return d


class Resize2Dd(MapTransform):
    def __init__(self, keys, spatial_size=(256, 256), mode="bilinear"):
        super().__init__(keys)
        self.spatial_size = spatial_size
        self.mode = mode

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.resize_image(d[key])
        return d

    def resize_image(self, image):
        image = image.float().unsqueeze(0)
        kwargs = {}
        if self.mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        return interpolate(
            image,
            size=self.spatial_size,
            mode=self.mode,
            **kwargs,
        ).squeeze(0)


class NormaliseZeroToOned(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalise_image(d[key])
        return d

    def normalise_image(self, image):
        image = image.float()
        image = image - image.min()
        denom = image.max()
        if denom > 0:
            image = image / denom
        return image


class RepeatChannelsd(MapTransform):
    def __init__(self, keys, channels=3):
        super().__init__(keys)
        self.channels = channels

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.repeat_image(d[key]).contiguous()
        return d

    def repeat_image(self, image):
        return image.repeat(self.channels, 1, 1)


def collate_projections(batch):
    images = []
    projection_batches = {"image1": [], "image2": []}
    image_orig = []
    projection_meta = []
    for case_index, item in enumerate(batch):
        image_orig.append(item["image_orig"])
        for projection_index, key in enumerate(("image1", "image2")):
            images.append(item[key])
            projection_batches[key].append(item[key])
            meta = dict(item[key].meta)
            meta.update(letterbox_meta(item[key]))
            meta["case_index"] = case_index
            meta["projection_index"] = projection_index
            meta["projection_key"] = key
            projection_meta.append(meta)
    return {
        "image": torch.cat(images, dim=0),
        "image1": torch.cat(projection_batches["image1"], dim=0),
        "image2": torch.cat(projection_batches["image2"], dim=0),
        "image_orig": image_orig,
        "projection_meta": projection_meta,
    }


def letterbox_meta(image):
    spatial_pad = [
        op for op in image.applied_operations if op["class"] == "SpatialPad"
    ][-1]
    resize = [op for op in image.applied_operations if op["class"] == "Resize"][-1]
    return {
        "letterbox_padded": spatial_pad["extra_info"]["padded"],
        "letterbox_orig_size": resize["orig_size"],
        "letterbox_resized_size": spatial_pad["orig_size"],
    }


def yolo_bbox_to_slices(
    bbox, spatial_shape, padding=None, padded_shape=None, resized_shape=None
):
    spatial_shape = torch.as_tensor(spatial_shape, dtype=torch.float64)
    bbox = torch.as_tensor(bbox, dtype=torch.float64)
    centers = bbox[[1, 0]]
    sizes = bbox[[3, 2]]
    n_dims = 2

    if padding is not None:
        padding = torch.as_tensor(padding, dtype=torch.float64)[-n_dims:]
        pad_before = padding[:, 0]
        pad_total = padding.sum(dim=1)
        if padded_shape is None:
            padded_shape = (
                torch.as_tensor(resized_shape, dtype=torch.float64) + pad_total
            )
        else:
            padded_shape = torch.as_tensor(padded_shape, dtype=torch.float64)
        resized_shape = padded_shape - pad_total
        centers = (centers * padded_shape - pad_before) / resized_shape
        sizes = sizes * padded_shape / resized_shape

    eps = 1e-6
    starts = torch.floor((centers - sizes / 2) * spatial_shape + eps).long()
    stops = torch.ceil((centers + sizes / 2) * spatial_shape - eps).long()
    shape = spatial_shape.long()
    starts = torch.minimum(torch.maximum(starts, torch.zeros_like(starts)), shape - 1)
    stops = torch.minimum(torch.maximum(stops, starts + 1), shape)
    return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))


def yolo_bbox_to_3d_slices(
    bbox,
    spatial_shape_3d,
    projection_key,
    padding=None,
    padded_shape=None,
    resized_shape=None,
):
    x, y, z = spatial_shape_3d
    projection_key = projection_key.replace("image", "p")
    if projection_key in {"p1", "lat"}:
        z_slice, y_slice = yolo_bbox_to_slices(
            bbox,
            spatial_shape=(z, y),
            padding=padding,
            padded_shape=padded_shape,
            resized_shape=resized_shape,
        )
        return (slice(None), y_slice, _flip_slice(z_slice, z))

    z_slice, x_slice = yolo_bbox_to_slices(
        bbox,
        spatial_shape=(z, x),
        padding=padding,
        padded_shape=padded_shape,
        resized_shape=resized_shape,
    )
    return (x_slice, slice(None), _flip_slice(z_slice, z))


def _flip_slice(slice_, size):
    return slice(size - slice_.stop, size - slice_.start)


class LocaliserInferer:
    _bbox_warning_loggers = {}

    def __init__(
        self,
        model,
        classes,
        imsize=256,
        window="a",
        projection_dim=(1, 2),
        batch_size=None,
        device=0,
        debug=False,
        out_folder=None,
        warnings_filename="warnings.log",
        save_jpg=True,
        letterbox=True,
        keys_preproc="E,O,Orig,W,S,P1,P2,PF,R,Rep,E2,N",
        precision="bf16-mixed",
        accelerator=None,
        fabric_kwargs=None,
        mem_quota=0.5,  # system ram
    ):
        self.classes = classes
        self.model = model
        self.imsize = imsize
        self.window = window
        self.projection_dim = projection_dim
        self.batch_size = batch_size
        self.device = device
        self.debug = debug
        self.out_folder = Path(out_folder) if out_folder is not None else None
        self.warnings_filename = warnings_filename
        self.save_jpg = save_jpg
        self.letterbox = letterbox
        self.keys_preproc = keys_preproc
        self.projections_dims = {"lat": 1, "ap": 2, "ax": 3}
        self.image_key = "image"
        self.image_orig_key = "image_orig"
        self.image2d_keys = ["image1", "image2"]
        self.precision = precision
        self.accelerator = accelerator
        self.fabric_kwargs = {} if fabric_kwargs is None else dict(fabric_kwargs)
        self.mem_quota = mem_quota
        self.bbox_warning_logger = self.setup_bbox_warning_logger()
        self.setup_fabric()
        self.setup_model()
        self.create_and_set_preprocess_transforms()

    def setup_bbox_warning_logger(self):
        if self.out_folder is None:
            return None
        self.out_folder.mkdir(parents=True, exist_ok=True)
        log_path = self.out_folder / self.warnings_filename
        log_key = str(log_path)
        if log_key in self._bbox_warning_loggers:
            return self._bbox_warning_loggers[log_key]

        logger = logging.getLogger(
            f"{__name__}.LocaliserInferer.bbox_warning.{len(self._bbox_warning_loggers)}"
        )
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        handler = logging.FileHandler(log_path, mode="a")
        handler.setLevel(logging.WARNING)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        self._bbox_warning_loggers[log_key] = logger
        return logger

    def setup_fabric(self):
        accelerator = self.accelerator or "gpu"
        devices = 1 if accelerator == "cpu" else [self.device]
        self.fabric = Fabric(
            accelerator=accelerator,
            devices=devices,
            precision=self.precision,
            **self.fabric_kwargs,
        )
        self.fabric_device = self.fabric.device

    def setup_model(self):
        self.model.to(self.fabric_device)
        self.model.eval()

    def create_and_set_preprocess_transforms(self):
        self.create_preprocess_transforms()
        self.set_preprocess_transforms()

    def create_preprocess_transforms(self):
        self.preprocess_transforms_dict = {
            "O": Orientationd(keys=[self.image_key], axcodes="RAS", labels=None),
            "E": EnsureChannelFirstd(keys=[self.image_key], channel_dim="no_channel"),
            "Dev": ToDeviced(keys=[self.image_key], device=self.fabric_device),
            "Orig": CloneKeyd(
                keys=[self.image_key],
                output_keys=[self.image_orig_key],
            ),
            "W": WindowOned(
                keys=[self.image_key],
                window=self.window,
            ),
            "S": Spacingd(
                keys=[self.image_key],
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
            ),
            "PF": PermuteFlip2Dd(keys=self.image2d_keys),
            "E2": EnsureChannelFirstd(keys=self.image2d_keys, channel_dim="no_channel"),
            "R": self.create_resize_transform(),
            "N": NormaliseZeroToOned(keys=self.image2d_keys),
            "Rep": RepeatChannelsd(keys=self.image2d_keys, channels=3),
            "P1": Project2D(
                keys=[self.image_key],
                operations=["mean"],
                dim=self.projections_dims["lat"],
                suffix="lat",
                output_keys=["image1"],
            ),
            "P2": Project2D(
                keys=[self.image_key],
                operations=["mean"],
                dim=self.projections_dims["ap"],
                suffix="ap",
                output_keys=["image2"],
            ),
        }

    def create_resize_transform(self):
        if self.letterbox:
            return Compose(
                [
                    Resized(
                        keys=self.image2d_keys,
                        spatial_size=self.imsize,
                        size_mode="longest",
                        mode="bilinear",
                    ),
                    SpatialPadd(
                        keys=self.image2d_keys,
                        spatial_size=(self.imsize, self.imsize),
                        method="symmetric",
                    ),
                ]
            )
        return Resize2Dd(
            keys=self.image2d_keys,
            spatial_size=(self.imsize, self.imsize),
        )

    def set_preprocess_transforms(self):
        self.preprocess_transforms = self.tfms_from_dict(
            self.keys_preproc, self.preprocess_transforms_dict
        )
        self.preprocess_compose = Compose(self.preprocess_transforms)

    def preprocess_iterate(self, data):
        for tfm in self.preprocess_transforms:
            headline(tfm)
            tr()
            data = tfm(data)
        return data

    def tfms_from_dict(self, keys, transforms_dict):
        keys = keys.replace(" ", "").split(",")
        return [transforms_dict[key] for key in keys]

    def load_images(self, images):
        return load_images_nifti(images)

    def prepare_data(self, data):
        transform = self.preprocess_iterate if self.debug else self.preprocess_compose
        self.ds = Dataset(data=data, transform=transform)
        batch_size = self.batch_size or len(data)
        self.pred_dl = DataLoader(
            self.ds,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=collate_projections,
        )
        self.pred_dl = self.fabric.setup_dataloaders(
            self.pred_dl,
            move_to_device=False,
        )

    def run(self, images, chunksize=64):
        if self.out_folder is not None:
            maybe_makedirs(self.out_folder)
        outputs = []
        image_chunks = list_to_chunks(listify(images), chunksize)
        for image_chunk in tqdm(image_chunks, desc="Localiser inference"):
            if self.system_mem_remaining() < self.mem_quota:
                self.delete_image_orig(outputs)
            chunk_outputs = self.process_data_sublist(image_chunk)
            if self.system_mem_remaining() < self.mem_quota:
                self.delete_image_orig(outputs)
            outputs.extend(chunk_outputs)
        return outputs

    def process_data_sublist(self, images):
        data = self.load_images(images)
        self.prepare_data(data)
        outputs = []
        for batch in self.predict():
            for outs in self.package_preds(batch):
                self.standardize_bboxes(outs)
                if self.system_mem_remaining() < self.mem_quota:
                    self.delete_image_orig(outputs)
                outs = self.postprocess(outs)
                self.save_bboxes_final(outs)
                outputs.append(outs)
        return outputs

    def standardize_bboxes(self, outs):
        bboxes = outs["pred"]
        projm = outs["projection_meta"]
        bbo_lat = largest_bbox(bboxes[0].boxes, projm[0], self.classes)
        bbo_ap = largest_bbox(bboxes[1].boxes, projm[1], self.classes)
        bb_fixed = bboxes_combine(
            bbo_ap["unified_bbox"],
            bbo_lat["unified_bbox"],
        )
        outs["bboxes_final"] = bb_fixed
        outs["bbox_messages"] = {
            "lat": bbo_lat["message"],
            "ap": bbo_ap["message"],
        }
        if self.bbox_warning_logger is not None:
            src = Path(projm[0]["filename_or_obj"]).name
            for projection, message in outs["bbox_messages"].items():
                if message != "OK":
                    self.bbox_warning_logger.warning(
                        "%s %s: %s", src, projection, message
                    )

    def postprocess(self, out):
        out["image"] = out["image"].detach().cpu()
        out["image1"] = out["image1"].detach().cpu()
        out["image2"] = out["image2"].detach().cpu()
        out[self.image_orig_key] = out[self.image_orig_key].detach().cpu()
        out["pred"] = [pred.cpu() for pred in out["pred"]]
        out["pred_image1"] = out["pred_image1"].cpu()
        out["pred_image2"] = out["pred_image2"].cpu()
        return out

    def system_mem_remaining(self):
        mem = psutil.virtual_memory()
        return mem.available / mem.total

    def delete_image_orig(self, outputs):
        for out in outputs:
            out.pop(self.image_orig_key, None)

    def package_preds(self, batch):
        preds = batch["pred"]
        projection_meta = batch["projection_meta"]

        outputs = []
        for case_index, pred_index in enumerate(range(0, len(preds), 2)):
            pred_image1 = preds[pred_index]
            pred_image2 = preds[pred_index + 1]
            meta_image1 = projection_meta[pred_index]
            meta_image2 = projection_meta[pred_index + 1]

            outputs.append(
                {
                    "pred": [pred_image1, pred_image2],
                    "pred_image1": pred_image1,
                    "pred_image2": pred_image2,
                    "image": batch["image"][pred_index : pred_index + 2],
                    "image1": batch["image1"][case_index : case_index + 1],
                    "image2": batch["image2"][case_index : case_index + 1],
                    "image_orig": batch["image_orig"][case_index],
                    "projection_meta": [meta_image1, meta_image2],
                    "projection_meta_image1": meta_image1,
                    "projection_meta_image2": meta_image2,
                }
            )
        return outputs

    def predict(self):
        with torch.inference_mode():
            for batch in self.pred_dl:
                image = batch["image"].float().to(self.fabric_device, non_blocking=True)
                with self.fabric.autocast():
                    pred = self.model(image, verbose=False)
                if self.save_jpg:
                    self.save_prediction_images(image, pred, batch["projection_meta"])
                pred = [p.cpu() for p in pred]
                batch["pred"] = pred
                yield batch

    def save_prediction_images(self, images, preds, projection_meta):
        if self.out_folder is None:
            raise ValueError("out_folder is required when save_jpg=True")
        for index, (image, pred, meta) in enumerate(
            zip(images, preds, projection_meta)
        ):
            src = strip_extension(Path(meta["filename_or_obj"]).name)
            projection = meta["projection_key"].replace("image", "p")
            filename = self.out_folder / f"{src}_{index:03d}_{projection}.jpg"
            draw_tensor_boxes(image[None], pred, filename=filename, show=False)

    def save_bboxes_final(self, out):
        if self.out_folder is None:
            raise ValueError("out_folder is required to save bboxes_final")
        filename = out["projection_meta"][0]["filename_or_obj"]
        src = strip_extension(Path(filename).name)
        filename = self.out_folder / f"{src}.txt"
        bbox = out["bboxes_final"]
        header= "filename: {filename}\n"
        text =header + "\n".join(f"{key}: {value[0]} {value[1]}" for key, value in bbox.items())
        filename.write_text(text + "\n")


# %%
# SECTION:--------------------  setup--------------------------------------------------------------------------------------
if __name__ == "__main__":
    import SimpleITK as sitk
    from fran.localiser.helpers import (
        jpg_to_tensor,
        make_multiwindow_inference_tensor,
        make_singlewindow_inference_tensor,
    )
    from ultralytics import YOLO
    from utilz.stringz import info_from_filename

    model = YOLO(
        "/s/fran_storage/yolo_output/totalseg_localiser/train32/weights/best.pt"
    )

# %%
    data_folder = Path("/media/UB/datasets/kits23/")
    imgs_fldr = data_folder / "images"
    out_fldr = data_folder / "loc2d"
    nii_path = Path("/media/UB/datasets/kits23/images/kits23_00412.nii.gz")
    assert nii_path.exists(), f"{nii_path} does not exist"

    classes = [0, 2, 3, 5]
    M = LocaliserInferer(
        model,
        classes,
        imsize=256,
        window="a",
        projection_dim=(1, 2),
        out_folder=out_fldr,
        batch_size=64)
# %%

    imgs = [nii_path]
    imgs = list(imgs_fldr.glob("*"))
    dones = list(out_fldr.glob("*.txt"))

    cids_dont = [
        info_from_filename(fn.name, full_caseid=True)["case_id"] for fn in dones
    ]
    imgs2 = [
        img
        for img in imgs
        if info_from_filename(img.name, full_caseid=True)["case_id"] not in cids_dont
    ]


    len(cids_dont)
# %%
    M.run(imgs2)
# %%
    images = imgs

    outs = M.process_data_sublist(images)
    print(outs.keys())
    outs["projection_meta"]
    outs["image"].meta
    # outs['image_orig'][0].meta
    outs["pred"]
# %%
    data = M.load_images(images)
    M.prepare_data(data)
    outs = next(M.predict())
    outs2 = M.package_preds(outs)
    # n= 2

    M.standardize_bboxes(outs)
    out = outs
    bboxes = outs["pred"]
    projm = outs["projection_meta"]
    bb0 = bboxes[0].boxes
    bb1 = bboxes[1].boxes
    bbo_lat = largest_bbox(bb0, projm[0], M.classes)
    bbo_ap = largest_bbox(bb1, projm[1], M.classes)
    bb_fixed = bboxes_combine(bbo_ap, bbo_lat)
    out["bboxes_final"] = bb_fixed

# %%

    box = bb0
    proj = projm[0]
    tol = 1e-2
    pads3tup = proj["letterbox_padded"]
    cls_box = [c for c in classes if c in box.cls]
    # inds = [box.cls == c for c in cls_box]
# %%
    want = torch.tensor(cls_box, device=box.cls.device)
    inds = []
    for c in want:
        m = box.cls == c
        if m.any():
            ii = torch.nonzero(m, as_tuple=True)[0]
            inds.append(ii[box.conf[ii].argmax()])
# %%

    inds = (
        torch.stack(inds)
        if inds
        else torch.empty(0, dtype=torch.long, device=box.cls.device)
    )
    xyxyn = box.xyxyn[inds]
    shp = box.orig_shape
    pads = pads3tup[1:]
    props = torch.zeros(4, device=xyxyn.device)
    assert len(pads) == 2, "Only works for 2D"
    pads_yolo = (pads[1], pads[0])
# %%
    for ind, pad_pair in enumerate(pads_yolo):
        props[ind * 1] = -(pad_pair[0] / shp[ind])
        props[ind * 1 + 2] = pad_pair[1] / shp[ind]
# %%
    xyxy_adjust = xyxyn + props

    xyxy_bounded = xyxy_adjust.clone()

    xyxy_bounded[(xyxy_bounded > 1) & (xyxy_bounded <= 1 + tol)] = 1
    xyxy_bounded[(xyxy_bounded < 0) & (xyxy_bounded >= -tol)] = 0

    if ((xyxy_bounded < 0) | (xyxy_bounded > 1)).any():
        raise ValueError("tensor contains values outside [0, 1] beyond tolerance")
    starts, _ = xyxy_bounded[:, :2].min(dim=0)
    stops, _ = xyxy_bounded[:, 2:].max(dim=0)
    xyxy_unified_padcorrected = torch.cat([starts, stops])
    xyxy_unified_padcorrected = xyxy_unified_padcorrected.detach().tolist()
    # return xyxy_unified_padcorrected

# %%
    for x in range(len(M.ds)):
        dat = M.ds[x]

# %%

    dici = outs
    img = dici["image_orig"]
    bbo = dici["bboxes_final"]
    im = img[0]
    im2 = crop_to_bbox(im, bbo)
    im3 = im2.permute(2, 0, 1)
    ImageMaskViewer([im3, im3], "ii")

# %%
    im1 = dici["image1"]
    im2 = dici["image2"]
    preds = dici["pred"]
    projm = dici["projection_meta"]
    bbox = preds

    ims = [im1, im2]
    x = 0
    draw_tensor_boxes(ims[x], bbox[x])
    y = 1
    draw_tensor_boxes(ims[y], bbox[y])
# %%

    image = dici["image_orig"]

# %%
    def show(dici):
        def _imi(im):
            print(im.shape)
            print(f"Max: {im.max()}")
            print(f"Min: {im.min()}")

        print(dici.keys())
        img = dici["image"]
        _imi(img)

# %%
    proj = projm[n].copy()
    tol = 1e-3

    pred1 = outs2[n]["pred_image1"]
    pred2 = outs2[n]["pred_image2"]
    projm = outs2[n]["projection_meta"]
# %%
    imn = 1
    proj = projm[imn]
    box = bbox[imn].boxes
# %%

    data = M.load_images(images[:3])
    dici = data[n]
    dici = M.preprocess_transforms_dict["E"](dici)
    dici = M.preprocess_transforms_dict["O"](dici)

    img = dici["image"]
    img.shape
# %%
    slice_props = bboxes_combine(bbo_ap, bbo_lat)
# %%
    img_cropped = img_cropped.permute(2, 0, 1)
    ImageMaskViewer([img_cropped, img_cropped])
# %%
    im2 = img_cropped.mean(dim=1)
    im3 = img_cropped.mean(dim=2)

    import matplotlib.pyplot as plt

    plt.imshow(im3)
    plt.show()

# %%

# %%
    for sh, pp, xy in zip(shp, pads, xyxyn):
        prop_bef = pp[0] / sh
        prop_after = pp[1] / sh
        props.append((-prop_bef, prop_after))

# %%
    # xs = xyxyn[]

# %%
    slices = []
    for pp in pads:
        slices.append(slice(int(pp[0]), int(pp[1])))
# %%
    im1p = im1[:, :, pads]
    im1p = im1[:, :, 58 : 256 - 58, :]
    im2p = im2[:, :, pads]
    im2p = im2[:, :, 58 : 256 - 58, :]
    draw_tensor_boxes(im1p, bbox[0])
    draw_tensor_boxes(im2p, bbox[1])
# %%

    dici = M.preprocess_transforms_dict["W"](dici)
    show(dici)
    dici = M.preprocess_transforms_dict["P1"](dici)
    dici = M.preprocess_transforms_dict["P2"](dici)
    dici = M.preprocess_transforms_dict["PF"](dici)
    dici["image1"].shape
    R = Resize2Dd(keys=M.image2d_keys, spatial_size=(M.imsize, M.imsize))
    dici = R(dici)

    dici["image1"].shape
# %%
# %%
    Rep = RepeatChannelsd(keys=M.image2d_keys, channels=3)
    dici = Rep(dici)
    E2 = EnsureChannelFirstd(keys=M.image2d_keys, channel_dim="no_channel")
    dici = E2(dici)
    dici["image2"].max()
    dici["image2"].shape
    N = NormaliseZeroToOned(keys=M.image2d_keys)
    dici = N(dici)
    dici["image1"].max()
    dici["image1"].shape

# %%
    im1 = dici["image1"]
    im2 = dici["image2"]
    ressi = model(im1)
    ress2 = model(im2)
    rr = ressi[0]
    rr2 = ress2[0]
    draw_tensor_boxes(im1, rr)
    draw_tensor_boxes(im2, rr2)
# %%
    from matplotlib import pyplot as plt

    plt.imshow(dici["image1"][0])
    plt.show()
# %%
    rr = out[0]

    rr.keys()
    yol = rr["pred"]
    yol[0].boxes
    draw_tensor_boxes(rx4, yol[0])
# %%

# %%
    img = sitk.ReadImage(nii_path)

    arr = sitk.GetArrayFromImage(img)
    imsize = 256
    raw_x = torch.from_numpy(arr).float().unsqueeze(0)
    rx_win = apply_window_tensor(raw_x, "a")
    rx2 = torch.mean(rx_win, 1)
    rx2_perm = rx2.permute(0, 2, 1)
    rx4 = rx2.unsqueeze(0)
    rx4 = rx4.permute(0, 1, 3, 2)
    rx4 = interpolate(rx4, (imsize, imsize))
    min_ = rx4.min()
    max_ = rx4.max()
    rx4 = (rx4 - min_) / (max_ - min_)

    # min_t = rx2.min()
    # max_t = rx2.max()
    # rx3 = (rx2-min_t)/(max_t-min_t)
    # rx4 = rx3.unsqueeze(0)
    rx4 = rx4.repeat(1, 3, 1, 1)
    print(rx4.shape)
    plt.imshow(rx4[0, 0])
# %%
    res2 = model(rx4)
    rr2 = res2[0]
# %%

    from fran.localiser.helpers import draw_tensor_boxes

    draw_tensor_boxes(rx4, rr2)

# %%

# %%
# SECTION:-------------------- jpg--------------------------------------------------------------------------------------

    import torch

# %%
    fn = "/s/xnat_shadow/totalseg2d/jpg/valid/images/totalseg_s0029_a2.jpg"
    img = jpg_to_tensor(fn)
    img_torch = img.permute(2, 0, 1).float()
    img_torch = img_torch / img_torch.max()
    img_torch = img_torch.unsqueeze(0)
    im2 = interpolate(img_torch, (imsize, imsize))

    im3 = torch.flip(im2, dims=[2, 3])
    res = model(im3)
    rr = res[0]
    # draw_tensor_boxes(im3, rr)

# %%

# %%

# SECTION:-------------------- 3window--------------------------------------------------------------------------------------
    xyxy_bounded = make_multiwindow_inference_tensor(raw_x)

    res = model(xyxy_bounded)

# %%

# %%
# SECTION:-------------------- one window--------------------------------------------------------------------------------------
    x2 = make_singlewindow_inference_tensor(raw_x, "b")

# %%
    res = model(x2)

# %%
    rr = res[0]
    imtnsr = x2
# %%


