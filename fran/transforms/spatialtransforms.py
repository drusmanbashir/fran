# %%
import csv
import fcntl
import logging
import math
import os
from copy import deepcopy
from pathlib import Path
import ipdb
import monai.transforms.spatial.functional as fm
import nibabel as nib
import skimage.transform as tf
import torch.nn.functional as F
from monai.config.type_definitions import KeysCollection, SequenceStr
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.croppad.array import CropForeground, SpatialPad
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import LazyTransform, MapTransform, Randomizable, RandomizableTransform
from monai.utils.enums import LazyAttr, Method, PytorchPadMode, TraceKeys
from torch import cos, pi, sin
from monai.transforms.croppad.dictionary import Padd, RandSpatialCropd
from fran.transforms.base import ItemTransform, KeepBBoxTransform, MapTransform, MonaiDictTransform, Union, np, torch
from fran.utils.affine import spacing_from_affine
from utilz.helpers import load_dict, tr
from utilz.image_utils import margin_mm_to_vox
from utilz.stringz import int_to_str


def _resize3d(data, spatial_shape, mode):
    data_out = fm.resize(
        img=data,
        out_size=spatial_shape,
        mode=mode,
        lazy=False,
        align_corners=None,
        dtype=None,
        input_ndim=3,
        anti_aliasing=False,
        anti_aliasing_sigma=0.0,
        transform_info=None,
    )
    return data_out


class CropMaybePad(MapTransform):
    """
    Crop a channel-first 4D tensor by voxel bbox after optional mm-margin expansion,
    then zero-pad to min_shape.
    """

    def __init__(self, keys, min_shape, margin=0, lazy=False):
        MapTransform.__init__(self, keys)
        self.min_shape = tuple(int(v) for v in min_shape)
        self.min_shape_np = np.asarray(self.min_shape, dtype=int)
        self.margin = margin
        self.lazy = lazy

    def spatial_shape(self, img):
        return tuple(int(v) for v in img.shape[1:])

    def add_margin_to_bbox(self, box_start, box_end, image_shape, spacing):
        box_start = np.asarray(box_start, dtype=int)
        box_end = np.asarray(box_end, dtype=int)
        image_shape = np.asarray(image_shape, dtype=int)
        margin = margin_mm_to_vox(self.margin, spacing)
        box_start = np.maximum(box_start - margin, 0)
        box_end = np.minimum(box_end + margin, image_shape)
        return tuple(int(v) for v in box_start), tuple(int(v) for v in box_end)

    def maybe_expand_bbox(self, box_start, box_end, image_shape):
        box_start = np.asarray(box_start, dtype=int)
        box_end = np.asarray(box_end, dtype=int)
        image_shape = np.asarray(image_shape, dtype=int)
        box_shape = box_end - box_start
        target_shape = np.minimum(np.maximum(box_shape, self.min_shape_np), image_shape)
        center = (box_start + box_end) / 2.0

        new_start = np.floor(center - target_shape / 2.0).astype(int)
        new_end = new_start + target_shape

        shift_right = np.minimum(new_start, 0)
        new_start = new_start - shift_right
        new_end = new_end - shift_right

        shift_left = np.maximum(new_end - image_shape, 0)
        new_start = new_start - shift_left
        new_end = new_end - shift_left

        new_start = np.maximum(new_start, 0)
        new_end = np.minimum(new_end, image_shape)
        return tuple(int(v) for v in new_start), tuple(int(v) for v in new_end)

    def crop(self, img, box_start, box_end):
        return img[
            :,
            int(box_start[0]) : int(box_end[0]),
            int(box_start[1]) : int(box_end[1]),
            int(box_start[2]) : int(box_end[2]),
        ]

    def maybe_pad(self, img):
        shape = np.asarray(self.spatial_shape(img), dtype=int)
        deficits = np.maximum(self.min_shape_np - shape, 0)
        if not deficits.any():
            return img

        pad = []
        for deficit in deficits[::-1]:
            left = int(deficit // 2)
            right = int(deficit - left)
            pad.extend([left, right])
        return F.pad(img, pad, value=0)

    def apply_image(self, img, box_start, box_end, spacing):
        box_start, box_end = self.add_margin_to_bbox(
            box_start,
            box_end,
            self.spatial_shape(img),
            spacing,
        )
        box_start, box_end = self.maybe_expand_bbox(
            box_start,
            box_end,
            self.spatial_shape(img),
        )
        img = self.crop(img, box_start, box_end)
        return self.maybe_pad(img)

    def __call__(self, data, box_start, box_end, spacing):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.apply_image(d[key], box_start, box_end, spacing)
        return d


class CropForegroundMinShaped(CropMaybePad):
    """
    Modes
    -----
    a) FG present and bbox >= min_shape:
        crop FG bbox as-is
    b) FG present and bbox < min_shape in any dim:
        expand bbox to >= min_shape, then shift inside image bounds
        so no padding is used
    c) FG absent:
        random crop of size min_shape

    """

    def __init__(
        self,
        keys,
        source_key,
        min_shape,
        margin=0,
        select_fn=lambda x: x > 0,
        channel_indices=None,
        allow_missing_keys=False,
        lazy=False,
        mode="constant",
    ):
        super().__init__(keys=keys, min_shape=min_shape, margin=margin, lazy=lazy)
        self.source_key = source_key
        self.mode = mode

        self.cropper = CropForeground(
            select_fn=select_fn,
            channel_indices=channel_indices,
            margin=self.margin,
            allow_smaller=True,
            k_divisible=1,
            lazy=lazy,
        )

        self.rand_crop = RandSpatialCropd(
            keys=keys,
            roi_size=self.min_shape,
            random_center=True,
            random_size=False,
            lazy=lazy,
        )

    @staticmethod
    def _has_foreground(box_start, box_end):
        return any(int(e) > int(st) for st, e in zip(box_start, box_end))

    def add_margin_to_bbox(self, box_start, box_end, image_shape, spacing):
        return tuple(int(v) for v in box_start), tuple(int(v) for v in box_end)

    def __call__(self, data, lazy=None):
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy

        src = d[self.source_key]
        box_start, box_end = self.cropper.compute_bounding_box(img=src)

        if not self._has_foreground(box_start, box_end):
            return self.rand_crop(d, lazy=lazy_)

        return super().__call__(d, box_start, box_end, spacing=(1.0, 1.0, 1.0))

class CropByYolo(CropMaybePad):
    audit_columns = (
        "case_id",
        "bbox_fn",
        "status",
        "stage",
        "fg_before",
        "fg_after",
        "margin",
        "message",
    )

    def __init__(
        self,
        keys=("image", "lm"),
        lm_key="lm",
        bbox_key="bbox",
        min_shape=(0, 0, 0),
        margin=20,
        sanitize=True,
        report_path=None,
        allow_missing_keys=False,
    ):
        super().__init__(keys=keys, min_shape=min_shape, margin=margin)
        self.lm_key = lm_key
        self.bbox_key = bbox_key
        self.sanitize = bool(sanitize)
        self.report_path = report_path
        self.logger = logging.getLogger(__name__)

    def __call__(self, data):
        dici = dict(data)
        key = self.lm_key if self.lm_key is not None else self.keys[0]
        spacing = spacing_from_affine(dici[key].meta["affine"])
        box_start, box_end = self._yolo_bbox_to_bounds(
            tuple(int(v) for v in dici["image"].shape[1:]),
            dici[self.bbox_key],
        )
        out = super().__call__(dici, box_start, box_end, spacing)
        out[self.bbox_key] = dici[self.bbox_key]
        fg_before = self._fg_count(dici)
        fg_after = self._fg_count(out)
        if not self.sanitize:
            if self.report_path:
                self._append_audit_row(
                    data=dici,
                    status="NORMAL",
                    stage="sanitize_disabled",
                    fg_before=fg_before,
                    fg_after=fg_after,
                    message="NORMAL",
                )
            return out

        if fg_before == fg_after:
            self._append_audit_row(
                data=dici,
                status="NORMAL",
                stage="single",
                fg_before=fg_before,
                fg_after=fg_after,
                message="NORMAL",
            )
            return out

        self._log_sanitize_mismatch(
            data=dici,
            fg_before=fg_before,
            fg_after=fg_after,
        )
        self._append_audit_row(
            data=dici,
            status="WARNING",
            stage="single",
            fg_before=fg_before,
            fg_after=fg_after,
            message="Foreground mismatch after single-pass YOLO crop.",
        )
        return out

    def _fg_count(self, data: dict) -> int:
        if self.lm_key is None:
            return 0
        return int(torch.count_nonzero(data[self.lm_key]).item())

    @staticmethod
    def _yolo_bbox_to_bounds(img_shape, bbox_dici):
        width3d, ap3d, height3d = img_shape
        wd = bbox_dici["width"]
        height = bbox_dici["height"]
        ap = bbox_dici["ap"]
        return (
            math.floor(wd[0] * width3d),
            math.floor(ap[0] * ap3d),
            math.floor((1.0 - height[1]) * height3d),
        ), (
            math.ceil(wd[1] * width3d),
            math.ceil(ap[1] * ap3d),
            math.ceil((1.0 - height[0]) * height3d),
        )

    def _append_audit_row(
        self,
        *,
        data: dict,
        status: str,
        stage: str,
        fg_before,
        fg_after,
        message: str,
    ) -> None:
        if not self.report_path:
            return

        row = {
            "case_id": "" if data.get("case_id") is None else str(data["case_id"]),
            "bbox_fn": "" if data.get("bbox_fn") is None else str(data["bbox_fn"]),
            "status": status,
            "stage": stage,
            "fg_before": "" if fg_before is None else int(fg_before),
            "fg_after": "" if fg_after is None else int(fg_after),
            "margin": float(self.margin),
            "message": message,
        }
        self._append_csv_row(row)

    def _append_csv_row(self, row: dict) -> None:
        report_path = Path(self.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("a+", newline="") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0, os.SEEK_END)
                write_header = handle.tell() == 0
                writer = csv.DictWriter(handle, fieldnames=self.audit_columns)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _log_sanitize_mismatch(
        self,
        *,
        data: dict,
        fg_before: int,
        fg_after: int,
    ) -> None:
        self.logger.warning(
            "CropByYolo fg mismatch: case_id=%s bbox=%s fg_before=%d fg_after=%d margin=%.1fmm",
            data.get("case_id"),
            data.get("bbox_fn"),
            fg_before,
            fg_after,
            self.margin,
        )



class UnsqueezeDimd(MapTransform):
    def __init__(self, keys, dim=0):
        super().__init__(keys)
        self.dim = dim

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = d[k].unsqueeze(self.dim)
        return d
class ExtractContiguousSlicesd(RandomizableTransform, MapTransform):
    """
    Extract 3 contiguous slices (z-1, z, z+1) from image and label volumes.
    Outputs:
    """

    def __init__(
        self,
        keys: KeysCollection = ("image_fns", "lm_fldr", "n_slices"),
        allow_missing_keys=False,
    ):
        RandomizableTransform.__init__(self, 1)  # always randomize
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]

    def randomize(self, n_slices):
        self.z = self.R.randint(1, n_slices - 2)

    def __call__(self, data):
        # Handle both single dictionary and list of dictionaries
        n_slices = data["n_slices"]
        self.randomize(n_slices)
        # image_prev = img_fns[self.z]
        # image_curr = img_fns[self.z+1]
        # image_next = img_fns[self.z+2]
        # lm_prev = lm/image_prev.name
        # lm_curr = lm/image_curr.name
        # lm_next = lm/image_next.name
        # dici = {"image_prev":image_prev, "image_curr":image_curr, "image_next":image_next, "lm_prev":lm_prev, "lm_curr":lm_curr, "lm_next":lm_next}

        lm_fldr = data["lm_fldr"]
        image_fns = data["image_fns"]
        outs = []
        for i in range(3):
            substring = "slice" + int_to_str(self.z + i, 3)
            img_fn = [fn for fn in image_fns if substring in fn.name][0]
            lm_fn = lm_fldr / (img_fn.name)
            listi = [img_fn, lm_fn]
            outs.append(listi)
        images, lms = [], []
        for sublist in outs:
            img_fn = sublist[0]
            lm_fn = sublist[1]
            img = torch.load(img_fn, weights_only=False)
            lm = torch.load(lm_fn, weights_only=False)
            img = torch.Tensor(img)
            lm = torch.Tensor(lm)
            images.append(img)
            lms.append(lm)
        images = torch.stack(images)
        lms = torch.stack(lms)
        dici = {"image": images, "lm": lms}
        return dici


class Project2D(MonaiDictTransform, Randomizable):
    def __init__(
        self, keys: KeysCollection, dim, operations , suffix, output_keys=None
    ):
        super().__init__(keys)
        assert len(keys) == len(operations), (
            "Same number of operations as keys must be given"
        )
        self.operations = [getattr(torch, operation) for operation in operations]
        self.output_keys = output_keys if output_keys else keys
        self.dim = dim
        self.suffix = suffix
        self.do_randomize = False if dim else True

    def __call__(self, data: dict):
        self.randomize()
        for key, output_key, operation in zip(
            self.key_iterator(data), self.output_keys, self.operations
        ):
            data[output_key] = self.func(data[key], operation)
        return data

    def randomize(self):
        if self.do_randomize:
            self.dim = self.R.randint(1, 4)

    def func(self, data, operation):
        data = data.clone()
        data = operation(data, dim=self.dim)
        data.meta["project2d"]= {"dim": self.dim, "operation": operation.__name__, "suffix": self.suffix}
        return data




class ResizeToTensord(MonaiDictTransform):
    def __init__(self, keys: KeysCollection, mode, key_template_tensor):
        super().__init__(keys)
        self.key_template_tensor = key_template_tensor  # spatial size is extracted from this key, typically an image
        self.mode = mode

    def __call__(self, data):
        tnsr = data[self.key_template_tensor]

        if (l := len(tnsr.shape)) == 4:
            spatial_shape = tnsr[0].shape
        elif l == 3:
            spatial_shape = tnsr.shape
        else:
            raise ValueError("spatial size is not 3 or 4, but is {}".format(l))
        for key in self.key_iterator(data):
            data[key] = _resize3d(data[key], spatial_shape, self.mode)
        return data


class ResizeToMetaSpatialShaped(MonaiDictTransform):
    def __init__(self, keys: KeysCollection, mode, meta_key="spatial_shape"):
        super().__init__(keys)
        self.meta_key = meta_key
        self.mode = mode

    def __call__(self, data):
        for key in self.key_iterator(data):
            tnsr = data[key]
            spatial_shape = tnsr.meta[self.meta_key]
            spatial_shape = spatial_shape.tolist()
            data[key] = _resize3d(tnsr, spatial_shape, self.mode)
        return data


def _ornt_current_to_original(meta):
    affine = torch.as_tensor(meta["affine"]).cpu().numpy()
    original_affine = torch.as_tensor(meta["original_affine"]).cpu().numpy()
    current = nib.orientations.io_orientation(affine)
    original = nib.orientations.io_orientation(original_affine)
    return nib.orientations.ornt_transform(current, original)


def _apply_ornt_channel_first(x, ornt):
    perm = [0] + [int(ax) + 1 for ax in ornt[:, 0]]
    x = torch.as_tensor(x).permute(*perm).contiguous()
    flip_dims = tuple(i + 1 for i, flip in enumerate(ornt[:, 1]) if int(flip) == -1)
    if len(flip_dims):
        x = torch.flip(x, dims=flip_dims).contiguous()
    return x


class RestoreOriginalOrientationd(MapTransform):
    def __init__(self, keys=["pred"], allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, d):
        for key in self.key_iterator(d):
            pred = d[key]
            meta = deepcopy(pred.meta)
            ornt = _ornt_current_to_original(meta)
            pred = _apply_ornt_channel_first(pred, ornt)
            meta["affine"] = torch.as_tensor(meta["original_affine"]).clone()
            meta["spatial_shape"] = tuple(int(v) for v in pred.shape[1:])
            d[key] = MetaTensor(pred, meta=meta)
        return d


class PermuteImageMask(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys,
        prob: float = 1,
        do_transform: bool = True,
    ):
        MapTransform.__init__(self, keys, False)
        RandomizableTransform.__init__(self, prob)
        self.keys = keys
        self.prob = prob
        self.do_transform = do_transform

    def func(self, x):
        if np.random.rand() < self.p:
            img, mask = x
            sequence = (0,) + tuple(
                np.random.choice([1, 2], size=2, replace=False)
            )  # if dim0 is different, this will make pblms
            img_permuted, mask_permuted = (
                torch.permute(img, dims=sequence),
                torch.permute(mask, dims=sequence),
            )
            return img_permuted, mask_permuted
        else:
            return x


def one_hot(tnsr, classes, axis, fnc=torch.stack):
    """
    takes tnsr, splits it into n_labels, then either stacks the results list on axis (creating a new axis), or concatenates it
    """

    tnsr = [torch.where(tnsr == c, 1, 0) for c in range(classes)]
    tnsr = fnc(tnsr, axis)
    return tnsr


def flip_random(x):
    """
    3d array, can be horizontal flip, vertical flip or craniocaudal
    """

    dims = np.random.choice(size=2, a=[0, 1, 2], replace=False)
    x = [torch.flip(arr, dims=[dims[0], dims[1]]) for arr in x]
    return x


class AffineTrainingTransform3D(KeepBBoxTransform):
    """
    to-do: verify if nearestneighbour method preserves multiple mask labels
    """

    def __init__(
        self,
        p=0.5,
        rotate_max=pi / 6,
        translate_factor=0.0,
        scale_ranges=[0.75, 1.25],
        shear: bool = True,
    ):
        """
        params:
        scale_ranges: [min,max]
        """
        self.p = p
        self.rotate_max = rotate_max
        self.translate_factor = translate_factor
        self.scale_ranges = scale_ranges
        self.shear = shear

    def func(self, x):
        img, mask = x
        if np.random.rand() < self.p:
            grid = get_affine_grid(
                img.shape,
                shear=self.shear,
                scale_ranges=self.scale_ranges,
                rotate_max=self.rotate_max,
                translate_factor=self.translate_factor,
                device=img.device,
            ).type(img.dtype)
            img = F.grid_sample(img, grid)
            mask = F.grid_sample(mask.type(img.dtype), grid, mode="nearest")
            return img, mask.to(torch.uint8)
        return img, mask


#
# def expand_lesion(lesion_start_ind,lesion_end_ind,all_locs,expand_factor=0.2):
#         lesion_start_loc = all_locs[lesion_start_ind]
#         lesion_end_loc = all_locs[lesion_end_ind-1]
#         lesion_span = lesion_end_ind-lesion_start_ind
#         lesion_to_total_frac = lesion_span/all_locs.numel()
#         if lesion_to_total_frac>0.5:
#             return all_locs
#         if lesion_to_total_frac> 0.4:
#             expand_factor = np.minimum(0.2,expand_factor)
#         if lesion_to_total_frac> 0.3:
#             expand_factor = np.minimum(0.3,expand_factor)
#         if lesion_to_total_frac> 0.2:
#             expand_factor = np.minimum(0.4,expand_factor)
#         expand_by = int(expand_factor*lesion_span)
#         lesion_span_new= int(lesion_span+ np.minimum(expand_by,lesion_start_ind)+np.minimum(all_locs.numel()-lesion_end_loc,expand_by))
#         smooth_out_zone=expand_by
#         smooth_out_before_ind= int(lesion_start_ind-expand_by-smooth_out_zone)
#         truncate_before = int(np.minimum(lesion_start_ind-expand_by-smooth_out_zone,0))
#         truncate_after = np.minimum(0,all_locs.numel()-5-(lesion_end_ind + expand_by+smooth_out_zone))# shorten the after_zone if it exceeds total image dim
#         smooth_out_after_ind = (int(lesion_end_ind + expand_by+smooth_out_zone+truncate_after))
#         smooth_out_before_loc = all_locs[smooth_out_before_ind]
#         smooth_out_after_loc = all_locs[smooth_out_after_ind-1]
#
#         smooth_out_before_zone = torch.linspace(smooth_out_before_loc,lesion_start_loc,np.maximum(0,smooth_out_zone+1+truncate_before))
#         smooth_out_after_zone = torch.linspace(lesion_end_loc,smooth_out_after_loc,smooth_out_zone+1+truncate_after)
#         lesion_new_locs = torch.linspace(lesion_start_loc,lesion_end_loc,lesion_span_new)
#         lesion_segment_new = torch.cat([smooth_out_before_zone[:-1],lesion_new_locs,smooth_out_after_zone[1:]])
#         all_locs_modified = all_locs.clone()
#         all_locs_modified[smooth_out_before_ind:smooth_out_after_ind]= lesion_segment_new
#         return all_locs_modified
#
#


class CropImgMask(KeepBBoxTransform):
    def __init__(self, patch_size, input_dims):
        self.dim = len(patch_size)
        self.patch_halved = [int(x / 2) for x in patch_size]
        self.input_dims = input_dims

    def func(self, x):
        img, mask = x
        center = [x / 2 for x in img.shape[-self.dim :]]
        slices = [
            slice(None),
        ] * (
            self.input_dims - 3
        )  # batch and channel dims if its a batch otherwise empty
        for ind in range(self.dim):
            source_sz = center[ind]
            target_sz = self.patch_halved[ind]
            if source_sz > target_sz:
                slc = slice(int(source_sz - target_sz), int(source_sz + target_sz))
            else:
                slc = slice(None)
            slices.append(slc)
        img, mask = img[slices], mask[slices]
        return img, mask


class ResizeBatch(ItemTransform):
    def __init__(self, target_size):
        self.target_size = target_size

    def encodes(self, x):
        img, mask = x
        if list(img.shape[2:]) != self.target_size:
            img = F.interpolate(img, size=self.target_size, mode="trilinear")
            mask = F.interpolate(mask, size=self.target_size, mode="nearest")
        return img, mask


class PadDeficitImgMask(KeepBBoxTransform):
    order = 0

    def __init__(
        self,
        patch_size,
        input_dims,
        pad_values: Union[list, tuple, int] = [0, 0],
        mode="constant",
        return_padding_array=False,
    ):
        """
        pad_values is a tuple/list 0 entry for img, 1 entry for mas
        """

        if isinstance(pad_values, int):
            pad_values = [pad_values, int(pad_values)]
        assert isinstance(pad_values[1], int), "Provide integer pad value for the mask"
        self.first_dims = input_dims - len(patch_size)
        self.last_dims = input_dims - self.first_dims
        self.patch_size = np.array(patch_size)
        self.pad_values = pad_values
        self.return_padding_array = return_padding_array

    def func(self, x):
        pad_deficits = []
        if any(self.patch_size - x[0].shape[-self.last_dims :]):
            difference = np.maximum(0, self.patch_size - x[0].shape[-self.last_dims :])
            for diff in difference:
                pad_deficits += [math.ceil(diff / 2), math.floor(diff / 2)]
            pad_deficits = pad_deficits[-1::-1]  # torch expects flipped backward
            pad_deficits += [
                0,
            ] * self.first_dims
            padded_arrays = []
            for arr, pv in zip(x, self.pad_values):
                arr = F.pad(arr, pad_deficits, value=pv)
                padded_arrays.append(arr)
            x = padded_arrays
        if self.return_padding_array == True:
            x.append(pad_deficits)
        return x


def crop_to_bbox(arr, bbox, crop_axes, crop_padding=0.0, stride=[1, 1, 1]):
    """
    param arr: torch tensor or np array to be cropped
    param bbox: Bounding box (3D only supported)
    param crop_axes:  any combination of 'xyz' may be used (e.g., 'xz' will crop in x and z axes)
    param crop_padding: add crop_padding [0,1] fraction to all the planes of cropping.
    param stride: stride in each plane
    """
    assert len(arr.shape) == 3, "only supports 3d images"
    bbox_extra_pct = [
        int((bbox[i][1] - bbox[i][0]) * crop_padding / 2) for i in range(len(bbox))
    ]
    bbox_mod = [
        [
            np.maximum(0, bbox[j][0] - bbox_extra_pct[j]),
            np.minimum(bbox[j][1] + bbox_extra_pct[j], arr.shape[j]),
        ]
        for j in range(arr.ndim)
    ]
    slices = []
    for dim, axis in zip(
        [0, 1, 2], ["z", "y", "x"]
    ):  # tensors are opposite arrranged to numpy
        if axis in crop_axes:
            slices.append(slice(bbox_mod[dim][0], bbox_mod[dim][1], stride[dim]))
        else:
            slices.append(slice(0, arr.shape[dim], stride[dim]))
    return arr[tuple(slices)]


def get_random_shear_matrix(scale=torch.rand(1) * 0.5 + 0.5, device="cpu"):
    shear_matrix = torch.eye(3, 3, device=device)
    shear_index = torch.randperm(3)[:2]
    shear_matrix[shear_index[0], shear_index[1]] = scale
    return shear_matrix


def get_random_rotation3d(angle_max=pi / 6, device="cpu"):
    angles = torch.rand(3) * angle_max
    alpha, beta, gamma = angles[0], angles[1], angles[2]
    rot_3d = torch.tensor(
        [
            [
                cos(beta) * cos(gamma),
                sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma),
                cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma),
            ],
            [
                cos(beta) * sin(gamma),
                sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
                cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma),
            ],
            [-sin(beta), sin(alpha) * cos(beta), cos(alpha) * cos(beta)],
        ],
        device=device,
    )

    return rot_3d


def get_affine_grid(
    input_shape,
    shear=True,
    scale_ranges=[0.75, 1.25],
    rotate_max=pi / 6,
    translate_factor=0.0,
    device="cpu",
):
    output_shape = torch.tensor(input_shape)
    output_shape_last3d = torch.tensor(input_shape[-3:])

    bs = int(output_shape[0])
    scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
    output_shape[-3:] = (output_shape_last3d * scale).int()
    theta = torch.zeros(bs, 3, 4, device=device)
    translate = (torch.rand(3, device=device) - 0.5) * translate_factor
    theta[:, :, 3] = translate
    if shear:
        shear_matrices = torch.stack(
            [get_random_shear_matrix(device=device) for b in range(bs)]
        )
    else:
        shear_matrices = torch.stack(
            [torch.eye(3, 3, device=device) for x in range(bs)]
        )
    if rotate_max > 0:
        rotation_matrices = torch.stack(
            [
                get_random_rotation3d(angle_max=rotate_max, device=device)
                for b in range(bs)
            ],
            dim=0,
        )
    else:
        rotation_matrices = torch.stack(
            [torch.eye(3, 3, device=device) for x in range(bs)]
        )
    final_transform = shear_matrices @ rotation_matrices
    theta[:, :, :3] = final_transform
    grid = F.affine_grid(theta, tuple(output_shape))
    return grid


class PermuteImageMask(KeepBBoxTransform):
    def __init__(self, p=0.3):
        self.p = p
        super().__init__()

    def func(self, x):
        if np.random.rand() < self.p:
            img, mask = x
            sequence = (0,) + tuple(
                np.random.choice([1, 2], size=2, replace=False)
            )  # if dim0 is different, this will make pblms
            img_permuted, mask_permuted = (
                torch.permute(img, dims=sequence),
                torch.permute(mask, dims=sequence),
            )
            return img_permuted, mask_permuted
        else:
            return x


def slices_from_lists(slc_start, slc_stop, stride=None):
    slices = []
    for start, stop, stride_ in zip(slc_start, slc_stop, stride):
        slices.append(slice(int(start), int(stop), stride_))
    return tuple(slices)


# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
if __name__ == "__main__":
    from fran.data.dataset import ImageMaskBBoxDataset
    from fran.transforms.misc_transforms import create_augmentations
    from fran.utils.common import *
    from fran.data.dataset import ImageMaskBBoxDataset
    from fran.transforms.misc_transforms import create_augmentations
    from fran.utils.common import *

# %%

    print("CASE 1: single-channel 4D, z smaller than min_shape")
    img = torch.arange(625 * 625 * 103, dtype=torch.float32).reshape(1, 625, 625, 103)
    spacing = (0.8, 0.8, 1.5)
    C = CropMaybePad(keys=["image"], min_shape=(176, 176, 106), margin=5.0)
# %%

    box_start = (180, 210, 8)
    box_end = (420, 390, 95)

    print("input shape:", tuple(img.shape))
    print("spacing mm:", spacing)
    print("bbox in:", box_start, box_end)

    box2 = C.add_margin_to_bbox(box_start, box_end, C.spatial_shape(img), spacing)
    box2 = C.maybe_expand_bbox(*box2, C.spatial_shape(img))
    print("bbox expanded:", box2)

# %%
    cropped = C.crop(img, *box2)
    print("cropped shape:", tuple(cropped.shape))

    padded = C.maybe_pad(cropped)
    print("padded shape:", tuple(padded.shape))

    out = C.apply_image(img, box_start, box_end, spacing=spacing)
    print("final shape:", tuple(out.shape))
    print("expected:", (1, 240, 180, 106))
    print("sum:", float(out.sum()))


    print("\nCASE 2: channel-first fallback style")
    img_cf = torch.ones((1, 625, 625, 103), dtype=torch.float32)
    spacing_cf = (1.0, 1.0, 2.5)
    C2 = CropMaybePad(keys=["image"], min_shape=(176, 176, 106), margin=5.0)

    box_start = (220, 240, 10)
    box_end = (360, 360, 92)

    print("input shape:", tuple(img_cf.shape))
    print("spacing mm:", spacing_cf)
    print("bbox in:", box_start, box_end)

    box2 = C2.add_margin_to_bbox(box_start, box_end, C2.spatial_shape(img_cf), spacing_cf)
    box2 = C2.maybe_expand_bbox(*box2, C2.spatial_shape(img_cf))
    print("bbox expanded:", box2)

    cropped = C2.crop(img_cf, *box2)
    print("cropped shape:", tuple(cropped.shape))

    padded = C2.maybe_pad(cropped)
    print("padded shape:", tuple(padded.shape))

    out = C2.apply_image(img_cf, box_start, box_end, spacing=spacing_cf)
    print("final shape:", tuple(out.shape))
    print("expected:", (1, 176, 176, 106))

    z_nonzero = torch.where(out[0].sum(dim=(0, 1)) > 0)[0]
    print("nonzero z idx:", z_nonzero.tolist()[:5], "...", z_nonzero.tolist()[-5:])
    print("nonzero z count:", len(z_nonzero))# %%
