# %%
import math

import ipdb
import monai.transforms.spatial.functional as fm
import skimage.transform as tf
import torch.nn.functional as F
from fran.transforms.base import *
from monai.config.type_definitions import KeysCollection, SequenceStr
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.croppad.array import CropForeground, SpatialPad
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import (
    LazyTransform,
    MapTransform,
    Randomizable,
    RandomizableTransform,
)
from monai.utils.enums import LazyAttr, Method, PytorchPadMode, TraceKeys
from torch import cos, pi, sin

# from utilz.fileio import *
from utilz.helpers import *
from utilz.stringz import int_to_str

tr = ipdb.set_trace
from monai.transforms.croppad.dictionary import Padd, RandSpatialCropd


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


class CropForegroundMinShaped(MapTransform):
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

    Assumption
    ----------
    Image spatial size is >= min_shape in every dim.
    Otherwise, with allow_smaller=True and no padding, exact min_shape
    cannot be guaranteed.
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
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.min_shape = tuple(min_shape)
        self.margin = margin
        self.lazy = lazy
        self.mode = mode

        # MONAI-style margin supports int or per-dim sequence.
        self.cropper = CropForeground(
            select_fn=select_fn,
            channel_indices=channel_indices,
            margin=self.margin,
            allow_smaller=True,  # required: never pad
            k_divisible=1,
            lazy=lazy,
        )

        # used only when no foreground exists
        self.rand_crop = RandSpatialCropd(
            keys=keys,
            roi_size=self.min_shape,
            random_center=True,
            random_size=False,
            lazy=lazy,
        )

    @staticmethod
    def _spatial_shape(x):
        # channel-first: (C, H, W, D) -> (H, W, D)
        return tuple(int(v) for v in x.shape[1:])

    @staticmethod
    def _has_foreground(box_start, box_end):
        return any(int(e) > int(st) for st, e in zip(box_start, box_end))

    @staticmethod
    def _fit_box_to_image(box_start, box_end, img_shape, min_shape):
        """
        Expand bbox to >= min_shape and shift inside image bounds.
        No padding.
        """
        box_start = np.asarray(box_start, dtype=int)
        box_end = np.asarray(box_end, dtype=int)
        img_shape = np.asarray(img_shape, dtype=int)
        min_shape = np.asarray(min_shape, dtype=int)

        fg_shape = box_end - box_start
        out_shape = np.maximum(fg_shape, min_shape)

        center = (box_start + box_end) / 2.0
        new_start = np.floor(center - out_shape / 2.0).astype(int)
        new_end = new_start + out_shape

        # shift right if start < 0
        neg = np.minimum(new_start, 0)
        new_start = new_start - neg
        new_end = new_end - neg

        # shift left if end > img_shape
        overflow = np.maximum(new_end - img_shape, 0)
        new_start = new_start - overflow
        new_end = new_end - overflow

        # final clamp
        new_start = np.maximum(new_start, 0)
        new_end = np.minimum(new_end, img_shape)

        return tuple(int(v) for v in new_start), tuple(int(v) for v in new_end)

    def __call__(self, data, lazy=None):
        d = dict(data)
        lazy_ = self.lazy if lazy is None else lazy

        src = d[self.source_key]
        img_shape = self._spatial_shape(src)

        # prereq check: cannot guarantee >= min_shape without padding
        if any(i < m for i, m in zip(img_shape, self.min_shape)):
            raise ValueError(
                f"Image spatial shape {img_shape} is smaller than min_shape "
                f"{self.min_shape}. With allow_smaller=True and no padding, "
                f">= min_shape cannot be guaranteed."
            )

        box_start, box_end = self.cropper.compute_bounding_box(img=src)

        # mode c: no FG
        if not self._has_foreground(box_start, box_end):
            return self.rand_crop(d, lazy=lazy_)

        fg_shape = tuple(int(e - st) for st, e in zip(box_start, box_end))

        # mode a: fg already large enough
        if all(f >= m for f, m in zip(fg_shape, self.min_shape)):
            final_start, final_end = tuple(box_start), tuple(box_end)

        # mode b: fg too small -> expand to min_shape, no padding
        else:
            final_start, final_end = self._fit_box_to_image(
                box_start=box_start,
                box_end=box_end,
                img_shape=img_shape,
                min_shape=self.min_shape,
            )

        for key in self.key_iterator(d):
            d[key] = self.cropper.crop_pad(
                img=d[key],
                box_start=np.asarray(final_start, dtype=int),
                box_end=np.asarray(final_end, dtype=int),
                mode=self.mode,
                lazy=lazy_,
            )

        return d


class UnsqueezeDimd(MapTransform):
    def __init__(self, keys, dim=0):
        super().__init__(keys)
        self.dim = dim

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = d[k].unsqueeze(self.dim)
        return d
# %%
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
        self, keys: KeysCollection, operations=["sum"], dim=None, output_keys=None
    ):
        super().__init__(keys)
        assert len(keys) == len(operations), (
            "Same number of operations as keys must be given"
        )
        self.operations = [getattr(torch, operation) for operation in operations]
        self.output_keys = output_keys if output_keys else keys
        self.dim = dim
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
        data = operation(data, dim=self.dim)
        return data


# %%


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


class CropExtra(ItemTransform):
    def __init__(self, patch_size):

        self.patch_size = patch_size

    def encodes(self, x):
        img, mask = x
        if list(img.shape) > self.patch_size:
            img = img[: self.patch_size[0], : self.patch_size[1], : self.patch_size[2]]
            mask = mask[
                : self.patch_size[0], : self.patch_size[1], : self.patch_size[2]
            ]
        return img, mask


class CropBatch(ItemTransform):
    def __init__(self, patch_size):
        self.dim = len(patch_size)
        self.patch_halved = [int(x / 2) for x in patch_size]

    def encodes(self, x):
        img, mask = x
        center = [int(x / 2) for x in img.shape[-self.dim :]]
        slices = [slice(None)] * 2
        for ind in range(self.dim):
            slc = slice(
                center[ind] - self.patch_halved[ind],
                center[ind] + self.patch_halved[ind],
            )
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


class StrideRandom(ItemTransform):
    def __init__(
        self, patch_size, input_dims, stride_max=[2, 2, 2], pad_value=-3.0, p=0.3
    ):
        self.patch_size = patch_size
        self.stride_max = stride_max
        self.p = p

        self.Padder = PadDeficitImgMask(
            patch_size=self.patch_size, input_dims=input_dims, pad_value=pad_value
        )

    def encodes(self, x):
        img, mask = x
        if np.random.rand() < self.p:
            stride = [np.random.randint(low=1, high=x + 1) for x in self.stride_max]
            try:
                img, mask = [xx[:: stride[0], :: stride[1], :: stride[2]] for xx in x]
            except:
                print("Stride Random error!")
                print(x[0].shape, x[1].shape, stride)
            img, mask = self.Padder.encodes([img, mask])
        return img, mask


# %%

if __name__ == "__main__":
    from fran.data.dataset import ImageMaskBBoxDataset
    from fran.transforms.misc_transforms import create_augmentations
    from fran.utils.common import *

    # %%
    # %%
    P = Project(project_title="lits")
    proj_defaults = P
    spacings = [1, 1, 1]
    src_patch_size = [220, 220, 110]
    patch_size = [160, 160, 128]
    images_folder = (
        proj_defaults.patches_folder
        / ("spc_100_100_250")
        / ("dim_{0}_{0}_{2}".format(*src_patch_size))
        / ("images")
    )
    bboxes_fn = images_folder.parent / "bboxes_info"
    bboxes = load_dict(bboxes_fn)
    fold = 0

    json_fname = proj_defaults.validation_folds_filename

    imgs = list((proj_defaults.raw_data_folder / ("images")).glob("*"))
    masks = list((proj_defaults.raw_data_folder / ("lms")).glob("*"))
    img_fn = imgs[0]
    mask_fn = masks[0]
    train_ids, val_ids, _ = get_fold_case_ids(
        fold=0, json_fname=proj_defaults.validation_folds_filename
    )
    train_ds = ImageMaskBBoxDataset(proj_defaults, train_ids, bboxes_fn)
    valid_ds = ImageMaskBBoxDataset(proj_defaults, val_ids, bboxes_fn)

    aa = train_ds.median_shape
    # %%
    after_item_intensity = {
        "brightness": [[0.7, 1.3], 0.1],
        "shift": [[-0.2, 0.2], 0.1],
        "noise": [[0, 0.1], 0.1],
        "brightness": [[0.7, 1.5], 0.01],
        "contrast": [[0.7, 1.3], 0.1],
    }
    after_item_spatial = {"flip_random": 0.5}
    intensity_augs, spatial_augs = create_augmentations(
        after_item_intensity, after_item_spatial
    )

    probabilities_intensity, probabilities_spatial = 0.1, 0.5
    after_item_intensity = TrainingAugmentations(
        augs=intensity_augs, p=probabilities_intensity
    )
    after_item_spatial = TrainingAugmentations(
        augs=spatial_augs, p=probabilities_spatial
    )
    # %%
    from fran.data.dataset import ImageMaskBBoxDataset
    from utilz.imageviewers import *

    P = Project(project_title="lits")
    proj_defaults = P
    # %%
    bboxes_pt_fn = proj_defaults.stage1_folder / ("cropped/images_pt/bboxes_info")
    bboxes_nii_fn = proj_defaults.stage1_folder / ("cropped/images_nii/bboxes_info")
    # %%

    n_slices = data["n_slices"]
    z = random.randint(1, n_slices - 2)
    img_fns = data["image_fns"]
    lm = data["lm"]
    image_prev = img_fns[z]
    image_curr = img_fns[z + 1]
    image_next = img_fns[z + 2]
    lm_prev = lm / image_prev.name
    lm_curr = lm / image_curr.name
    lm_next = lm / image_next.name
    dici = {}
    dici = {
        "image_lm_prev": [image_prev, lm_prev],
        "image_lm_curr": [image_curr, lm_curr],
        "image_lm_next": [image_next, lm_next],
    }
    # %%
    # %%

    lm = dici["lm"]
    img = dici["image"]
    # img = img.permute(1,2,0)
    # lm = lm.permute(1,2,0)

    # %%
    # zz = int_to_str(14,3)
    Affine = RandAffined(
        keys=["image", "lm"],
        mode=["bilinear", "nearest"],
        prob=1,
        rotate_range=0.6,
        scale_range=2,
        shear_range=0.5,
    )
    # %%

    dici = Affine(dici)
    # %%

    RP = RandomPatch()

    # %%
    Rva = RandCropByPosNegLabeld(
        keys=["image", "lm"],
        label_key="lm",
        image_key="image",
        image_threshold=-2600,
        spatial_size=tm.plan["patch_size"],
        pos=1,
        neg=1,
        num_samples=1,
        lazy=True,
        allow_smaller=True,
    )
    # %%
    tm.plan["patch_size"]
    dici = Rva(dici)

    # %%
    img, lm = dici["image"], dici["lm"]
    dici = Rva(dici)

# %%
