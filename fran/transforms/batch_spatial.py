from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.transforms.transform import MapTransform, RandomizableTransform


def _as_spatial_size(spatial_size) -> tuple[int, int, int]:
    return tuple(int(v) for v in spatial_size)

def delete_unwanted_files_folders(
        parent, delete_these=["SECTRA",  "README", "ComponentUpdate", "Viewer","DICOMDIR"]
    ):
        dd = list(parent.rglob("*"))
        for dirr in dd:
            if dirr.exists():
                if any((match := substring) in str(dirr) for substring in delete_these):
                    print("Deleting {}".format(dirr))
                    if dirr.is_file() == True:
                        dirr.unlink()
                    else:
                        shutil.rmtree(dirr)

def _spatial_pad_values(current_shape, target_shape) -> tuple[int, int, int, int, int, int]:
    pad_pairs = []
    for current, target in zip(current_shape, target_shape):
        deficit = max(target - current, 0)
        pad_left = deficit // 2
        pad_right = deficit - pad_left
        pad_pairs.append((pad_left, pad_right))
    return (
        pad_pairs[2][0],
        pad_pairs[2][1],
        pad_pairs[1][0],
        pad_pairs[1][1],
        pad_pairs[0][0],
        pad_pairs[0][1],
    )


def _center_crop_slices(current_shape, target_shape) -> tuple[slice, slice, slice]:
    slices = []
    for current, target in zip(current_shape, target_shape):
        if current <= target:
            slices.append(slice(0, current))
            continue
        start = (current - target) // 2
        stop = start + target
        slices.append(slice(start, stop))
    return tuple(slices)


class BatchRandFlipd(MapTransform, RandomizableTransform):
    def __init__(self, keys, prob: float, spatial_axis: int, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=1.0)
        self.item_prob = float(prob)
        self.spatial_axis = int(spatial_axis)

    def randomize(self, data):
        batch_size = data[self.keys[0]].shape[0]
        self._active = self.R.rand(batch_size) < self.item_prob

    def __call__(self, data):
        d = dict(data)
        self.randomize(d)
        key0 = d[self.keys[0]]
        batch_mask = torch.as_tensor(self._active, device=key0.device, dtype=torch.bool)
        if not torch.any(batch_mask):
            return d

        flip_dim = self.spatial_axis + 2
        mask = batch_mask.view(-1, 1, 1, 1, 1)
        for key in self.key_iterator(d):
            src = d[key]
            flipped = torch.flip(src, dims=(flip_dim,))
            dst = torch.where(mask, flipped, src)
            if hasattr(src, "meta"):
                dst.meta = src.meta
            d[key] = dst
        return d


class BatchCenterCropOrPadd(MapTransform):
    def __init__(self, keys, spatial_size, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.spatial_size = _as_spatial_size(spatial_size)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            src = d[key]
            crop_slices = _center_crop_slices(src.shape[-3:], self.spatial_size)
            dst = src[..., crop_slices[0], crop_slices[1], crop_slices[2]]
            pad = _spatial_pad_values(dst.shape[-3:], self.spatial_size)
            if any(pad):
                dst = F.pad(dst, pad)
            if hasattr(src, "meta"):
                dst.meta = src.meta
            d[key] = dst
        return d


class BatchSpatialPadd(MapTransform):
    def __init__(self, keys, spatial_size, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.spatial_size = _as_spatial_size(spatial_size)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            src = d[key]
            pad = _spatial_pad_values(src.shape[-3:], self.spatial_size)
            if any(pad):
                dst = F.pad(src, pad)
                if hasattr(src, "meta"):
                    dst.meta = src.meta
                d[key] = dst
        return d


class BatchResized(MapTransform):
    def __init__(self, keys, spatial_size, mode, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.spatial_size = _as_spatial_size(spatial_size)
        self.mode = tuple(mode)

    def __call__(self, data):
        d = dict(data)
        for key, mode in zip(self.keys, self.mode):
            src = d[key]
            interp_mode = "trilinear" if mode == "linear" else mode
            src_cast = src.float() if mode == "nearest" else src
            kwargs = {"size": self.spatial_size, "mode": interp_mode}
            if interp_mode != "nearest":
                kwargs["align_corners"] = False
            dst = F.interpolate(src_cast, **kwargs).to(dtype=src.dtype)
            if hasattr(src, "meta"):
                dst.meta = src.meta
            d[key] = dst
        return d
