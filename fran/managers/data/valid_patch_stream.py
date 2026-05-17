from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import PatchIterd
from monai.data.meta_tensor import MetaTensor
from torch.utils.data import IterableDataset

from fran.utils.common import PAD_VALUE


def _is_patch_padded(coords, original_spatial_shape) -> bool:
    coords_arr = np.asarray(coords)
    shape_arr = np.asarray(original_spatial_shape)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        return False
    if coords_arr.shape[0] == shape_arr.shape[0] + 1:
        coords_arr = coords_arr[1:]
    starts = coords_arr[:, 0]
    stops = coords_arr[:, 1]
    return bool(np.any(starts < 0) or np.any(stops > shape_arr))


def _pad_mask_from_coords(coords, original_spatial_shape, spatial_shape):
    coords_arr = np.asarray(coords)
    shape_arr = np.asarray(original_spatial_shape)
    if coords_arr.shape[0] == shape_arr.shape[0] + 1:
        coords_arr = coords_arr[1:]

    padded_mask = torch.ones(spatial_shape, dtype=torch.bool)
    valid_slices = []
    for coord_pair, orig_dim, patch_dim in zip(coords_arr, shape_arr, spatial_shape):
        start, stop = (int(v) for v in coord_pair)
        pad_before = max(0, -start)
        pad_after = max(0, stop - int(orig_dim))
        valid_start = min(patch_dim, pad_before)
        valid_stop = max(valid_start, patch_dim - pad_after)
        valid_slices.append(slice(valid_start, valid_stop))
    padded_mask[tuple(valid_slices)] = False
    return padded_mask


def _rewrite_padded_lm(lm, coords, original_spatial_shape):
    spatial_shape = tuple(int(v) for v in lm.shape[1:])
    padded_mask = _pad_mask_from_coords(coords, original_spatial_shape, spatial_shape)
    if not padded_mask.any():
        return lm, False
    lm_out = lm.clone()
    lm_out[(slice(None), padded_mask)] = PAD_VALUE
    return lm_out, True


def _pad_tensor_to_patch_size(tensor, patch_size, pad_value):
    spatial_shape = tuple(int(v) for v in tensor.shape[1:])
    target_shape = tuple(int(v) for v in patch_size)
    if spatial_shape == target_shape:
        return tensor, False

    pad_widths = []
    for current_dim, target_dim in zip(reversed(spatial_shape), reversed(target_shape)):
        pad_after = max(0, int(target_dim) - int(current_dim))
        pad_widths.extend([0, pad_after])

    padded = F.pad(tensor.as_tensor() if isinstance(tensor, MetaTensor) else tensor, pad=pad_widths, mode="constant", value=pad_value)
    padded = padded.contiguous()
    if isinstance(tensor, MetaTensor):
        return MetaTensor(padded, meta=deepcopy(tensor.meta)), True
    return padded, True


def _rewrite_manual_tail_padding_lm(lm, original_patch_spatial_shape):
    spatial_shape = tuple(int(v) for v in lm.shape[1:])
    if tuple(int(v) for v in original_patch_spatial_shape) == spatial_shape:
        return lm, False
    padded_mask = torch.ones(spatial_shape, dtype=torch.bool)
    valid_slices = tuple(slice(0, int(v)) for v in original_patch_spatial_shape)
    padded_mask[valid_slices] = False
    if not padded_mask.any():
        return lm, False
    lm_out = lm.clone()
    lm_out[(slice(None), padded_mask)] = PAD_VALUE
    return lm_out, True


class ValidPatchStreamDataset(IterableDataset):
    """
    Validation patch stream.
    One yielded patch never mixes case contents.
    Batching may mix case_ids across patch slots downstream.
    """

    def __init__(self, case_dataset, patch_size):
        self.case_dataset = case_dataset
        self.patch_size = tuple(int(v) for v in patch_size)
        self.patch_iter = PatchIterd(
            keys=["image", "lm"],
            patch_size=self.patch_size,
            mode="constant",
            constant_values=0,
        )

    def __iter__(self):
        for case_ds_idx in range(len(self.case_dataset)):
            case_data = self.case_dataset[case_ds_idx]
            case_id = str(case_data["case_id"])
            case_patches = list(self.patch_iter(case_data))
            patches_in_case = len(case_patches)
            for patch_index, (patch_dict, coords) in enumerate(case_patches):
                patch_out = deepcopy(patch_dict)
                patch_out["case_id"] = case_id
                patch_out["patch_index"] = patch_index
                patch_out["patches_in_case"] = patches_in_case
                patch_out["validation_impl"] = "patch_stream"
                patch_out["is_padded"] = _is_patch_padded(
                    coords=patch_out["patch_coords"],
                    original_spatial_shape=patch_out["original_spatial_shape"],
                )

                original_patch_spatial_shape = tuple(int(v) for v in patch_out["lm"].shape[1:])

                image, image_was_padded = _pad_tensor_to_patch_size(
                    tensor=patch_out["image"],
                    patch_size=self.patch_size,
                    pad_value=0,
                )
                lm, lm_was_padded = _pad_tensor_to_patch_size(
                    tensor=patch_out["lm"],
                    patch_size=self.patch_size,
                    pad_value=0,
                )
                patch_out["image"] = image
                lm_rewritten, was_padded = _rewrite_padded_lm(
                    lm=lm,
                    coords=patch_out["patch_coords"],
                    original_spatial_shape=patch_out["original_spatial_shape"],
                )
                lm_rewritten, was_padded_manual = _rewrite_manual_tail_padding_lm(
                    lm=lm_rewritten,
                    original_patch_spatial_shape=original_patch_spatial_shape,
                )
                patch_out["lm"] = lm_rewritten
                patch_out["is_padded"] = bool(
                    was_padded
                    or was_padded_manual
                    or image_was_padded
                    or lm_was_padded
                    or patch_out["is_padded"]
                )

                image = patch_out["image"]
                if isinstance(image, MetaTensor):
                    image.meta["case_id"] = case_id
                    image.meta["patch_index"] = patch_index
                    image.meta["patches_in_case"] = patches_in_case
                if isinstance(lm_rewritten, MetaTensor):
                    lm_rewritten.meta["case_id"] = case_id
                    lm_rewritten.meta["patch_index"] = patch_index
                    lm_rewritten.meta["patches_in_case"] = patches_in_case
                yield patch_out


def valid_patch_stream_collated(batch):
    images = []
    labels = []
    image_fns = []
    lm_fns = []
    case_ids = []
    patch_coords = []
    start_pos = []
    is_padded = []
    patch_index = []
    patches_in_case = []
    original_spatial_shape = []

    for item in batch:
        image = item["image"]
        lm = item["lm"]
        images.append(image)
        labels.append(lm)
        image_fns.append(image.meta["filename_or_obj"])
        lm_fns.append(lm.meta["filename_or_obj"])
        case_ids.append(item["case_id"])
        patch_coords.append(item["patch_coords"])
        start_pos.append(item["start_pos"])
        is_padded.append(bool(item["is_padded"]))
        patch_index.append(int(item["patch_index"]))
        patches_in_case.append(int(item["patches_in_case"]))
        original_spatial_shape.append(tuple(int(v) for v in item["original_spatial_shape"]))

    if len(batch) == 1:
        image_fns = image_fns[0]
        lm_fns = lm_fns[0]

    images_out = torch.stack(images, 0)
    labels_out = torch.stack(labels, 0)
    images_out.meta["filename_or_obj"] = image_fns
    labels_out.meta["filename_or_obj"] = lm_fns
    images_out.meta["case_id"] = case_ids
    labels_out.meta["case_id"] = case_ids
    images_out.meta["patch_index"] = patch_index
    labels_out.meta["patch_index"] = patch_index

    return {
        "image": images_out,
        "lm": labels_out,
        "case_id": case_ids,
        "patch_coords": patch_coords,
        "start_pos": start_pos,
        "is_padded": is_padded,
        "patch_index": patch_index,
        "patches_in_case": patches_in_case,
        "original_spatial_shape": original_spatial_shape,
        "validation_impl": "patch_stream",
    }
