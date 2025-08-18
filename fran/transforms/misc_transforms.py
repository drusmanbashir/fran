# %%
from collections.abc import Hashable, Mapping
from functools import partial
from pathlib import Path
from typing import Union

import ipdb
import SimpleITK as sitk
from monai.apps.detection.transforms.array import ConvertBoxMode
from monai.data.meta_tensor import MetaTensor
import torch
from label_analysis.helpers import listify, relabel
# from label_analysis.merge import merge_pt
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms.transform import MapTransform
from monai.transforms.utility.dictionary import FgBgToIndicesd

from fran.transforms.base import MonaiDictTransform
from utilz.string import ast_literal_eval

tr = ipdb.set_trace

from fastcore.basics import Dict, store_attr
from fasttransform.transform import ItemTransform, store_attr

import fran.transforms.intensitytransforms as intensity
import fran.transforms.spatialtransforms as spatial


#
# class BBoxFromLabelMap(MonaiDictTransform):
#     def func(self,data):
#


def merge_pt(base_labelmap: torch.Tensor, 
             overlay_labelmaps: Union[torch.Tensor, list[torch.Tensor]],
             label_lists: list[list[int]] = None) -> torch.Tensor:
    """Merge multiple labelmaps by overlaying them onto a base labelmap.
    
    Args:
        base_labelmap: Base labelmap to overlay others onto
        overlay_labelmaps: Single labelmap or list of labelmaps to overlay
        label_lists: Optional list of label lists, one per overlay labelmap.
                    If None, uses all non-zero labels from each overlay.
                    
    Returns:
        Merged labelmap with overlays applied in order (later ones override earlier)
    """
    # Ensure overlay_labelmaps is a list
    if not isinstance(overlay_labelmaps, (tuple, list)):
        overlay_labelmaps = [overlay_labelmaps]
        
    # Get non-zero labels if not provided
    if label_lists is None:
        label_lists = [lm.unique()[1:] for lm in overlay_labelmaps]
        
    # Apply each overlay
    result = base_labelmap.clone()
    for overlay, labels in zip(overlay_labelmaps, label_lists):
        for label in labels:
            result[overlay == label] = label
            
    return result

class MaskLabelRemapd(MapTransform):
    # there should be no channel dim
    # src_dest_labels should include background label, e.g., 0 too. n_classes = length of this list.
    def __init__(self, keys, src_dest_labels: dict, allow_missing_keys=False, use_sitk=True):
        super().__init__(keys, allow_missing_keys)
        if isinstance(src_dest_labels, str):
            src_dest_labels = ast_literal_eval(src_dest_labels)
        if use_sitk==True:
            # self.src_dest_labels = {x: y for x, y in src_dest_labels}
            self.remapper = self.remapper_sitk
        else:
            # self.src_dest_labels = src_dest_labels
            self.remapper = self.remapper_pt
        self.src_dest_labels = src_dest_labels


    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            lm = d[key]
            if not lm.dim()==3:
                raise ValueError("Only 3D tensors supported")
            lm = self.remapper(lm)
            d[key] = lm
        return d

    def remapper_pt(self, mask):
        n_classes = len(self.src_dest_labels)
        mask_out = torch.zeros(mask.shape, dtype=mask.dtype)
        mask_tmp = one_hot(mask, n_classes, 0)
        mask_reassigned = torch.zeros(mask_tmp.shape, device=mask.device)
        for src,dest in self.src_dest_labels.items():
            print(src)
            # src, dest = src_des[0], src_des[1]
            mask_reassigned[dest] += mask_tmp[src]

        for x in range(n_classes):
            mask_out[torch.isin(mask_reassigned[x], 1.0)] = x
        return mask_out 

    def remapper_sitk(self,lm):
        lm_dtype = lm.dtype
        meta = lm.meta
        lm_sitk= sitk.GetImageFromArray(lm.cpu().numpy())
        lm_sitk = relabel(lm_sitk, self.src_dest_labels)
        lm_out   = sitk.GetArrayFromImage(lm_sitk)
        lm_out = MetaTensor(lm_out, meta=meta, dtype=lm_dtype)
        return lm_out


class BoundingBoxYOLOd(MonaiDictTransform):
    """
    operates on bounding_box input as xxyy monai boundingbox
    returns a dict of bounding box axes with keys 'class','centers','size' all normalized between 0 and 1

    """

    def __init__(
        self,
        keys: KeysCollection,
        dim,
        key_template_tensor="image",
        output_keys=None,
        return_dict=False,
    ):
        super().__init__(keys)
        self.return_dict = return_dict
        self.resolve_bbox_mode(dim)
        self.dim = dim
        if output_keys is None:
            output_keys = keys
        self.output_keys = output_keys
        assert len(keys) == len(output_keys), "Same number of keys and output_keys"
        self.key_template_tensor = key_template_tensor

    def resolve_bbox_mode(self, dim):
        if dim == 2:
            self.src_mode = "xyxy"
            self.dst_mode = "ccwh"
        elif dim == 3:
            self.src_mode = "xyzxyz"
            self.dst_mode = "cccwhd"
        else:
            raise ValueError("dim must be 2 or 3")

    def __call__(self, d: dict):
        shape = d[self.key_template_tensor].shape
        shape = shape[-self.dim :]
        for key, output_key in zip(self.key_iterator(d), self.output_keys):
            d[output_key] = self.func(d[key], shape=shape)
        return d

    def func(self, bb, shape):
        box_converter = ConvertBoxMode(src_mode=self.src_mode, dst_mode=self.dst_mode)
        if self.dim == 3:
            raise NotImplementedError
        yolo_box = box_converter(bb)
        yolo_box_pt = torch.tensor(yolo_box)
        centres = yolo_box_pt[:2]
        xmax = shape[0]
        ymax = shape[1]

        xscaled = yolo_box_pt[0][::2] / xmax
        yscaled = yolo_box_pt[0][1::2] / ymax

        yolo_box_scaled = torch.zeros(4)
        yolo_box_scaled[::2] = xscaled
        yolo_box_scaled[1::2] = yscaled

        if self.return_dict == True:
            centres = yolo_box_scaled[:2]
            sizes = yolo_box_scaled[2:]
            dat = {"class": 0, "centers": centres, "size": sizes}
            return dat
        else:
            # yolo_box_scaled = yolo_box_scaled.unsqueeze(0)
            return yolo_box_scaled


def one_hot(x, classes, axis=1):
    "Creates one binay mask per class"
    return torch.stack([torch.where(x == c, 1, 0) for c in range(classes)], axis=axis)


def reassign_labels(remapping, lm):
    # input is a torch tensor. It remaps labels and copies meta information to output tensor
    # works 2% slower than converting to sitk first then using remap
    n_classes = max(remapping.values()) + 1  # include bgclass
    lm_out = torch.zeros(lm.shape, dtype=lm.dtype)
    lm_out = MetaTensor(lm_out)
    lm_out.copy_meta_from(lm)
    lm_tmp = one_hot(lm, n_classes, 0)
    lm_reassigned = torch.zeros(lm_tmp.shape, device=lm.device)
    for src, dest in remapping.items():
        lm_reassigned[dest] += lm_tmp[src]
    for x in range(n_classes):
        lm_out[torch.isin(lm_reassigned[x], 1.0)] = x
    return lm_out


class FgBgToIndicesd2(FgBgToIndicesd):
    """
    modified version. This allows 'ignore_labels' entry of fg labels which will be considered part of bg for indexing
    """

    def __init__(
        self,
        keys: KeysCollection,
        ignore_labels: list = None,
        fg_postfix: str = "_fg_indices",
        bg_postfix: str = "_bg_indices",
        image_key=None,
        image_threshold: float = 0,
        output_shape=None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(
            keys,
            fg_postfix,
            bg_postfix,
            image_key,
            image_threshold,
            output_shape,
            allow_missing_keys,
        )
        self.ignore_labels = ignore_labels

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        image = d[self.image_key] if self.image_key else None
        for key in self.key_iterator(d):
            if self.ignore_labels:
                lm = d[key].clone()  # clone so the original is untouched
                for label in self.ignore_labels:
                    lm[lm == label] = 0
                if lm.max() == 0:
                    print(
                        "Warning: No foreground in label {}".format(lm.meta["filename_or_obj"])
                    )
                    print("Not removing any labels to avoid bugs")
                    lm = d[key].clone()  # clone so the original is untouched
            else:
                lm = d[key]
            d[str(key) + self.fg_postfix], d[str(key) + self.bg_postfix] = (
                self.converter(lm, image)
            )

        return d

    # keys=["lm"], image_key="image", image_threshold=-2600)


class ApplyBBox(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        bbox_key: str,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        lm which will be overwritten by others should be first in the keys
        """

        self.bbox_key = bbox_key
        super().__init__(keys, allow_missing_keys)

    def __call__(self, d: dict):

        bbox = d[self.bbox_key]
        for key in self.key_iterator(d):
            d[key] = d[key][bbox]
        return d


class SelectLabels(MonaiDictTransform):
    def __init__(self, keys: KeysCollection, labels) -> None:
        labels = listify(labels)
        self.labels = labels
        super().__init__(keys)

    def func(self, lm):
        lm_neo = torch.zeros_like(lm)
        for label in self.labels:
            lm_neo[lm == label] = label
        return lm_neo


class LoadTorchDict(MonaiDictTransform):
    """
    when a tensor us just a dictionary stored in pt format, this returns the stored keys
    """

    def __init__(self, keys, select_keys: list = None, drop_keys=False):
        """
        select_keys will only extract mentioned keys. If none, all keys will be extracted
        """
        self.select_keys = select_keys
        self.drop_keys = drop_keys
        super().__init__(keys)

    def __call__(self, d: dict):

        mini_dict = {}
        for key in self.key_iterator(d):
            dici = torch.load(d[key],weights_only=False)
            for k in self.select_keys:
                mini_dict[k] = dici[k]
            if self.drop_keys == True:
                d.pop(key)
        d = d | mini_dict
        return d


class MetaToDict(MonaiDictTransform):
    def __init__(self, keys, meta_keys, renamed_keys=None):
        """
        keys cannot be more than len 1!
        """

        assert (
            len(keys) == 1
        ), "keys cannot be more than len 1! Otherwise duplicate keys will be created from metadatas"
        if renamed_keys is None:
            renamed_keys = meta_keys
        store_attr("meta_keys,renamed_keys")
        super().__init__(keys)

    def extract_metadata(self, tnsr):
        meta_data = {
            k1: tnsr.meta[k2] for k1, k2 in zip(self.renamed_keys, self.meta_keys)
        }
        return meta_data

    def __call__(self, d: dict):

        for key in self.key_iterator(d):
            meta_dict = self.extract_metadata(d[key])
        d.update(meta_dict)
        return d


class Recastd(MonaiDictTransform):
    def func(self, img):
        img = img.float()
        return img


class ChangeDtyped(MonaiDictTransform):
    def func(self, data):
        data = data.to(self.target_dtype)
        return data


class MergeLabelmapsd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        meta_key:str,
        key_output: str,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        lm which will be overwritten by others should be first in the keys
        meta_key: is the key whose meta will be used in the output
        """

        self.key_output = key_output
        self.meta_key = key_output
        assert len(keys) == 2, "Only allows 2 keys, i.e., 2 pt lms to merge"
        super().__init__(keys, allow_missing_keys)

    def __call__(self, d: dict):

        meta = d[self.meta_key].meta
        lms = []
        for key in self.key_iterator(d):
            lms.append(d[key])
        lm_out = merge_pt(lms[0], lms[1])
        lm_out.meta = meta
        d[self.key_output] = lm_out
        return d


class LabelRemapd(MapTransform):
    """
    Works on Metatensor. Accepts remapping dictionary as in sitk

    """

    def __init__(
        self,
        keys: KeysCollection,
        remapping_key: str,
        allow_missing_keys: bool = False,
    ) -> None:
        self.remapping_key = remapping_key
        super().__init__(keys, allow_missing_keys)

    def need_remapping(self, remapping):
        if remapping is None:
            return False
        else:
            same = [a == b for a, b in remapping.items()]
            return not all(same)

    def __call__(self, d: dict):
        remapping = d[self.remapping_key]

        for key in self.key_iterator(d):
            d[key] = self.func(d[key], remapping)
        return d

    def func(self, lm, remapping):
        if self.need_remapping(remapping):
            lm_sitk = sitk.GetImageFromArray(lm)
            lm_sitk = relabel(lm_sitk, remapping)
            lm_np = sitk.GetArrayFromImage(lm_sitk)
            lm_pt = torch.tensor(lm_np)
            lm_out = MetaTensor(lm_pt)
            lm_out.copy_meta_from(lm)
            return lm_out
        return lm


class LabelRemapSITKd(LabelRemapd):
    """
    input can be a file or Image
    """

    def func(self, lm, remapping):
        if isinstance(lm, Union[str, Path]):
            lm = sitk.ReadImage(lm)
        if self.need_remapping(remapping):
            lm = relabel(lm, remapping)
        return lm


class HalfPrecisiond(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, d: dict):
        for key in self.key_iterator(d):
            d[key] = d[key].to(torch.float16)
        return d


class DictToMeta(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        meta_keys,
        renamed_keys=None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        if renamed_keys, meta_keys will be renamed from the list
        """
        if renamed_keys is None:
            renamed_keys = meta_keys

        super().__init__(keys, allow_missing_keys)
        store_attr("meta_keys,renamed_keys")

    def extract_metadata(self, d: dict):
        meta_data = {k1: d[k2] for k1, k2 in zip(self.renamed_keys, self.meta_keys)}
        return meta_data

    def __call__(self, d: dict):
        meta_data = self.extract_metadata(d)
        for key in self.key_iterator(d):
            d[key].meta.update(meta_data)
        return d


def create_augmentations(after_item_intensity: dict, after_item_spatial: dict):
    intensity_augs = []
    spatial_augs = []
    probabilities_intensity = []
    probabilities_spatial = []
    for key, value in after_item_intensity.items():
        func = getattr(intensity, key)
        out_fnc = partial(func, factor_range=value[0])
        intensity_augs.append(out_fnc)
        probabilities_intensity.append(value[1])

    for key, value in after_item_spatial.items():
        spatial_augs.append(getattr(spatial, key))
        probabilities_spatial.append(value)
    return intensity_augs, spatial_augs


class FilenameFromBBox(ItemTransform):
    def encodes(self, x):
        img, mask, bbox = x
        fname = str(bbox["filename"])
        return img, mask, fname


class Squeeze(ItemTransform):

    def __init__(self, dim):
        store_attr()

    def encodes(self, x):
        outputs = []
        for tensr in x:
            tensr = tensr.squeeze(self.dim)
            outputs.append(tensr)
        return outputs

    def decodes(self, x):
        outputs = []
        for tensr in x:
            tensr = tensr.unsqueeze(self.dim)
            outputs.append(tensr)
        return outputs


# %%
if __name__ == "__main__":
    img_fn = Path(
        "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/images/drli_006.pt"
    )
    lm_fn = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/lms/drli_006.pt")

    img = torch.load(img_fn)
    lm = torch.load(lm_fn)

    fn = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/indices_fg_exclude_1/drli_002.pt"
    tnser = torch.load(fn)
    tnser["lm_fg_indices"] = tnser["lm_fg_indicesmask_label"]
    tnser["lm_bg_indicesmask_label"].pop()
    dici = {"indices": tnser}
    T = TensorToDict(keys=["indices"], select_keys=["lm_fg_indices", "lm_bg_indices"])
    dici = T(dici)

    # %%
    F = FgBgToIndicesd2(keys=["lm"], ignore_labels=[1])
    dici = {"image": img, "lm": lm}
    dici = F(dici)
# %%
