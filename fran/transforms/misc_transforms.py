# %%
from collections.abc import Hashable, Mapping
from functools import partial
from pathlib import Path
from typing import Sequence, Union

import ipdb
import SimpleITK as sitk
import torch
from label_analysis.helpers import listify, relabel
from label_analysis.merge import merge_pt
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms.transform import MapTransform
from monai.transforms.utility.dictionary import FgBgToIndicesd

from fran.transforms.base import MonaiDictTransform

tr = ipdb.set_trace

from fastcore.basics import store_attr
from fastcore.transform import ItemTransform, store_attr

import fran.transforms.intensitytransforms as intensity
import fran.transforms.spatialtransforms as spatial


class FgBgToIndicesd2(FgBgToIndicesd):
    def __init__(
        self,
        keys: KeysCollection,
        ignore_labels: list,
        fg_postfix: str = "_fg_indices",
        bg_postfix: str = "_bg_indices",
        image_key: str | None = None,
        image_threshold: float = 0,
        output_shape: Sequence[int] | None = None,
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
            lm = d[key]
            tr()
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


class Recast(MonaiDictTransform):
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
        key_output: str,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        lm which will be overwritten by others should be first in the keys
        """

        self.key_output = key_output
        assert len(keys) == 2, "Only allows 2 keys, i.e., 2 pt lms to merge"
        super().__init__(keys, allow_missing_keys)

    def __call__(self, d: dict):

        lms = []
        for key in self.key_iterator(d):
            lms.append(d[key])
        lm_out = merge_pt(lms[0], lms[1])
        d[self.key_output] = lm_out
        return d


class RemapSITK(MapTransform):
    """
    input can be a file or Image
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
        same = [a == b for a, b in remapping.items()]
        return not all(same)

    def __call__(self, d: dict):
        remapping = d[self.remapping_key]

        for key in self.key_iterator(d):
            d[key] = self.func(d[key], remapping)
        return d

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
