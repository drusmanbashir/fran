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
from fran.utils.imageviewers import ImageMaskViewer

tr = ipdb.set_trace

from fastcore.basics import store_attr
from fastcore.transform import ItemTransform, store_attr

import fran.transforms.intensitytransforms as intensity
import fran.transforms.spatialtransforms as spatial





class FgBgToIndicesd2(FgBgToIndicesd):
    '''
    modified version. This allows 'ignore_labels' entry of fg labels which will be considered part of bg for indexing
    '''
    
    def __init__(
        self,
        keys: KeysCollection,
        ignore_labels: list=None,
        fg_postfix: str = "_fg_indices",
        bg_postfix: str = "_bg_indices",
        image_key  = None,
        image_threshold: float = 0,
        output_shape= None,
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
                lm = d[key].clone() # clone so the original is untouched
                for label in self.ignore_labels:
                    lm[lm == label] = 0
                if lm.max()==0: 
                        print("Warning: No foreground in label {}".format(lm.meta['filename']))
                        print("Not removing any labels to avoid bugs")
                        lm = d[key].clone() # clone so the original is untouched
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

class LoadDict(MonaiDictTransform):
    '''
    when a tensor us just a dictionary stored in pt format, this returns the stored keys
    '''
    def __init__(self, keys, select_keys:list=None,drop_keys=False):
        '''
        select_keys will only extract mentioned keys. If none, all keys will be extracted
        '''
        self.select_keys = select_keys
        self.drop_keys = drop_keys
        super().__init__(keys)

    def __call__(self, d: dict):

        mini_dict ={}
        for key in self.key_iterator(d):
            dici = torch.load(d[key])
            for k in self.select_keys:
                mini_dict[k] = dici[k]
            if self.drop_keys==True:
                d.pop(key)
        d = d|mini_dict
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
if __name__ == "__main__":
    img_fn = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/images/drli_006.pt")
    lm_fn = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/lms/drli_006.pt")

    img =  torch.load(img_fn)
    lm = torch.load(lm_fn)

    fn = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/indices_fg_exclude_1/drli_002.pt"
    tnser = torch.load(fn)
    tnser["lm_fg_indices"] = tnser["lm_fg_indicesmask_label"]
    tnser["lm_bg_indicesmask_label"].pop()
    dici = {"indices": tnser}
    T = TensorToDict(keys= ["indices"],select_keys =  ["lm_fg_indices","lm_bg_indices"])
    dici = T(dici)

# %%
    F = FgBgToIndicesd2(keys = ['lm'], ignore_labels=[1])
    dici = {'image':img, 'lm':lm}
    dici = F(dici)
# %%
