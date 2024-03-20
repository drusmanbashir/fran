from functools import partial
import torch
import ipdb
from label_analysis.helpers import relabel
from monai.config.type_definitions import KeysCollection
from monai.transforms.transform import MapTransform

tr = ipdb.set_trace

from fastcore.basics import store_attr
from fastcore.transform import ItemTransform

import fran.transforms.intensitytransforms as intensity
import fran.transforms.spatialtransforms as spatial

from fastcore.transform import ItemTransform, store_attr



class RemapSITKImage(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def need_remapping(self,remapping):
        same = [a==b for a,b in remapping.items()]
        return not all(same)

    def __call__(self, d: dict):
        remapping = d['remapping']

        for key in self.key_iterator(d):
            d[key] = self.func(d[key],remapping)
        return d

    def func(self, lm, remapping):
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

class AddMetadata(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        meta_keys,
        renamed_keys=None,
        allow_missing_keys: bool = False,
    ) -> None:
        '''
        if renamed_keys, meta_keys will be renamed from the list
        '''
        if renamed_keys is None: renamed_keys=meta_keys
        
        super().__init__(keys, allow_missing_keys)
        store_attr('meta_keys,renamed_keys')


    def extract_metadata(self,d:dict):
        meta_data = {k1:d[k2] for k1,k2 in zip(self.renamed_keys, self.meta_keys)}
        return meta_data

    def __call__(self, d: dict):
        meta_data = self.extract_metadata(d)
        for key in self.key_iterator(d):
            d[key].meta.update(meta_data)
        return d


def create_augmentations(after_item_intensity:dict, after_item_spatial:dict):
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
    def encodes(self,x):
        img,mask,bbox = x
        fname = str(bbox['filename'])
        return img,mask, fname

class Squeeze(ItemTransform):

    def __init__(self, dim):
        store_attr()
    def encodes(self,x):
        outputs =[]
        for tensr in x:
            tensr= tensr.squeeze(self.dim)
            outputs.append(tensr)
        return outputs

    def decodes(self,x):
        outputs =[]
        for tensr in x:
            tensr= tensr.unsqueeze(self.dim)
            outputs.append(tensr)
        return outputs

# %%
