
# %%
from typing import Union
from fastcore.basics import listify, store_attr
from fastcore.transform import ItemTransform
from monai.config.type_definitions import KeysCollection
from monai.transforms.transform import MapTransform
import numpy as np
import torch
from torch.functional import Tensor 
import ipdb

import itertools as il

from fran.utils.dictopts import DictToAttr

tr = ipdb.set_trace


class MonaiDictTransform(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        **kwargs,
    ) -> None:
        for key,val in kwargs.items():
                    setattr(self,key,val)
        super().__init__(keys, False)
    def __call__(self, d: dict):
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
        return d

    def func(self,data):
        pass




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
 
class KeepBBoxTransform(ItemTransform):

    def encodes(self,x:Union[list,tuple]):
        if not isinstance(x[-1],(Tensor,np.ndarray)): # may be dict / list or str
            if len(x)==2:
                y = [self.func(x[0])]
            elif len(x)>2:
                y = self.func(x[:-1])
            else:
                y = self.func(x)
            y = listify(y)
            y.append(x[-1])
            return y
        else: return self.func(x)


class ValidAndTrainingTransform(ItemTransform):

    def __init__(self, aug):  #type: ignore
        store_attr()

    def encodes(self, x):
        if np.random.rand() < self.p:
            x = self.aug(x)
        return x

class TrainingAugmentations(KeepBBoxTransform):
    # DO NOT SET SPLIT_IDX IF SEPARATE TRAIN_DL AND VALID_DL ARE FORMED

    def __init__(self, augs, p: Union[float, list] = 0.2):
        augs = listify(augs)
        if isinstance(p, float):
            p = [p] * len(augs)
        assert len(p) == len(
            augs
        ), "Either provide a single probability for all augs, or a list of probabilities of equal length as augs"
        store_attr()
        super().__init__()

    def func(self, x):
        for aug, p in zip(self.augs, self.p):
            if np.random.rand() < p:
                x = aug(x)
        return x


class TrainingAugmentationsListOfLists(TrainingAugmentations):
    '''
    This transform  expects each input list to contain sub-lists, one for each organ. For example encodes(self,x)-> x[0] is one img/mask pair and x[1] is another, and so on..
    '''

    def encodes(self, x):
        final_img_mask_pairs = []
        for img_mask_pair in x:
            final_img_mask_pairs.append(super().encodes(img_mask_pair))

        return final_img_mask_pairs


class GenericPairedOrganTransform(ItemTransform):

    def __init__(self, func):
        store_attr()

    def encodes(self, x):
        final_img_mask_pairs = []
        for img_mask_pair in x:
            img_mask_pair_new = self.func(img_mask_pair)
            final_img_mask_pairs.append(img_mask_pair_new)
        return final_img_mask_pairs


class FixDType(ItemTransform):
    def encodes(self,x):
        img,mask = x
        if not img.dtype==torch.float32:
            img=img.float()
        return img,mask

# %%
if __name__ == "__main__":
# %%
    fn = "/home/ub/datasets/preprocessed/litsmc/patches/spc_080_080_150/dim_192_192_128/masks/lits_129ub_17.pt"
    mask = torch.load(fn)
    dici = {'mask':mask}
    C = ChangeDtype(keys=['mask'], target_dtype=torch.uint8)
    # C = MonaiDictTransform(keys=['mask'], target_dtype=torch.int8)
    dici2 = C(dici)
# %%
