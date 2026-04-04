# %%
from typing import Union

import numpy as np
import torch
from monai.config.type_definitions import KeysCollection
from monai.transforms.transform import MapTransform
from torch.functional import Tensor


def listify(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


class Transform:
    def __call__(self, *args, **kwargs):
        return self.encodes(*args, **kwargs)


class ItemTransform(Transform):
    pass


class Pipeline:
    def __init__(self, funcs):
        self.funcs = list(funcs)

    def __call__(self, x):
        for func in self.funcs:
            x = func(x)
        return x


class MonaiDictTransform(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        **kwargs,
    ) -> None:
        for key, val in kwargs.items():
            setattr(self, key, val)
        super().__init__(keys, False)

    def __call__(self, d: dict):
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
        return d

    def func(self, data):
        raise NotImplementedError


class Squeeze(ItemTransform):
    def __init__(self, dim):
        self.dim = dim

    def encodes(self, x):
        outputs = []
        for tensr in x:
            outputs.append(tensr.squeeze(self.dim))
        return outputs

    def decodes(self, x):
        outputs = []
        for tensr in x:
            outputs.append(tensr.unsqueeze(self.dim))
        return outputs


class KeepBBoxTransform(ItemTransform):
    def encodes(self, x: Union[list, tuple]):
        if not isinstance(x[-1], (Tensor, np.ndarray)):
            if len(x) == 2:
                y = [self.func(x[0])]
            elif len(x) > 2:
                y = self.func(x[:-1])
            else:
                y = self.func(x)
            y = listify(y)
            y.append(x[-1])
            return y
        return self.func(x)


class TrainingAugmentations(KeepBBoxTransform):
    def __init__(self, augs, p: Union[float, list] = 0.2):
        augs = listify(augs)
        if isinstance(p, float):
            p = [p] * len(augs)
        assert len(p) == len(augs), (
            "Either provide a single probability for all augs, or a list of probabilities of equal length as augs"
        )
        self.augs = augs
        self.p = p
        super().__init__()

    def func(self, x):
        for aug, p in zip(self.augs, self.p):
            if np.random.rand() < p:
                x = aug(x)
        return x


class GenericPairedOrganTransform(ItemTransform):
    def __init__(self, func):
        self.func = func

    def encodes(self, x):
        final_img_mask_pairs = []
        for img_mask_pair in x:
            final_img_mask_pairs.append(self.func(img_mask_pair))
        return final_img_mask_pairs

