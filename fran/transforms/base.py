# %%
from typing import Union

import numpy as np
import torch
from monai.config.type_definitions import KeysCollection
from monai.transforms.transform import MapTransform


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


class KeepBBoxTransform(ItemTransform):
    def encodes(self, x: Union[list, tuple]):
        if not isinstance(x[-1], (torch.Tensor, np.ndarray)):
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
