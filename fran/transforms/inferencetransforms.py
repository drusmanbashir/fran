# %%
from pathlib import Path
from collections.abc import Hashable, Mapping
from typing import Any, Sequence

from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.data.meta_tensor import MetaTensor
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.post.dictionary import KeepLargestConnectedComponentd
from monai.transforms.transform import MapTransform
from monai.transforms.utils import generate_spatial_bounding_box
from utilz.stringz import strip_extension

import torch


class SqueezeListofListsd(MapTransform):
    """
    If the value at `keys` is a list containing a single list of slices,
    unwrap it so that [[slice(...), ...]] becomes [slice(...), ...].
    Leaves other cases unchanged.
    """

    def __init__(self, keys: Sequence[Hashable]):
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, Any]):
        d = dict(data)
        for key in self.keys:
            if key in d:
                val = d[key]
                # check for [[slice,...]]
                if (
                    isinstance(val, list)
                    and len(val) == 1
                    and isinstance(val[0], list)
                    and all(isinstance(x, slice) for x in val[0])
                ):
                    d[key] = val[0]
        return d


class MakeWritabled(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, MetaTensor]):
        d = dict(data)
        for k in self.key_iterator(d):
            # turn “inference tensors” into normal writable tensors
            d[k] = d[k].clone()
        return d


class KeepLargestConnectedComponentWithMetad(KeepLargestConnectedComponentd):
    def __init__(
        self,
        keys,
        applied_labels=None,
        is_onehot=None,
        independent: bool = True,
        connectivity=None,
        num_components: int = 1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(
            keys,
            applied_labels,
            is_onehot,
            independent,
            connectivity,
            num_components,
            allow_missing_keys,
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            meta = d[key].meta
            d[key] = self.converter(d[key].clone())
            d[key].meta = meta
        return d


class RenameDictKeys(MapTransform):
    def __init__(
        self, keys: KeysCollection, new_keys, allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.new_keys = new_keys

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        old_keys = list(set(self.key_iterator(d)))
        new_keys = list(set((self.new_keys)))
        for old, new in zip(old_keys, new_keys):
            d[new] = d.pop(old)
        return d


class ToCPUd(MapTransform):
    def __call__(self, d: dict):
        for key in self.key_iterator(d):
            d[key] = d[key].cpu()
        return d


class BBoxFromPTd(MapTransform):
    # CODE: this should accept margin as arg instead of both spacing and expand_by, just l9ike CropToForeground
    def __init__(
        self,
        spacing,
        expand_by: int,  # in millimeters
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.spacing = spacing
        self.expand_by = expand_by

    def __call__(self, d: dict):
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
            d["bounding_box"] = d[key].meta["bounding_box"]
        return d

    def func(self, img):
        add_to_bbox = [int(self.expand_by / sp) for sp in self.spacing]
        bb = generate_spatial_bounding_box(
            img, channel_indices=0, margin=add_to_bbox, allow_smaller=True
        )
        sls = [slice(0, 100, None)] + [slice(a, b, None) for a, b in zip(*bb)]
        img.meta["bounding_box"] = sls
        return img


class SaveMultiChanneld(SaveImaged):
    def __call__(self, data):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(
            d, self.meta_keys, self.meta_key_postfix
        ):
            if meta_key is None and meta_key_postfix is not None:
                meta_key = f"{key}_{meta_key_postfix}"
            meta_data = d.get(meta_key) if meta_key is not None else None
            img = d[key]
            imgs = torch.unbind(img)
            for ind, img in enumerate(imgs):
                img.meta = self.apply_postfix(img.meta, ind)
                self.saver(img=img, meta_data=meta_data)
        return d

    def apply_postfix(self, img_meta, ind):
        fname = Path(img_meta["filename_or_obj"])
        fname_neo = strip_extension(fname.name) + "_{}".format(ind)
        fname_neo = fname_neo + ".nii.gz"
        fname_neo = fname.parent / fname_neo
        img_meta["filename_or_obj"] = str(fname_neo)
        return img_meta


# %%
# %%
