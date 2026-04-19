from __future__ import annotations

from collections.abc import Hashable, Mapping

from monai.transforms.utility.dictionary import FgBgToIndicesd
import torch
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms.transform import MapTransform


class FgBgToIndicesd2(FgBgToIndicesd):
    """
    modified version. This allows 'ignore_labels' entry of fg labels which will be considered part of bg for indexing
    """

    def __init__(
        self,
        keys: KeysCollection,
        ignore_labels: list | int = [],
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
        if isinstance(ignore_labels, int):
            ignore_labels = [ignore_labels]
        self.ignore_labels = ignore_labels

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        def _plain_tensor(x):
            # FgBgToIndices from MONAI can fail on batched MetaTensor metadata collation.
            # Convert to raw tensor for robust indexing/collation.
            if hasattr(x, "as_tensor"):
                x = x.as_tensor()
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x)

        d = dict(data)
        image = _plain_tensor(d[self.image_key]) if self.image_key else None
        for key in self.key_iterator(d):
            lm_src = d[key]
            if self.ignore_labels:
                lm = _plain_tensor(lm_src).clone()
                for label in self.ignore_labels:
                    lm[lm == label] = 0
                if lm.max() == 0:
                    fname = None
                    meta = getattr(lm_src, "meta", None)
                    if isinstance(meta, dict):
                        fname = meta.get("filename_or_obj")
                    print("Warning: No foreground in label {}".format(fname))
                    print("Not removing any labels to avoid bugs")
                    lm = _plain_tensor(lm_src).clone()
            else:
                lm = _plain_tensor(lm_src)
            d[str(key) + self.fg_postfix], d[str(key) + self.bg_postfix] = (
                self.converter(lm, image)
            )

        return d

    # keys=["lm"], image_key="image", image_threshold=-2600)



class FgBgToIndicesSubsampled(FgBgToIndicesd2):
    """
    Compute foreground/background indices, optionally keeping every Nth bg index.

    `subsample_bg=None` preserves stock `FgBgToIndicesd2` output.
    """

    def __init__(
        self,
        keys: KeysCollection,
        ignore_labels: list[int] | int | None = None,
        subsample_bg: int | None = 5,
        fg_postfix: str = "_fg_indices",
        bg_postfix: str = "_bg_indices",
        image_key=None,
        image_threshold: float = 0,
        output_shape=None,
        allow_missing_keys: bool = False,
    ) -> None:
        if ignore_labels is None:
            ignore_labels = []
        elif isinstance(ignore_labels, int):
            ignore_labels = [ignore_labels]
        else:
            ignore_labels = list(ignore_labels)
        self.subsample_bg = self._validate_subsample_bg(subsample_bg)
        super().__init__(
            keys=keys,
            ignore_labels=ignore_labels,
            fg_postfix=fg_postfix,
            bg_postfix=bg_postfix,
            image_key=image_key,
            image_threshold=image_threshold,
            output_shape=output_shape,
            allow_missing_keys=allow_missing_keys,
        )

    @staticmethod
    def _validate_subsample_bg(subsample_bg: int | None) -> int | None:
        if subsample_bg is None:
            return None
        subsample_bg = int(subsample_bg)
        if subsample_bg < 1:
            raise ValueError("subsample_bg must be None or an int >= 1")
        return subsample_bg

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = super().__call__(data)
        if self.subsample_bg is None or self.subsample_bg == 1:
            return d
        for key in self.key_iterator(d):
            bg_key = str(key) + self.bg_postfix
            d[bg_key] = d[bg_key][:: self.subsample_bg].contiguous()
        return d
