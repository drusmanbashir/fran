from torch import Tensor

from .transforms import clamp_labels, normalize
from .types import Batch


def training_step(b: Batch) -> Tensor:
    x = b["image"]
    x = normalize(x)  # error (if normalize returns float)
    b = clamp_labels(b)  # error (None handling)
    return x
