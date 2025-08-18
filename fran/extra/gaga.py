from torch import Tensor
from .types import Batch
from .transforms import normalize, clamp_labels
def training_step(b: Batch) -> Tensor:
    x = b["image"]
    x = normalize(x)           # error (if normalize returns float)
    b = clamp_labels(b)        # error (None handling)
    return x
