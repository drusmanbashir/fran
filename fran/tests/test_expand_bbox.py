import numpy as np
import pytest
import torch
from monai.data.meta_tensor import MetaTensor

from fran.transforms.spatialtransforms import ExpandBBox


def make_template_tensor(suffix):
    tensor = MetaTensor(
        torch.zeros((1, 80, 80), dtype=torch.float32),
        affine=torch.diag(torch.tensor([2.0, 3.0, 5.0, 1.0], dtype=torch.float32)),
    )
    tensor.meta["project2d"] = {"suffix": suffix}
    return tensor


def test_expand_bbox_ap_projection_expands_lat_only():
    transform = ExpandBBox(
        bbox_key="bbox",
        axes="lat",
        expand_by=5,
        template_tensor_key="template",
    )
    data = {
        "bbox": np.array([[10, 20, 30, 40]], dtype=np.int64),
        "template": make_template_tensor("ap"),
    }
    out = transform(data)
    assert out["bbox"].tolist() == [[7, 20, 33, 40]]


def test_expand_bbox_lat_projection_expands_ap_only():
    transform = ExpandBBox(
        bbox_key="bbox",
        axes="ap",
        expand_by=5,
        template_tensor_key="template",
    )
    data = {
        "bbox": torch.tensor([[10, 20, 30, 40]], dtype=torch.int64),
        "template": make_template_tensor("lat"),
    }
    out = transform(data)
    assert out["bbox"].tolist() == [[8, 20, 32, 40]]


def test_expand_bbox_zero_expand_by_is_identity():
    bbox = torch.tensor([[10, 20, 30, 40]], dtype=torch.int64)
    transform = ExpandBBox(
        bbox_key="bbox",
        axes="ap,lat",
        expand_by=0,
        template_tensor_key="template",
    )
    data = {"bbox": bbox.clone(), "template": make_template_tensor("ap")}
    out = transform(data)
    assert torch.equal(out["bbox"], bbox)


def test_expand_bbox_invalid_axes_type_raises_type_error():
    with pytest.raises(TypeError):
        ExpandBBox(
            bbox_key="bbox",
            axes=("ap", "lat"),
            expand_by=5,
            template_tensor_key="template",
        )
