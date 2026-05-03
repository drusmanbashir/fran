import torch

from fran.transforms.spatialtransforms import CropForegroundMinShaped, CropMaybePad
from utilz.image_utils import margin_mm_to_vox


def test_margin_mm_to_vox_uses_ceil_per_axis():
    out = margin_mm_to_vox(2.1, (2.0, 0.75, 2.0))

    assert tuple(out) == (2, 3, 2)


def test_crop_maybe_pad_adds_margin_then_expands_bbox_inside_bounds():
    helper = CropMaybePad(keys=["image"], min_shape=(6, 6, 6), margin=2.0)
    image = torch.zeros((1, 8, 10, 12))

    box_start, box_end = helper.add_margin_to_bbox(
        box_start=(1, 4, 5),
        box_end=(4, 6, 7),
        image_shape=helper.spatial_shape(image),
        spacing=(2.0, 1.0, 1.0),
    )
    box_start, box_end = helper.maybe_expand_bbox(
        box_start=box_start,
        box_end=box_end,
        image_shape=helper.spatial_shape(image),
    )

    assert box_start == (0, 2, 3)
    assert box_end == (6, 8, 9)


def test_crop_maybe_pad_pads_full_image_to_exact_min_shape():
    helper = CropMaybePad(keys=["image"], min_shape=(6, 6, 6), margin=0)
    image = torch.arange(1 * 3 * 4 * 5).reshape(1, 3, 4, 5)

    out = helper(
        {"image": image},
        box_start=(0, 0, 0),
        box_end=(3, 4, 5),
        spacing=(1.0, 1.0, 1.0),
    )["image"]

    assert out.shape == (1, 6, 6, 6)
    assert torch.equal(out[:, 1:4, 1:5, 0:5], image)


def test_crop_maybe_pad_uses_smaller_left_pad_for_odd_deficits():
    helper = CropMaybePad(keys=["image"], min_shape=(6, 8, 8), margin=0)
    image = torch.ones((2, 3, 4, 5))

    out = helper.maybe_pad(image)

    assert out.shape == (2, 6, 8, 8)
    assert torch.equal(out[:, 1:4, 2:6, 1:6], image)
    assert torch.count_nonzero(out[:, :1, :, :]) == 0
    assert torch.count_nonzero(out[:, 4:, :, :]) == 0
    assert torch.count_nonzero(out[:, :, :2, :]) == 0
    assert torch.count_nonzero(out[:, :, 6:, :]) == 0
    assert torch.count_nonzero(out[:, :, :, :1]) == 0
    assert torch.count_nonzero(out[:, :, :, 6:]) == 0


def test_crop_foreground_min_shaped_uses_shared_crop_pad_flow():
    image = torch.arange(1 * 4 * 5 * 6, dtype=torch.float32).reshape(1, 4, 5, 6)
    lm = torch.zeros((1, 4, 5, 6), dtype=torch.uint8)
    lm[:, 1:3, 2:4, 1:4] = 1
    cropper = CropForegroundMinShaped(
        keys=["image", "lm"],
        source_key="lm",
        min_shape=(6, 6, 6),
        margin=0,
    )

    out = cropper({"image": image, "lm": lm})

    assert out["image"].shape == (1, 6, 6, 6)
    assert out["lm"].shape == (1, 6, 6, 6)
    assert int(torch.count_nonzero(out["lm"]).item()) == int(torch.count_nonzero(lm).item())
