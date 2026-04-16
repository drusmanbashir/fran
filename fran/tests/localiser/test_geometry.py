import pytest

from fran.localiser.geometry import (
    ComputeCropSlicesFromYolo,
    bbox_center_size_to_bounds,
    compute_crop_slices_from_yolo,
    invert_resize_bbox,
)


def slice_tuple(slices):
    return tuple((s.start, s.stop) for s in slices)


def test_letterbox_false_centered_boxes():
    slices = compute_crop_slices_from_yolo(
        shape_3d=(100, 200, 300),
        bbox_cor=(0.5, 0.5, 0.2, 0.4),
        bbox_sag=(0.5, 0.5, 0.3, 0.2),
        proj_shape_cor=(100, 300),
        proj_shape_sag=(100, 200),
        letterbox=False,
    )

    assert slice_tuple(slices) == ((30, 70), (70, 130), (120, 180))


def test_letterbox_true_padded_case():
    slices = compute_crop_slices_from_yolo(
        shape_3d=(100, 200, 300),
        bbox_cor=(0.5, 0.5, 0.2, 2 / 15),
        bbox_sag=(0.5, 0.5, 0.4, 0.3),
        proj_shape_cor=(100, 300),
        proj_shape_sag=(100, 200),
        letterbox=True,
    )

    assert slice_tuple(slices) == ((20, 80), (60, 140), (120, 180))


def test_border_clamp_case():
    slices = compute_crop_slices_from_yolo(
        shape_3d=(50, 60, 70),
        bbox_cor=(0.05, 0.1, 0.2, 0.4),
        bbox_sag=(0.95, 0.9, 0.2, 0.4),
        proj_shape_cor=(50, 70),
        proj_shape_sag=(50, 60),
        target_size=100,
        letterbox=False,
    )

    assert slice_tuple(slices) == ((0, 50), (51, 60), (0, 11))


def test_different_z_extents_merge_outer_union():
    slices = compute_crop_slices_from_yolo(
        shape_3d=(128, 80, 64),
        bbox_cor=(0.625, 30 / 128, 20 / 64, 20 / 128),
        bbox_sag=(0.375, 85 / 128, 0.5, 30 / 128),
        proj_shape_cor=(128, 64),
        proj_shape_sag=(128, 80),
        target_size=128,
        letterbox=False,
    )

    assert slice_tuple(slices) == ((20, 100), (10, 50), (30, 50))


def test_exact_numeric_regression_case():
    bbox_cor_src = invert_resize_bbox(
        (0.61, 0.43, 0.22, 0.18),
        src_shape=(101, 307),
        letterbox=True,
    )
    bbox_sag_src = invert_resize_bbox(
        (0.37, 0.52, 0.31, 0.24),
        src_shape=(101, 211),
        letterbox=True,
    )

    assert bbox_cor_src == pytest.approx((187.27, 29.01, 67.54, 55.26))
    assert bbox_sag_src == pytest.approx((78.07, 54.72, 65.41, 50.64))
    assert bbox_center_size_to_bounds(bbox_cor_src) == pytest.approx(
        (153.5, 221.04, 1.38, 56.64)
    )

    wrapper = ComputeCropSlicesFromYolo(letterbox=True)
    slices = wrapper(
        shape_3d=(101, 211, 307),
        bbox_cor=(0.61, 0.43, 0.22, 0.18),
        bbox_sag=(0.37, 0.52, 0.31, 0.24),
        proj_shape_cor=(101, 307),
        proj_shape_sag=(101, 211),
    )

    assert slice_tuple(slices) == ((1, 81), (45, 111), (153, 222))
