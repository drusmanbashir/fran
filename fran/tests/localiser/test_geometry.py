import pytest

from fran.localiser.geometry import (
    ComputeCropSlicesFromYolo,
    ComputeOrientedCropSlicesFromYolo,
    bbox_center_size_to_bounds,
    clamp_bounds_to_slice,
    compute_crop_slices_from_yolo,
    compute_oriented_crop_slices_from_boxes_xyxy,
    compute_oriented_crop_slices_from_yolo,
    invert_resize_bbox,
    projection_bbox_from_xyxy,
    select_yolo_bbox_xywh,
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

    assert slice_tuple(slices) == ((15, 35), (51, 60), (0, 11))


def test_different_z_extents_merge_larger_height_about_mean_center():
    slices = compute_crop_slices_from_yolo(
        shape_3d=(128, 80, 64),
        bbox_cor=(0.625, 30 / 128, 20 / 64, 20 / 128),
        bbox_sag=(0.375, 85 / 128, 0.5, 30 / 128),
        proj_shape_cor=(128, 64),
        proj_shape_sag=(128, 80),
        target_size=128,
        letterbox=False,
    )

    assert slice_tuple(slices) == ((42, 73), (10, 50), (30, 50))


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

    assert slice_tuple(slices) == ((14, 70), (45, 111), (153, 222))


def test_oriented_slices_use_post_orientation_axis_order_and_letterbox_metadata():
    meta = {
        "letterbox_orig_size": (61, 512),
        "letterbox_resized_size": (30, 256),
        "letterbox_padded": ((0, 0), (113, 113), (0, 0)),
    }
    bbox_cor = (133.6191 / 256, 127.8429 / 256, 139.7419 / 256, 66.0803 / 256)
    bbox_sag = (140.1458 / 256, 127.9499 / 256, 177.4849 / 256, 66.4521 / 256)

    slices = compute_oriented_crop_slices_from_yolo(
        spatial_shape_3d=(512, 512, 61),
        bbox_cor=bbox_cor,
        bbox_sag=bbox_sag,
        meta_cor=meta,
        meta_sag=meta,
        letterbox=True,
    )

    assert slice_tuple(slices) == ((102, 458), (127, 407), (0, 61))

    wrapper = ComputeOrientedCropSlicesFromYolo(letterbox=True)
    assert slice_tuple(
        wrapper(
            spatial_shape_3d=(512, 512, 61),
            bbox_cor=bbox_cor,
            bbox_sag=bbox_sag,
            meta_cor=meta,
            meta_sag=meta,
        )
    ) == ((102, 458), (127, 407), (0, 61))


def test_select_yolo_bbox_xywh_filters_class_and_confidence():
    class Boxes:
        xywhn = [
            (0.1, 0.2, 0.3, 0.4),
            (0.5, 0.6, 0.7, 0.8),
            (0.9, 0.8, 0.7, 0.6),
        ]
        cls = [1, 2, 2]
        conf = [0.9, 0.2, 0.8]

    assert select_yolo_bbox_xywh(Boxes(), class_id=2) == (0.9, 0.8, 0.7, 0.6)


def test_select_yolo_bbox_xywh_missing_class_fails_observably():
    class Boxes:
        xywhn = [(0.1, 0.2, 0.3, 0.4)]
        cls = [1]

    with pytest.raises(ValueError, match="No YOLO boxes matched"):
        select_yolo_bbox_xywh(Boxes(), class_id=2)


def test_clamp_bounds_preserves_one_index_at_boundary():
    assert slice_tuple((clamp_bounds_to_slice(80, 90, 61),)) == ((60, 61),)
    assert slice_tuple((clamp_bounds_to_slice(-20, -10, 61),)) == ((0, 1),)


def test_projection_bbox_from_xyxy_filters_repeated_classes_and_unletterboxes():
    class Boxes:
        orig_shape = (256, 256)
        cls = [3, 4, 5, 5]
        conf = [0.9577, 0.8330, 0.5769, 0.5045]
        xyxy = [
            (65.7083, 60.0278, 206.4184, 197.7821),
            (55.2819, 58.1271, 183.5392, 141.5031),
            (65.6615, 181.2122, 129.9661, 197.8898),
            (66.5323, 175.2623, 125.4538, 197.6352),
        ]

    meta = {
        "letterbox_padded": ((0, 0), (58, 58), (0, 0)),
        "letterbox_orig_size": (263, 482),
        "letterbox_resized_size": (140, 256),
        "projection_key": "image1",
    }

    out = projection_bbox_from_xyxy(Boxes(), classes=[3, 4, 5], proj_dict=meta)

    assert out["selected_indices"] == [0, 1, 2]
    assert out["xyxy"] == pytest.approx((55.2819, 58.1271, 206.4184, 197.8898))
    assert out["source_xyxy"] == pytest.approx(
        (104.085452, 0.238766, 388.647144, 262.792981), abs=1e-6
    )


def test_compute_oriented_crop_slices_from_boxes_xyxy_with_provided_values():
    class Boxes1:
        orig_shape = (256, 256)
        cls = [3, 4, 5, 5]
        conf = [0.9577, 0.8330, 0.5769, 0.5045]
        xyxy = [
            (65.7083, 60.0278, 206.4184, 197.7821),
            (55.2819, 58.1271, 183.5392, 141.5031),
            (65.6615, 181.2122, 129.9661, 197.8898),
            (66.5323, 175.2623, 125.4538, 197.6352),
        ]

    class Boxes2:
        orig_shape = (256, 256)
        cls = [0, 1, 2]
        conf = [0.9257, 0.8226, 0.5643]
        xyxy = [
            (51.1308, 58.4212, 217.9471, 196.8399),
            (46.7113, 58.0564, 212.0678, 137.7467),
            (50.8259, 172.6048, 215.9904, 198.0775),
        ]

    meta1 = {
        "letterbox_padded": ((0, 0), (58, 58), (0, 0)),
        "letterbox_orig_size": (263, 482),
        "letterbox_resized_size": (140, 256),
        "projection_key": "image1",
    }
    meta2 = {
        "letterbox_padded": ((0, 0), (58, 58), (0, 0)),
        "letterbox_orig_size": (263, 482),
        "letterbox_resized_size": (140, 256),
        "projection_key": "image2",
    }

    slices = compute_oriented_crop_slices_from_boxes_xyxy(
        Boxes1(),
        Boxes2(),
        classes=[0, 1, 2, 3, 4, 5],
        proj1_dict=meta1,
        proj2_dict=meta2,
    )

    assert slice_tuple(slices) == ((87, 411), (104, 389), (0, 263))
