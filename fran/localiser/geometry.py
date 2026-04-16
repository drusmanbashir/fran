# 3D shape is (Z, Y, X); coronal projection is (Z, X), sagittal is (Z, Y).
# Crop is in rescaled 3D space before projection resize; YOLO boxes are normalized.
import math


def invert_resize_bbox(
    bbox,
    src_shape,
    target_size=256,
    letterbox=True,
):
    """
    Return (cx, cy, w, h) in source projection coordinates.
    """
    cx, cy, w, h = bbox
    H, W = src_shape
    cx_t = cx * target_size
    cy_t = cy * target_size
    w_t = w * target_size
    h_t = h * target_size

    if letterbox:
        scale = min(target_size / H, target_size / W)
        new_H = H * scale
        new_W = W * scale
        pad_y = (target_size - new_H) / 2
        pad_x = (target_size - new_W) / 2
        return (
            (cx_t - pad_x) / scale,
            (cy_t - pad_y) / scale,
            w_t / scale,
            h_t / scale,
        )

    scale_x = W / target_size
    scale_y = H / target_size
    return (
        cx_t * scale_x,
        cy_t * scale_y,
        w_t * scale_x,
        h_t * scale_y,
    )


def bbox_center_size_to_bounds(bbox):
    """
    Input: (cx, cy, w, h)
    Return: (x_start, x_stop, y_start, y_stop)
    """
    cx, cy, w, h = bbox
    return (
        cx - w / 2,
        cx + w / 2,
        cy - h / 2,
        cy + h / 2,
    )


def clamp_bounds_to_slice(start, stop, dim):
    """
    floor start, ceil stop, clamp to [0, dim], return slice
    """
    return slice(
        min(max(math.floor(start), 0), dim),
        min(max(math.ceil(stop), 0), dim),
    )


def compute_crop_slices_from_yolo(
    shape_3d,
    bbox_cor,
    bbox_sag,
    proj_shape_cor,
    proj_shape_sag,
    target_size=256,
    letterbox=True,
):
    """
    Return (slice_z, slice_y, slice_x)
    """
    Z, Y, X = shape_3d

    bbox_cor_src = invert_resize_bbox(
        bbox_cor,
        proj_shape_cor,
        target_size=target_size,
        letterbox=letterbox,
    )
    bbox_sag_src = invert_resize_bbox(
        bbox_sag,
        proj_shape_sag,
        target_size=target_size,
        letterbox=letterbox,
    )

    x_start, x_stop, zc_start, zc_stop = bbox_center_size_to_bounds(bbox_cor_src)
    y_start, y_stop, zs_start, zs_stop = bbox_center_size_to_bounds(bbox_sag_src)

    z_start = min(zc_start, zs_start)
    z_stop = max(zc_stop, zs_stop)

    slice_z = clamp_bounds_to_slice(z_start, z_stop, Z)
    slice_y = clamp_bounds_to_slice(y_start, y_stop, Y)
    slice_x = clamp_bounds_to_slice(x_start, x_stop, X)
    return (slice_z, slice_y, slice_x)


class ComputeCropSlicesFromYolo:
    def __init__(self, target_size=256, letterbox=True):
        self.target_size = target_size
        self.letterbox = letterbox

    def __call__(self, shape_3d, bbox_cor, bbox_sag, proj_shape_cor, proj_shape_sag):
        return compute_crop_slices_from_yolo(
            shape_3d=shape_3d,
            bbox_cor=bbox_cor,
            bbox_sag=bbox_sag,
            proj_shape_cor=proj_shape_cor,
            proj_shape_sag=proj_shape_sag,
            target_size=self.target_size,
            letterbox=self.letterbox,
        )
