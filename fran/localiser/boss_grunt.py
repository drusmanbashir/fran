import math

def invert_resize_bbox(bbox, src_shape, target_size=256, letterbox=True):
    cx, cy, w, h = bbox
    src_h, src_w = src_shape
    cx_t = cx * target_size
    cy_t = cy * target_size
    w_t = w * target_size
    h_t = h * target_size

    if letterbox:
        scale = min(target_size / src_h, target_size / src_w)
        new_h = src_h * scale
        new_w = src_w * scale
        pad_y = (target_size - new_h) / 2
        pad_x = (target_size - new_w) / 2
        return (
            (cx_t - pad_x) / scale,
            (cy_t - pad_y) / scale,
            w_t / scale,
            h_t / scale,
        )

    return (
        cx_t * src_w / target_size,
        cy_t * src_h / target_size,
        w_t * src_w / target_size,
        h_t * src_h / target_size,
    )


def bbox_center_size_to_bounds(bbox):
    cx, cy, w, h = bbox
    return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)


def clamp_bounds_to_slice(start, stop, dim):
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
    z_dim, y_dim, x_dim = shape_3d
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

    z_height = max(zc_stop - zc_start, zs_stop - zs_start)
    z_center = (zc_start + zc_stop + zs_start + zs_stop) / 4
    z_start = z_center - z_height / 2
    z_stop = z_center + z_height / 2

    return (
        clamp_bounds_to_slice(z_start, z_stop, z_dim),
        clamp_bounds_to_slice(y_start, y_stop, y_dim),
        clamp_bounds_to_slice(x_start, x_stop, x_dim),
    )


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
