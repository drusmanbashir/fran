# %% import math

import math

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn.functional import interpolate

CLASSES = ["img"]

CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
CLASS2NUM = {class_: idx for idx, class_ in enumerate(CLASSES)}


def _get_window_ops():
    from .yolo_ct_augment import WINDOW_PRESETS, apply_window_tensor

    return WINDOW_PRESETS, apply_window_tensor


def filter_boxes(output_tensor, threshold):
    b, a, h, w, c = output_tensor.shape
    x = output_tensor.contiguous().view(b, a * h * w, c)

    boxes = x[:, :, 0:4]
    confidence = x[:, :, 4]
    scores, idx = torch.max(x[:, :, 5:], -1)
    idx = idx.float()
    scores = scores * confidence
    mask = scores > threshold

    filtered = []
    for c, s, i, m in zip(boxes, scores, idx, mask):
        if m.any():
            detected = torch.cat([c[m, :], s[m, None], i[m, None]], -1)
        else:
            detected = torch.zeros((0, 6), dtype=x.dtype, device=x.device)
        filtered.append(detected)
    return filtered


def load_image(idx, size, device="cpu"):
    filename = (
        f"/s/datasets_bkp/VOCdevkit/VOC2012/JPEGImages/2008_{str(idx).zfill(6)}.jpg"
    )
    img = Image.open(filename)
    transforms = [torchvision.transforms.ToTensor()]
    try:
        width, height = size
    except TypeError:
        width = height = size
    scale = min(width / img.width, height / img.height)
    new_width, new_height = int(img.width * scale), int(img.height * scale)
    diff_width, diff_height = width - new_width, height - new_height
    resize = torchvision.transforms.Resize(size=(new_height, new_width))
    pad = torchvision.transforms.Pad(
        padding=(
            diff_width // 2,
            diff_height // 2,
            diff_width // 2 + diff_width % 2,
            diff_height // 2 + diff_height % 2,
        )
    )
    transforms = [resize, pad] + transforms
    transformation = torchvision.transforms.Compose(transforms)
    x = transformation(img).to(device)
    return x


def load_image_batch(idxs, size, device="cpu"):
    imgs = [load_image(idx, size=size, device="cpu") for idx in idxs]
    x = torch.stack(imgs, 0)
    return x.to(device)


def load_bboxes(idx, size, num_bboxes=None, device="cpu"):
    boxfilename = (
        f"/s/datasets_bkp/VOCdevkit/VOC2012/Annotations/2008_{str(idx).zfill(6)}.xml"
    )
    imgfilename = (
        f"/s/datasets_bkp/VOCdevkit/VOC2012/JPEGImages/2008_{str(idx).zfill(6)}.jpg"
    )
    img = Image.open(
        imgfilename
    )  # img won't be loaded into memory, just needed for metadata.
    try:
        width, height = size
    except TypeError:
        width = height = size
    scale = min(width / img.width, height / img.height)
    new_width, new_height = int(img.width * scale), int(img.height * scale)
    diff_width = (width - new_width) / width * img.width
    diff_height = (height - new_height) / height * img.height
    bboxes = []
    with open(boxfilename, "r") as file:
        for line in file:
            if "<name>" in line:
                class_ = line.split("<name>")[-1].split("</name>")[0].strip()
            elif "<xmin>" in line:
                x0 = float(line.split("<xmin>")[-1].split("</xmin>")[0].strip())
            elif "<xmax>" in line:
                x1 = float(line.split("<xmax>")[-1].split("</xmax>")[0].strip())
            elif "<ymin>" in line:
                y0 = float(line.split("<ymin>")[-1].split("</ymin>")[0].strip())
            elif "<ymax>" in line:
                y1 = float(line.split("<ymax>")[-1].split("</ymax>")[0].strip())
            elif "</object>" in line:
                if class_ not in CLASS2NUM:
                    continue
                bbox = [
                    (diff_width / 2 + (x0 + x1) / 2)
                    / (img.width + diff_width),  # center x
                    (diff_height / 2 + (y0 + y1) / 2)
                    / (img.height + diff_height),  # center y
                    (max(x0, x1) - min(x0, x1)) / (img.width + diff_width),  # width
                    (max(y0, y1) - min(y0, y1)) / (img.height + diff_height),  # height
                    1.0,  # confidence
                    CLASS2NUM[class_],  # class idx
                ]
                bboxes.append(
                    (
                        bbox[2]
                        * bbox[3]
                        * (img.width + diff_width)
                        * (img.height + diff_height),
                        bbox,
                    )
                )

    bboxes = torch.tensor(
        [bbox for _, bbox in sorted(bboxes)],
        dtype=torch.get_default_dtype(),
        device=device,
    )
    if num_bboxes:
        if num_bboxes > len(bboxes):
            zeros = torch.zeros(
                (num_bboxes - bboxes.shape[0], 6),
                dtype=torch.get_default_dtype(),
                device=device,
            )
            zeros[:, -1] = -1
            bboxes = torch.cat([bboxes, zeros], 0)
        elif num_bboxes < len(bboxes):
            bboxes = bboxes[:num_bboxes]

    return bboxes


def display_torch(img):
    img = img.permute(1, 2, 0)
    img = img.detach().cpu().numpy()
    plt.imshow(img)


def load_bboxes_batch(idxs, size, num_bboxes, device="cpu"):
    bboxes = [
        load_bboxes(idx, size, num_bboxes=num_bboxes, device="cpu") for idx in idxs
    ]
    bboxes = torch.stack(bboxes, 0)
    return bboxes


def make_subplots(num_imgs):
    n_row = int(math.sqrt(num_imgs))
    n_col = int(np.ceil(num_imgs / n_row))
    fig, axs = plt.subplots(n_row, n_col)
    return fig, axs


def show_images(x):
    if x.dim() == 4 and not x.shape[0] == 1:
        num_imgs = x.shape[0]
        fig, axs = make_subplots(num_imgs)
        n_row = axs.shape[0]
        for i, xx in enumerate(x):
            ax_ind = divmod(i, n_row)
            xx = xx.permute(1, 2, 0)
            axs[*ax_ind].imshow(xx)

    elif x.dim() == 4 and x.shape[0] == 1:
        xx = x[0]
        xx = xx.permute(1, 2, 0)
        plt.imshow(xx)

    elif x.dim() == 3:
        x = x.permute(1, 2, 0)
        plt.imshow(x)
    else:
        raise ValueError


def show_images_with_boxes(input_tensor, output_tensor):
    n_imgs = input_tensor.shape[0]
    fig, axs = make_subplots(n_imgs)
    try:
        n_row = axs.shape[0]
    except:
        n_row = 1
    for ind in range(n_imgs):
        try:
            ax_ind = divmod(ind, n_row)
            ax = axs[*ax_ind]
        except:
            ax = axs
        img = input_tensor[ind]
        predictions = output_tensor[ind]
        img = img.permute(1, 2, 0)
        if 0 in predictions.shape:  # empty tensor
            plt.imshow(img)
            continue
        confidences = predictions[..., 4].flatten()
        boxes = (
            predictions[..., :4].contiguous().view(-1, 4)
        )  # only take first four features: x0, y0, w, h
        classes = predictions[..., 5:].contiguous().view(boxes.shape[0], -1)
        boxes[:, ::2] *= img.shape[0]
        boxes[:, 1::2] *= img.shape[1]
        boxes = (
            torch.stack(
                [
                    boxes[:, 0] - boxes[:, 2] / 2,
                    boxes[:, 1] - boxes[:, 3] / 2,
                    boxes[:, 0] + boxes[:, 2] / 2,
                    boxes[:, 1] + boxes[:, 3] / 2,
                ],
                -1,
            )
            .cpu()
            .to(torch.int32)
            .numpy()
        )
        for box, confidence, class_ in zip(boxes, confidences, classes):
            if confidence < 0.01:
                continue  # don't show boxes with very low confidence
            # make sure the box fits within the picture:
            box = [
                max(0, int(box[0])),
                max(0, int(box[1])),
                min(img.shape[0] - 1, int(box[2])),
                min(img.shape[1] - 1, int(box[3])),
            ]
            try:  # either the class is given as the sixth feature
                idx = int(class_.item())
            except:  # or the 20 softmax probabilities are given as features 6-25
                idx = int(torch.max(class_, 0)[1].item())
            try:
                class_ = CLASSES[idx]  # the first index of torch.max is the argmax.
            except (
                IndexError
            ):  # if the class index does not exist, don't draw anything:
                continue

            color = (  # green color when confident, red color when not confident.
                int((1 - (confidence.item()) ** 0.8) * 255),
                int((confidence.item()) ** 0.8 * 255),
                0,
            )
            draw_image_bbox(img, *box, class_, box[:2], ax)


def show_image_with_boxes(input_tensor, output_tensor):
    fig, ax = plt.subplots()
    for img, predictions in zip(input_tensor, output_tensor):
        img = img.permute(1, 2, 0)
        if 0 in predictions.shape:  # empty tensor
            plt.imshow(img)
            continue
        confidences = predictions[..., 4].flatten()
        boxes = (
            predictions[..., :4].contiguous().view(-1, 4)
        )  # only take first four features: x0, y0, w, h
        classes = predictions[..., 5:].contiguous().view(boxes.shape[0], -1)
        boxes[:, ::2] *= img.shape[0]
        boxes[:, 1::2] *= img.shape[1]
        boxes = (
            torch.stack(
                [
                    boxes[:, 0] - boxes[:, 2] / 2,
                    boxes[:, 1] - boxes[:, 3] / 2,
                    boxes[:, 0] + boxes[:, 2] / 2,
                    boxes[:, 1] + boxes[:, 3] / 2,
                ],
                -1,
            )
            .cpu()
            .to(torch.int32)
            .numpy()
        )
        for box, confidence, class_ in zip(boxes, confidences, classes):
            if confidence < 0.01:
                continue  # don't show boxes with very low confidence
            # make sure the box fits within the picture:
            box = [
                max(0, int(box[0])),
                max(0, int(box[1])),
                min(img.shape[0] - 1, int(box[2])),
                min(img.shape[1] - 1, int(box[3])),
            ]
            try:  # either the class is given as the sixth feature
                idx = int(class_.item())
            except:  # or the 20 softmax probabilities are given as features 6-25
                idx = int(torch.max(class_, 0)[1].item())
            try:
                class_ = CLASSES[idx]  # the first index of torch.max is the argmax.
            except (
                IndexError
            ):  # if the class index does not exist, don't draw anything:
                continue

            color = (  # green color when confident, red color when not confident.
                int((1 - (confidence.item()) ** 0.8) * 255),
                int((confidence.item()) ** 0.8 * 255),
                0,
            )

            draw_image_bbox(img, *box, class_, box[:2], ax)


def iou(bboxes1, bboxes2):
    """calculate iou between each bbox in `bboxes1` with each bbox in `bboxes2`"""
    px, py, pw, ph = bboxes1[..., :4].reshape(-1, 4).split(1, -1)
    lx, ly, lw, lh = bboxes2[..., :4].reshape(-1, 4).split(1, -1)
    px1, py1, px2, py2 = px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph
    lx1, ly1, lx2, ly2 = lx - 0.5 * lw, ly - 0.5 * lh, lx + 0.5 * lw, ly + 0.5 * lh
    zero = torch.tensor(0.0, dtype=px1.dtype, device=px1.device)
    dx = torch.max(torch.min(px2, lx2.T) - torch.max(px1, lx1.T), zero)
    dy = torch.max(torch.min(py2, ly2.T) - torch.max(py1, ly1.T), zero)
    intersections = dx * dy
    pa = (px2 - px1) * (py2 - py1)  # area
    la = (lx2 - lx1) * (ly2 - ly1)  # area
    unions = (pa + la.T) - intersections
    ious = (intersections / unions).reshape(*bboxes1.shape[:-1], *bboxes2.shape[:-1])

    return ious


def iou_wh(bboxes1, bboxes2):
    """calculate iou between each bbox in `bboxes1` with each bbox in `bboxes2`

    The bboxes should be defined by their width and height and are centered around (0,0)

    """

    w1 = bboxes1[..., 0].view(-1)
    h1 = bboxes1[..., 1].view(-1)
    w2 = bboxes2[..., 0].view(-1)
    h2 = bboxes2[..., 1].view(-1)

    intersections = torch.min(w1[:, None], w2[None, :]) * torch.min(
        h1[:, None], h2[None, :]
    )
    unions = (w1 * h1)[:, None] + (w2 * h2)[None, :] - intersections
    ious = (intersections / unions).reshape(*bboxes1.shape[:-1], *bboxes2.shape[:-1])

    return ious


def nms(filtered_tensor, threshold):
    result = []
    for x in filtered_tensor:
        # Sort coordinates by descending confidence
        scores, order = x[:, 4].sort(0, descending=True)
        x = x[order]
        ious = iou(x, x)  # get ious between each bbox in x

        # Filter based on iou
        keep = (ious > threshold).long().triu(1).sum(0, keepdim=True).t().expand_as(
            x
        ) == 0

        result.append(x[keep].view(-1, 6).contiguous())
    return result


def draw_image_bbox(
    img, start_x, start_y, stop_x, stop_y, text=None, text_xy=[], ax=None
):
    # img= torch.load(fn_img)
    size_x = stop_x - start_x
    size_y = stop_y - start_y
    if not ax:
        fig, ax = plt.subplots()
    ax.imshow(img)

    rect = patches.Rectangle(
        (start_x, start_y), size_x, size_y, linewidth=1, edgecolor="r", facecolor="none"
    )
    ax.add_patch(rect)
    if text:
        ax.text(text_xy[0], text_xy[1], text)


def draw_image_lm_bbox(img, lm, start_x, start_y, stop_x, stop_y):
    # img= torch.load(fn_img)
    size_x = stop_x - start_x
    size_y = stop_y - start_y
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    rect = patches.Rectangle(
        (start_x, start_y), size_x, size_y, linewidth=1, edgecolor="r", facecolor="none"
    )
    ax1.add_patch(rect)

    ax2.imshow(lm)
    rect = patches.Rectangle(
        (start_x, start_y), size_x, size_y, linewidth=1, edgecolor="r", facecolor="none"
    )
    ax2.add_patch(rect)


def load_weights(
    network, filename="/s/fran_storage/checkpoints/detection/yolov2-tiny-voc.weights"
):
    with open(filename, "rb") as file:
        version = np.fromfile(file, count=3, dtype=np.int32)
        seen_so_far = np.fromfile(file, count=1, dtype=np.int32)
        weights = np.fromfile(file, dtype=np.float32)
        idx = 0
        print("loading weights")
        for layer in network.children():
            if isinstance(layer, torch.nn.Conv2d):
                if layer.bias is not None:
                    n = layer.bias.numel()
                    layer.bias.data[:] = torch.from_numpy(
                        weights[idx : idx + n]
                    ).view_as(layer.bias.data)
                    idx += n
                n = layer.weight.numel()
                layer.weight.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(
                    layer.weight.data
                )
                idx += n
            if isinstance(layer, torch.nn.BatchNorm2d):
                n = layer.bias.numel()
                layer.bias.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(
                    layer.bias.data
                )
                idx += n
                layer.weight.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(
                    layer.weight.data
                )
                idx += n
                layer.running_mean.data[:] = torch.from_numpy(
                    weights[idx : idx + n]
                ).view_as(layer.running_mean)
                idx += n
                layer.running_var.data[:] = torch.from_numpy(
                    weights[idx : idx + n]
                ).view_as(layer.running_var)
                idx += n
            if isinstance(layer, torch.nn.Linear):
                n = layer.bias.numel()
                layer.bias.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(
                    layer.bias.data
                )
                idx += n
                n = layer.weight.numel()
                layer.weight.data[:] = torch.from_numpy(weights[idx : idx + n]).view_as(
                    layer.weight.data
                )
                idx += n

def draw_tensor_boxes(imtnsr, rr, filename=None, show=True):
    image_np = (imtnsr[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    image_np = np.ascontiguousarray(image_np.copy())
    image_h, image_w = image_np.shape[:2]
    boxes = rr.boxes.xyxy.cpu().numpy().astype(int)
    confs = rr.boxes.conf.cpu().numpy()
    clss = rr.boxes.cls.cpu().numpy().astype(int)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    pad = 2
    bg_alpha = 0.4
    palette = [
        (0, 255, 0),
        (255, 140, 0),
        (0, 200, 255),
        (255, 80, 160),
        (180, 80, 255),
        (255, 220, 0),
    ]

    def overlaps(rect_a, rect_b):
        ax1, ay1, ax2, ay2 = rect_a
        bx1, by1, bx2, by2 = rect_b
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

    placed_label_rects = []
    detections = sorted(
        zip(boxes, confs, clss, range(len(boxes))),
        key=lambda det: (det[0][1], det[0][0]),
    )

    for box, conf, cls_id, det_idx in detections:
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, image_w - 1))
        x2 = max(0, min(x2, image_w - 1))
        y1 = max(0, min(y1, image_h - 1))
        y2 = max(0, min(y2, image_h - 1))
        label = f"{rr.names[cls_id]} {conf:.2f}"
        color = palette[det_idx % len(palette)]
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        label_w = text_w + 2 * pad
        label_h = text_h + baseline + 2 * pad
        label_x2 = x1 - 2
        label_x1 = label_x2 - label_w
        if label_x1 < 0:
            label_x1 = 0
            label_x2 = label_w
        base_y1 = max(0, min(y1, image_h - label_h))
        label_y1 = base_y1
        step = label_h + 2
        max_down_steps = max(0, (image_h - label_h - base_y1) // step)
        max_up_steps = max(0, base_y1 // step)
        candidate_offsets = [0]
        for step_idx in range(1, max(max_down_steps, max_up_steps) + 1):
            if step_idx <= max_down_steps:
                candidate_offsets.append(step_idx * step)
            if step_idx <= max_up_steps:
                candidate_offsets.append(-step_idx * step)

        for offset in candidate_offsets:
            candidate_y1 = base_y1 + offset
            candidate_rect = (
                label_x1,
                candidate_y1,
                label_x2,
                candidate_y1 + label_h,
            )
            if not any(
                overlaps(candidate_rect, placed_rect)
                for placed_rect in placed_label_rects
            ):
                label_y1 = candidate_y1
                break

        label_y2 = min(image_h - 1, label_y1 + label_h)
        text_org = (label_x1 + pad, label_y2 - baseline - pad)
        placed_label_rects.append((label_x1, label_y1, label_x2, label_y2))

        overlay = image_np.copy()
        cv2.rectangle(
            overlay,
            (label_x1, label_y1),
            (label_x2, label_y2),
            color,
            -1,
        )
        cv2.addWeighted(overlay, bg_alpha, image_np, 1 - bg_alpha, 0, image_np)
        cv2.putText(
            image_np,
            label,
            text_org,
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    if filename is not None:
        Image.fromarray(image_np).save(filename)
    if show:
        plt.figure(figsize=(8, 8))
        plt.imshow(image_np)
        plt.axis("off")
        plt.title("YOLO prediction")
        plt.show()
    return image_np


def validate_yolo_input(x):
    # Direct tensor inputs to YOLO are expected to already be float32 in [0, 1].
    x = x.float().contiguous()
    x_min = x.min().item()
    x_max = x.max().item()
    assert x_min >= 0.0 and x_max <= 1.0, (
        f"YOLO tensor input must be normalized to [0, 1], got [{x_min:.4f}, {x_max:.4f}]"
    )
    return x

def make_multiwindow_inference_tensor(image, imsize=256):
    WINDOW_PRESETS, apply_window_tensor = _get_window_ops()
    chs = []
    for window in WINDOW_PRESETS:
        chs.append(apply_window_tensor(image, window))
    x = torch.cat(chs, dim=0)
    x = torch.mean(x, dim=1)
    x = x.unsqueeze(0)
    x = interpolate(x, (imsize, imsize))
    return validate_yolo_input(x)

def make_singlewindow_inference_tensor(image, window, imsize=256):
    _, apply_window_tensor = _get_window_ops()
    ch = apply_window_tensor(image, window)
    x = ch.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    x = interpolate(x, (imsize, imsize))
    return validate_yolo_input(x)

def jpg_to_tensor(fn):
    img = Image.open(fn).convert("RGB")
    x = torch.from_numpy(np.array(img))
    return x
