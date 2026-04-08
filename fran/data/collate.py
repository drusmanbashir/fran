# %%
from dataclasses import dataclass
import torch
from torch.utils.data.dataloader import (
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
)

_loaders = (_MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter)
import ipdb

tr = ipdb.set_trace

@dataclass
class VarType:
    name: str
    type: type
    op:str|None = None

Dict = VarType(name="dict", type=dict, op= "append")
Meta = Dict
Str = VarType(name="str", type=str)
Image = VarType(name="image", type=torch.tensor)  #float
LM = VarType(name="lm", type=torch.tensor) # uint8
Number = VarType(name="number", type=float|int)
BBox = VarType(name="tuple", type=tuple)

CATALOG  = {
        "image" : Image,
        "pred" :  Image,
        "lm": LM,
        "bbox": BBox,
        "nbrhood": Dict,
        "label": Number,
        "images_meta": Meta,
        "lms_meta": Meta,
        }

def img_lm_bbox_collate(batch):
    imgs = []
    lms = []
    bboxes = []
    for i, item in enumerate(batch):
        imgs.append(item[0])
        lms.append(item[1])
        bboxes.append(item[2])
    return torch.stack(imgs, 0), torch.stack(lms, 0), bboxes


def img_lm_bbox_collated(batch):
    imgs = []
    lms = []
    bboxes = []
    for i, item in enumerate(batch):
        imgs.append(item["image"])
        lms.append(item["lm"])
        bboxes.append(item["bbox"])
    output = {"image": torch.stack(imgs, 0), "lm": torch.stack(lms, 0), "bbox": bboxes}
    return output


def img_lm_metadata_lists_collated(batch):
    images = []
    lms = []
    images_meta = []
    lms_meta = []
    for i, item in enumerate(batch):
        images.append(item["image"])
        images_meta.append(item["image"].meta)
        lms.append(item["lm"])
        lms_meta.append(item["lm"].meta)
    output = {
        "image": images,
        "lm": lms,
        "images_meta": images_meta,
        "lms_meta": lms_meta,
    }
    return output


def as_is_collated(batch):
    keys = batch[0].keys()
    output_dict = {k: [] for k in keys}
    listlen = len(keys)
    [] * listlen
    for i, item in enumerate(batch):
        for k in keys:
            tnsr = item[k]
            output_dict[k].append(tnsr)
    return output_dict


def dict_list_collated(keys):
    def _inner(batch):
        output = {key: [] for key in keys}
        for i, item in enumerate(batch):
            for key in keys:
                output[key].append(item[key])
        return output

    return _inner


def process_items(items):
    # items is a list of dictionaries each dictionary has keys: "image", "lm
    # Helper function to process items and append to lists
    imgs = []
    labels = []
    fns_imgs = []
    fns_labels = []
    for item in items:
        imgs.append(item["image"])
        fns_imgs.append(item["image"].meta["filename_or_obj"])
        labels.append(item["lm"])
        fns_labels.append(item["lm"].meta["filename_or_obj"])
    return imgs, labels, fns_imgs, fns_labels


def process_items_whole(items):
    # items is a list of list of  dictionaries  (as ooposed to above)
    imgs = []
    labels = []
    fns_imgs = []
    fns_labels = []
    for item in items:
        imgs.append(item["image"])
        fns_imgs.append(item["image"].meta["filename_or_obj"])
        labels.append(item["lm"])
        fns_labels.append(item["lm"].meta["filename_or_obj"])
    return imgs, labels, fns_imgs, fns_labels


def process_grid_items(item):
    # items is a list of dictionaries each dictionary has keys: "image", "lm
    # Helper function to process items and append to lists
    imgs = []
    labels = []
    fns_imgs = []
    fns_labels = []

    imgs.append(item["image"])
    fns_imgs.append(item["image"].meta["filename_or_obj"])
    labels.append(item["lm"])
    fns_labels.append(item["lm"].meta["filename_or_obj"])

    return imgs, labels, fns_imgs, fns_labels


def grid_collated(batch):
    # Supports MONAI GridPatchDataset output as either:
    # - tuple: (patch_dict, coords), or
    # - dict: patch_dict (if dataset uses with_coordinates=False).
    imgs = []
    lms = []
    fns_imgs = []
    fns_lms = []
    patch_coords = []
    start_pos = []
    is_padded = []
    for i, item in enumerate(batch):
        if isinstance(item, tuple):
            item2 = item[0]
            coords = item[1]
        else:
            item2 = item
            coords = item2.get("patch_coords")
        patch_coords.append(coords)
        imgs_, lms_, fns_imgs_, fns_lms_ = process_grid_items(item2)
        start_pos.append(item2["start_pos"])
        is_padded.append(bool(item2.get("is_padded", False)))
        imgs.extend(imgs_)
        lms.extend(lms_)
        fns_imgs.extend(fns_imgs_)
        fns_lms.extend(fns_lms_)
    if len(batch) == 1:
        fns_imgs = fns_imgs[0]
        fns_lms = fns_lms[0]
    imgs_out = torch.stack(imgs, 0)
    lms_out = torch.stack(lms, 0)
    imgs_out.meta["filename_or_obj"] = fns_imgs
    lms_out.meta["filename_or_obj"] = fns_lms
    output = {
        "image": imgs_out,
        "lm": lms_out,
        "patch_coords": patch_coords,
        "start_pos": start_pos,
        "is_padded": is_padded,
    }
    return output


def patch_collated(batch):
    imgs = []
    labels = []
    fns_imgs = []
    fns_labels = []
    for i, item in enumerate(batch):
        image = item["image"]
        lm = item["lm"]
        imgs.append(image)
        labels.append(lm)
        fns_imgs.append(image.meta["filename_or_obj"])
        fns_labels.append(lm.meta["filename_or_obj"])
    if len(batch) == 1:
        fns_imgs = fns_imgs[0]
        fns_labels = fns_labels[0]
    imgs_out = torch.stack(imgs, 0)
    lms_out = torch.stack(labels, 0)
    imgs_out.meta["filename_or_obj"] = fns_imgs
    lms_out.meta["filename_or_obj"] = fns_labels
    output = {"image": imgs_out, "lm": lms_out}
    return output


def source_collated(batch):

    imgs = []
    labels = []
    fns_imgs = []
    fns_labels = []
    for i, item in enumerate(batch):
        imgs_, labels_, fns_imgs_, fns_labels_ = process_items(item)
        imgs.extend(imgs_)
        labels.extend(labels_)
        fns_imgs.extend(fns_imgs_)
        fns_labels.extend(fns_labels_)
    if len(batch) == 1:
        fns_imgs = fns_imgs[0]
        fns_labels = fns_labels[0]
    imgs_out = torch.stack(imgs, 0)
    lms_out = torch.stack(labels, 0)

    imgs_out.meta["filename_or_obj"] = fns_imgs
    lms_out.meta["filename_or_obj"] = fns_labels
    output = {"image": imgs_out, "lm": lms_out}
    return output


def whole_collated(batch):
    imgs, labels, fns_imgs, fns_labels = process_items_whole(batch)
    if len(batch) == 1:
        fns_imgs = fns_imgs[0]
        fns_labels = fns_labels[0]
    imgs_out = torch.stack(imgs, 0)
    lms_out = torch.stack(labels, 0)
    imgs_out.meta["filename_or_obj"] = fns_imgs
    lms_out.meta["filename_or_obj"] = fns_labels
    output = {"image": imgs_out, "lm": lms_out}
    return output


# %%
if __name__ == "__main__":
    from fran.data.dataset import NormaliseClipd
    from fran.transforms.imageio import LoadSITKd
    from monai.transforms.croppad.dictionary import (
        RandCropByPosNegLabeld,
        ResizeWithPadOrCropd,
    )
    from monai.transforms.spatial.dictionary import RandAffined, RandFlipd, Resized
    from monai.transforms.utility.dictionary import EnsureChannelFirstd
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR>

    patch_size = [128, 128, 96]
    d1 = {
        "image": "/s/xnat_shadow/nodes/images/nodes_100_410107_CAP1p5SoftTissue.nii.gz",
        "lm": "/s/xnat_shadow/nodes/lms/nodes_100_410107_CAP1p5SoftTissue.nii.gz",
    }
    d2 = {
        "image": "/s/xnat_shadow/nodes/images/nodes_101_Ta91212_CAP1p5SoftTissue.nii.gz",
        "lm": "/s/xnat_shadow/nodes/lms/nodes_101_Ta91212_CAP1p5SoftTissue.nii.gz",
    }
# %%
    keys = ["image", "lm"]
    L = LoadSITKd(keys=["image", "lm"])
    E = EnsureChannelFirstd(keys=keys)
    Res = Resized(keys=keys, spatial_size=patch_size)

    Re = ResizeWithPadOrCropd(
        keys=["image", "lm"],
        spatial_size=patch_size,
        lazy=False,
    )

    # Additional transforms for mode='source' as per training.py
    Rtr = RandCropByPosNegLabeld(
        keys=["image", "lm"],
        label_key="lm",
        image_key="image",
        spatial_size=patch_size,
        pos=1,
        neg=1,
        num_samples=2,
        lazy=True,
        allow_smaller=True,
    )

    F1 = RandFlipd(keys=["image", "lm"], prob=0.5, spatial_axis=0, lazy=True)

    F2 = RandFlipd(keys=["image", "lm"], prob=0.5, spatial_axis=1, lazy=True)

    Affine = RandAffined(
        keys=["image", "lm"],
        mode=["bilinear", "nearest"],
        prob=0.2,
        rotate_range=0.1,
        scale_range=0.1,
    )

    N = NormaliseClipd(
        keys=["image"],
        clip_range=(-1000, 1000),  # Adjust based on your data
        mean=0,
        std=1,
    )

# %%
# SECTION:--------------------  Whole_collated-------------------------------------------------------------------------------------- <CR> <CR>

    d1 = E(L(d1))
    d2 = E(L(d2))
    d3 = Res(d2)
    d4 = Res(d1)

# %%
    batch = [[d3], [d4]]

    b2 = source_collated(batch)
    # b2 = whole_collated(batch)
# %%
# %%
# SECTION:-------------------- SOURCE COLLATED-------------------------------------------------------------------------------------- <CR> <CR>

    d1 = ResizePC(E(L(d1)))
    d2 = E(L(d2))
    d3 = Res(d2)
# %%
# SECTION:-------------------- 2-------------------------------------------------------------------------------------- <CR> <CR>

    d1 = E(L(d1))
    d2 = E(L(d2))
    d3 = Re(d2)
    d4 = Re(d1)

# %%
    batch = [d3, d4]

    bx = whole_collated(batch)
