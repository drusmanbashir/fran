# %%
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropd
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.utility.dictionary import EnsureChannelFirstd
import torch
from torch.utils.data.dataloader import (
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
)

from fran.transforms.imageio import LoadSITKd

_loaders = (_MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter)
import ipdb

tr = ipdb.set_trace


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
    lists = [] * listlen
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
    #items is a list of dictionaries each dictionary has keys: "image", "lm
    # Helper function to process items and append to lists
    imgs = []
    labels = []
    fns_imgs = []
    fns_labels = []
    
    for item in items:
        imgs.append(item["image"])
        fns_imgs.append(item["image"].meta['filename_or_obj'])
        labels.append(item["lm"])
        fns_labels.append(item["lm"].meta['filename_or_obj'])
        
    return imgs, labels, fns_imgs, fns_labels

def process_grid_items(item):
    #items is a list of dictionaries each dictionary has keys: "image", "lm
    # Helper function to process items and append to lists
    imgs = []
    labels = []
    fns_imgs = []
    fns_labels = []

    imgs.append(item["image"])
    fns_imgs.append(item["image"].meta['filename_or_obj'])
    labels.append(item["lm"])
    fns_labels.append(item["lm"].meta['filename_or_obj'])
        
    return imgs, labels, fns_imgs, fns_labels

def grid_collated(batch):
    # same as source except each item in a batch is a 2-tuple. The second item in the tuple has locations of the grid which I will ignore in training
    imgs = []
    labels = []
    fns_imgs=[]
    fns_labels = []
    for i, item in enumerate(batch):
        item=item[0]
        imgs_,labels_,fns_imgs_,fns_labels_ = process_grid_items(item)
        imgs.extend(imgs_)
        labels.extend(labels_)
        fns_imgs.extend(fns_imgs_)
        fns_labels.extend(fns_labels_)
    if len(batch) == 1:
        fns_imgs = fns_imgs[0]
        fns_labels = fns_labels[0]
    imgs_out = torch.stack(imgs, 0)
    lms_out= torch.stack(labels, 0)

    imgs_out.meta['filename_or_obj']=fns_imgs
    lms_out.meta['filename_or_obj']=fns_labels
    output = {"image": imgs_out , "lm": lms_out}
    return output

def source_collated(batch):

    imgs = []
    labels = []
    fns_imgs=[]
    fns_labels = []
    for i, item in enumerate(batch):
        imgs_,labels_,fns_imgs_,fns_labels_ = process_items(item)
        imgs.extend(imgs_)
        labels.extend(labels_)
        fns_imgs.extend(fns_imgs_)
        fns_labels.extend(fns_labels_)
    if len(batch) == 1:
        fns_imgs = fns_imgs[0]
        fns_labels = fns_labels[0]
    imgs_out = torch.stack(imgs, 0)
    lms_out= torch.stack(labels, 0)

    imgs_out.meta['filename_or_obj']=fns_imgs
    lms_out.meta['filename_or_obj']=fns_labels
    output = {"image": imgs_out , "lm": lms_out}
    return output


def whole_collated(batch):
    imgs,labels,fns_imgs,fns_labels = process_items(batch)

    if len(batch) == 1:
        fns_imgs = fns_imgs[0]
        fns_labels = fns_labels[0]
    imgs_out = torch.stack(imgs, 0)
    lms_out= torch.stack(labels, 0)
    imgs_out.meta['filename_or_obj']=fns_imgs
    lms_out.meta['filename_or_obj']=fns_labels
    output = {"image": imgs_out , "lm": lms_out}
    return output


# %%
if __name__ == "__main__":

# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------

    patch_size = [128,128,96]
    d1 = {
        "image": "/s/xnat_shadow/crc/images/crc_CRC002_20190415_CAP1p5.nii.gz",
        "lm": "/s/xnat_shadow/crc/lms/crc_CRC002_20190415_CAP1p5.nrrd",
    }
    d2 = {
        "image": "/s/xnat_shadow/crc/images/crc_CRC004_20190425_CAP1p5.nii.gz",
        "lm": "/s/xnat_shadow/crc/lms/crc_CRC004_20190425_CAP1p5.nii.gz",
    }
    keys = ["image", "lm"]
    L = LoadSITKd(keys=["image", "lm"])
    E = EnsureChannelFirstd(keys=keys)
    Res = Resized(keys=keys, spatial_size = patch_size)

    Re = ResizeWithPadOrCropd(
            keys=["image", "lm"],
            spatial_size=patch_size,
            lazy=False,
        )
# %%
# %%
#SECTION:--------------------  Whole_collated--------------------------------------------------------------------------------------


    d1 =E( L(d1))
    d2 = E(L(d2))
    d3 = Res(d2)
    d4 = Res(d1)

# %%
    batch = [d3,d4]

    b2 = whole_collated(batch)
# %%
# %%
#SECTION:-------------------- 2--------------------------------------------------------------------------------------


    d1 =E( L(d1))
    d2 = E(L(d2))
    d3 = Re(d2)
    d4 = Re(d1)

# %%
    batch = [d3,d4]

    bx = whole_collated(batch)



