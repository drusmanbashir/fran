import numpy as np
from numpy.core.fromnumeric import resize
import torch
import SimpleITK as sitk
from fran.utils.fileio import save_np
from fran.utils.helpers import abs_list, get_case_id_from_filename
import torch.nn.functional as F
import ipdb

tr = ipdb.set_trace





# %%
def convert_float16_to_float32(img_fn):  # for torchfloat32
    img = np.load(img_fn)
    img.dtype
    ret_val = [img_fn, img.dtype]
    if img.dtype != np.float32:
        img = img.astype(np.float32)
        save_np(img, img_fn)
    return ret_val


def convert_np_to_tensor(img_fn):  # for torchfloat32
    img = np.load(img_fn)
    ret_val = [img_fn, img.dtype]
    img = torch.tensor(img)
    ret_val.append(img.dtype)
    torch.save(img, str(img_fn).replace("npy", "pt"))
    return ret_val


def resize_tensor_3d(x: torch.Tensor, output_size, mode=None):
    def _inner(x, output_size, mode):
        x = x.unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, output_size, mode=mode)
        x = x.squeeze(0).squeeze(0)
        return x

    if mode == None:
        mode = "nearest" if "int" in str(x.dtype) else "trilinear"
    x = _inner(x, output_size, mode)
    return x


# %%
def resize_multilabel_mask_torch(mask_np, sz_dest_np, label_priority=None):
    if mask_np.dtype == np.uint16:
        mask_np = mask_np.astype(np.uint8)
    mask_torch = torch.tensor(mask_np, dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
    mask_out = F.interpolate(mask_torch, size=sz_dest_np, mode="nearest-exact")
    mask_out = mask_out.squeeze(0).squeeze(0)
    return mask_out.numpy()


def resize_multilabel_mask_sitk(mask_np, sz_dest_np, label_priority):
    """
    skimage resize expects a boolean mask. This function creates bool masks for each label and then overlays them in label priority given. The last label overlays all the rest
    """

    masks_out = []
    mask_template = np.zeros(sz_dest_np, dtype=np.uint8)
    for label in label_priority:
        mask_tmp = np.zeros(mask_np.shape, dtype=bool)
        # if label == 1: mask_tmp[mask_np>0]=True
        # else: mask_tmp[mask_np== label]=True

        mask_tmp[mask_np == label] = True
        mask_tmp = resize(mask_tmp, sz_dest_np, order=0)
        masks_out.append(mask_tmp)

    for mask, label in zip(masks_out, label_priority):
        mask_template[mask == True] = label
    return mask_template


# %%
def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    sizes = np.array([maxzidx - minzidx, maxxidx - minxidx, maxyidx - minyidx])
    return (
        slice(minzidx, maxzidx),
        slice(minxidx, maxxidx),
        slice(minyidx, maxyidx),
    ), sizes


def crop_to_bbox(arr, bbox, crop_axes, crop_padding=0.0, stride=[1, 1, 1]):
    """
    param arr: torch tensor or np array to be cropped
    param bbox: Bounding box (3D only supported)
    param crop_axes:  any combination of 'xyz' may be used (e.g., 'xz' will crop in x and z axes)
    param crop_padding: add crop_padding [0,1] fraction to all the planes of cropping.
    param stride: stride in each plane
    """
    assert len(arr.shape) == 3, "only supports 3d images"
    bbox_extra_pct = [
        int((bbox[i][1] - bbox[i][0]) * crop_padding / 2) for i in range(len(bbox))
    ]
    bbox_mod = [
        [
            np.maximum(0, bbox[j][0] - bbox_extra_pct[j]),
            np.minimum(bbox[j][1] + bbox_extra_pct[j], arr.shape[j]),
        ]
        for j in range(arr.ndim)
    ]
    slices = []
    for dim, axis in zip(
        [0, 1, 2], ["z", "y", "x"]
    ):  # tensors are opposite arrranged to numpy
        if axis in crop_axes:
            slices.append(slice(bbox_mod[dim][0], bbox_mod[dim][1], stride[dim]))
        else:
            slices.append(slice(0, arr.shape[dim], stride[dim]))
    return arr[tuple(slices)]


def is_standard_orientation(direction: tuple):
    standard =tuple(np.eye(3).flatten())
    direction = tuple(0.0 if aa == -0.0 else float(aa) for aa in direction)
    return standard == direction


def get_img_mask_from_nii(case_files_tuple, outside_value=0):

    properties = dict()
    data_itk = [sitk.ReadImage(f) for f in case_files_tuple]
    direction = abs_list(data_itk[0].GetDirection())
    if not is_standard_orientation(direction):
        print(
            "Warning. Casefiles {0} are not in standard orientation.\n Orientation:{1} ...\
            \nBoth the image and mask raw data are being transposed to standard DICOM orientation {2}, and overwritten".format(
                case_files_tuple[0], direction, tuple(np.eye(3).flatten())
            )
        )
        data_itk = list(
            map(lambda x: sitk.DICOMOrient(x, "LPS"), data_itk)
        )  # fixes orientation of image - CRUCIAL STEP
        for img, fn in zip(data_itk, case_files_tuple):
            sitk.WriteImage(img, fn)
    img, mask = [
        sitk.GetArrayFromImage(d)[None].astype(np.float32) for d in data_itk
    ]  # returns channel x width x height x depth
    properties["img_file"] = case_files_tuple[0]
    properties["mask_file"] = case_files_tuple[1]
    properties["itk_size"] = data_itk[0].GetSize()
    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()
    # properties["bbox"] = get_bbox_from_mask(mask, outside_value=outside_value)
    return img, mask, properties


#
# def get_img_mask_from_nii(case):
#         assert isinstance(case, list) or isinstance(case, tuple), "case must be either a list or a tuple"
#         properties = dict()
#         properties["case_id"] = get_case_id_from_filename(case[0])
#         properties["img_file"] = case[0]
#         properties["mask_file"] = case[1]
#         data_itk = [sitk.ReadImage(f) for f in case]
#         properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
#         properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
#
#         properties["itk_origin"] = data_itk[0].GetOrigin()
#         properties["itk_spacing"] = data_itk[0].GetSpacing()
#         properties["itk_direction"] = data_itk[0].GetDirection()
#         img,mask= [sitk.GetArrayFromImage(d)[None].astype(np.float32) for d in data_itk] # returns channel x width x height x depth

#
def retrieve_properties_from_nii(case):
    properties = dict()
    properties["case_id"] = get_case_id_from_filename(case[0])
    properties["img_file"] = case[0]
    properties["mask_file"] = case[1]
    data_itk = [sitk.ReadImage(f) for f in case]
    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()
    return properties

if __name__ == "__main__":
    tupl = "/s/fran_storage/datasets/raw_data/lits/masks/lits-51.nii", "/s/fran_storage/datasets/raw_data/lits/masks/lits-51.nii"

