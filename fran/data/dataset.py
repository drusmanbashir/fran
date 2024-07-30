# %%
# nvidia measure command
# timeout -k 1 2700 nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > results-file.csv
# short version below
# timeout -k 1 2700 nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > results-file.csv

import itertools
import operator
from collections.abc import Hashable, Mapping
from functools import reduce
from random import choice

import ipdb
import numpy as np
from fastcore.basics import Dict
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Compose, MapTransform
from monai.transforms.io.array import SaveImage
from monai.transforms.transform import Transform

# %%
from fran.preprocessing.patch import contains_bg_only
from fran.transforms.imageio import LoadSITKd
<<<<<<< HEAD
from fran.transforms.intensitytransforms import standardize
=======
from fran.transforms.intensitytransforms import NormaliseClipd, standardize
>>>>>>> efc2e4fb (jj)
from fran.transforms.spatialtransforms import *
from fran.utils.helpers import *
from fran.utils.imageviewers import ImageMaskViewer
from fran.utils.string import strip_extension

tr = ipdb.set_trace
from collections.abc import Callable, Sequence
from pathlib import Path

# %%
import itk
import numpy as np
import SimpleITK as sitk
from fastcore.all import listify, store_attr
from monai.data.dataset import Dataset, PersistentDataset
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged, SaveImaged
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd

from fran.utils.common import *
# path=  proj_default_folders.preprocessing_output_folder
# imgs_folder =  proj_default_folders.preprocessing_output_folder/("images")
# masks_folder=  proj_default_folders.preprocessing_output_folder/("masks")
#
from fran.utils.fileio import *
from fran.utils.fileio import maybe_makedirs
from fran.utils.itk_sitk import ConvertSimpleItkImageToItkImage


class InferenceDatasetNii(Dataset):
    def __init__(self, project, imgs, dataset_params, reader=None):

        self.reader = reader
        self.dataset_params = dataset_params
        self.project = project
        self.imgs = self.parse_input(imgs)
        self.create_transforms()

    def __len__(self) -> int:
        return len(self.imgs)

    def parse_input(self, imgs_inp):
        """
        input types:
            folder of img_fns
            nifti img_fns
            itk imgs (slicer)
        returns list of img_fns if folder. Otherwise just the imgs
        """

        if not isinstance(imgs_inp, list):
            imgs_inp = [imgs_inp]
        imgs_out = []
        for dat in imgs_inp:
            if any([isinstance(dat, str), isinstance(dat, Path)]):
                self.input_type = "files"
                dat = Path(dat)
                if dat.is_dir():
                    dat = list(dat.glob("*"))
                else:
                    dat = [dat]
            else:
                self.input_type = "itk"
                if isinstance(dat, sitk.Image):
                    dat = ConvertSimpleItkImageToItkImage(dat, itk.F)
                # if isinstance(dat,itk.Image):
                dat = itm(dat)
            imgs_out.extend(dat)
        imgs_out = [{"image": img} for img in imgs_out]
        return imgs_out

    def create_transforms(self):
        # single letter name is must for each tfm to use with set_transforms
        if self.reader:
            self.L = LoadImaged(
                keys=["image"],
                image_only=True,
                ensure_channel_first=False,
                simple_keys=True,
                reader=self.reader,
            )
        else:
            self.L = LoadSITKd(
                keys=["image"],
                image_only=True,
                ensure_channel_first=False,
                simple_keys=True,
            )
        self.E = EnsureChannelFirstd(
            keys=["image"], channel_dim="no_channel"
        )  # this creates funny shapes mismatch
        self.S = Spacingd(keys=["image"], pixdim=self.dataset_params["spacing"])
        self.N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )

        self.O = Orientationd(keys=["image"], axcodes="RPS")  # nOTE RPS

        # tfms += [E,S,N]

        # self.transform=Compose(tfms)
    #CODE: match set_transforms function thoughout project
    def set_transforms(self, tfms: str = ""):
        tfms_final = []
        for tfm in tfms:
            tfms_final.append(getattr(self, tfm))
        if self.input_type == "files":
            tfms_final.insert(0, self.L)
        self.transform = Compose(tfms_final)

    def __getitem__(self, index):
        dici = self.imgs[index]
        if self.transform:
            dici = self.transform(dici)
        return dici


# class InferenceDatasetCascade(InferenceDatasetNii):
#     '''
#     This creates two image formats, one low-res ,and one high-res
#     '''
#     def __init__(self,project, imgs,dataset_params_w, dataset_params_p):
#         super().__init__(project, imgs,dataset_params_p)
#         self.dataset_params_w= dataset_params_w
#
#     def set_transforms(I):
#         super().set_transforms()
#         self.transform_w = Resize(spatial_size=self.dataset_params_w["patch_size"])
#
#     def __getitem__(self, index):
#         dici = super().__getitem__(index)
#         dici['image_w'] = self.transform_w(dici['image'])
#         return dici
#


class InferenceDatasetPersistent(InferenceDatasetNii, PersistentDataset):
    def __init__(
        self,
        project,
        data: Sequence,
        dataset_params,
        cache_dir,
        hash_func: Callable[..., bytes] = ...,
        pickle_module: str = "pickle",
        pickle_protocol: int = ...,
        hash_transform=None,
        reset_ops_id: bool = True,
    ) -> None:
        maybe_makedirs(cache_dir)
        InferenceDatasetNii.__init__(
            self, project=project, imgs=data, dataset_params=dataset_params
        )
        PersistentDataset.__init__(
            data=data,
            transform=self.transform,
            cache_dir=cache_dir,
            hash_func=hash_func,
            pickle_module=pickle_module,
            pickle_protocol=pickle_protocol,
            hash_transform=hash_transform,
            reset_ops_id=reset_ops_id,
        )


# export
def foldername_from_shape(parent_folder, shape):
    shape = str(shape).strip("[]").replace(",", "_").replace(" ", "")
    output = Path(parent_folder) / shape
    return output


def maybe_set_property(func):
    def inner(cls, *args, **kwargs):
        prop_name = "_" + func.__name__
        if not hasattr(cls, prop_name):
            prop = func(cls, *args, **kwargs)
            setattr(cls, prop_name, prop)
        return getattr(cls, prop_name)

    return inner


class SimpleDataset(Dataset):
    def __init__(self, data, transform=None) -> None:
        """
        takes files_list, converts to pt format, and creates img/mask pair dataset
        fnames: files_list
        """

        super().__init__()
        self.data = data

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        dici = self.data[ind]
        if self.transform:
            dici = self.transform(dici)
        return dici


class ImageMaskBBoxDataset(Dataset):
    """
    takes a list of case_ids and returns bboxes image and label
    """

    def __init__(self, fnames, bbox_fn, class_ratios: list = None, transform=None):
        """
        ratios decide the proportionate guarantee of each class in the output including background. While that class is guaranteed to be present at that frequency, others may still be present if they coexist. However, the exception to this is label 0. This allows some patches which have no foreground voxels to participate in trianing too.
        """
        self.transform = transform
        if not class_ratios:
            self.enforce_ratios = False
        else:
            self.class_ratios = class_ratios
            self.enforce_ratios = True

        print("Loading dataset from BBox file {}".format(bbox_fn))
        bboxes_unsorted = load_dict(bbox_fn)
        self.bboxes_per_id = []
        for fn in fnames:
            bboxes = self.match_raw_filename(bboxes_unsorted, fn)
            bboxes.append(self.get_label_info(bboxes))
            self.bboxes_per_id.append(bboxes)

    def match_raw_filename(self, bboxes, fname: str):
        bboxes_out = []
        fname = strip_extension(fname)
        for bb in bboxes:
            fn = bb["filename"]
            fn_no_suffix = cleanup_fname(fn.name)
            if fn_no_suffix == fname:
                bboxes_out.append(bb)
        if len(bboxes_out) == 0:
            print("Missing filename {0} from bboxfile".format(fname))
            tr()
        return bboxes_out

    def __len__(self):
        return len(self.bboxes_per_id)

    def __getitem__(self, idx):
        self.set_bboxes_labels(idx)
        if self.enforce_ratios == True:
            self.mandatory_label = self.randomize_label()
            self.maybe_randomize_idx()

        filename, bbox = self.get_filename_bbox()
        img, label = self.load_tensors(filename)
        if self.transform is not None:
            img, label, bbox = self.transform([img, label, bbox])

        return img, label, bbox

    def load_tensors(self, filename: Path):
        label = torch.load(filename)
        if isinstance(label, dict):
            img, label = label["img"], label["label"]
        else:
            img_folder = filename.parent.parent / ("images")
            img_fn = img_folder / filename.name
            img = torch.load(img_fn)
        return img, label

    def set_bboxes_labels(self, idx):
        self.bboxes = self.bboxes_per_id[idx][:-1]
        self.label_info = self.bboxes_per_id[idx][-1]

    def get_filename_bbox(self):
        if self.enforce_ratios == True:
            candidate_indices = self.get_inds_with_label()
        else:
            candidate_indices = range(0, len(self.bboxes))
        sub_idx = choice(candidate_indices)
        bbox = self.bboxes[sub_idx]
        fn = bbox["filename"]
        return fn, bbox

    def maybe_randomize_idx(self):
        while (self.mandatory_label not in self.label_info["labels_this_case"])^(self.mandatory_label==0 and len(self.label_info["labels_this_case"])==1) :
            idx = np.random.randint(0, len(self))
            self.set_bboxes_labels(idx)

    def get_inds_with_label(self):
        labels_per_file = self.label_info["labels_per_file"]
        inds_label_status = [
            self.mandatory_label in labels for labels in labels_per_file
        ]
        indices = self.label_info["file_indices"]
        inds_with_label = list(itertools.compress(indices, inds_label_status))
        return inds_with_label

    def randomize_label(self):
        mandatory = np.random.multinomial(1, self.class_ratios, 1)
        _, mandatory_label = np.where(mandatory == 1)
        return mandatory_label.item()

    def shape_per_id(self, id):
        bb = self.bboxes_per_id[id]
        bb_stats = bb[0]["bbox_stats"]
        bb_any = bb_stats[0]["bounding_boxes"][0]
        shape = [sl.stop for sl in bb_any]
        return shape

    def get_label_info(self, case_bboxes):
        indices = []
        labels_per_file = []
        for indx, bb in enumerate(case_bboxes):
            bbox_stats = bb["bbox_stats"]
            labels = [(a["label"]) for a in bbox_stats if not a["label"] == "all_fg"]
            if contains_bg_only(bbox_stats) == True:
                labels = [0]
            else:
                labels = [0]+labels
            # if len(labels) == 0:
            #     tr()
            #     labels = [0]  # background class only by exclusion
            indices.append(indx)
            labels_per_file.append(labels)
        labels_this_case = list(set(reduce(operator.add, labels_per_file)))
        return {
            "file_indices": indices,
            "labels_per_file": labels_per_file,
            "labels_this_case": labels_this_case,
        }

    @property
    def class_ratios(self):
        """The ratios property."""
        return self._ratios

    @class_ratios.setter
    def class_ratios(self, raw_ratios):
        denom = reduce(operator.add, raw_ratios)
        self._ratios = [x / denom for x in raw_ratios]

    @property
    @maybe_set_property
    def median_shape(self):
        aa = []
        for i in range(len(self)):
            aa.append(self.shape_per_id(i))
        return np.median(aa, 0).astype(int)

    @property
    @maybe_set_property
    def parent_folder(self):
        fn, _ = self.get_filename_bbox(0)
        return fn.parent.parent

    @property
    @maybe_set_property
    def dataset_min(self):
        try:
            data_properties = load_dict(
                self.parent_folder.parent / ("resampled_dataset_properties")
            )
        except:
            raise FileNotFoundError
        return data_properties["dataset_min"]

    def contains_bg(self, bbox_stats):
        all_fg_bbox = [bb for bb in bbox_stats if bb["label"] == "all_fg"][0]
        bboxes = all_fg_bbox["bounding_boxes"]
        if len(bboxes) == 1:
            return True
        if bboxes[0] != bboxes[1]:
            return True


class ImageMaskBBoxDatasetd(ImageMaskBBoxDataset):
    def __getitem__(self, idx):
        self.set_bboxes_labels(idx)
        if self.enforce_ratios == True:
            self.mandatory_label = self.randomize_label()
            self.maybe_randomize_idx()

        filename, bbox = self.get_filename_bbox()
        img, lm = self.load_tensors(filename)
        dici = {"image": img, "lm": lm, "bbox": bbox}
        if self.transform is not None:
            dici = self.transform(dici)

        return dici


class SavePatchd(MapTransform):
    """
    input data must be a dictionary, Must contain a bbox key to create a full-sized image from
    """

    def __init__(self, keys, output_folder, postfix_channel=False):
        super().__init__(keys, False)
        store_attr("output_folder,postfix_channel")

    def func(self, cropped_tnsr, bbox):
        chs = cropped_tnsr.shape[0]
        for ch in range(1, chs):
            postfix = str(ch) if self.postfix_channel == True else None
            img_full = fill_bbox(bbox, cropped_tnsr)
            img_save = img_full[ch : ch + 1]

            S = SaveImage(
                output_dir=self.output_folder,
                output_postfix=postfix,
                separate_folder=False,
            )
            S(img_save)

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.func(d[key], d["bbox"])
        return d


class CropImgMaskd(MapTransform):

    def __init__(self, patch_size, input_dims):
        self.dim = len(patch_size)
        self.patch_halved = [int(x / 2) for x in patch_size]
        self.input_dims = input_dims

    def func(self, x):
        img, label = x
        center = [x / 2 for x in img.shape[-self.dim :]]
        slices = [
            slice(None),
        ] * (
            self.input_dims - 3
        )  # batch and channel dims if its a batch otherwise empty
        for ind in range(self.dim):
            source_sz = center[ind]
            target_sz = self.patch_halved[ind]
            if source_sz > target_sz:
                slc = slice(int(source_sz - target_sz), int(source_sz + target_sz))
            else:
                slc = slice(None)
            slices.append(slc)
        img, label = img[slices], label[slices]
        return img, label


class Affine3D(MapTransform):
    """
    to-do: verify if nearestneighbour method preserves multiple mask labels
    """

    def __init__(
        self,
        keys,
        p=0.5,
        rotate_max=pi / 6,
        translate_factor=0.0,
        scale_ranges=[0.75, 1.25],
        shear: bool = True,
        allow_missing_keys=False,
    ):
        """
        params:
        scale_ranges: [min,max]
        """
        super().__init__(keys, allow_missing_keys)
        store_attr("p,rotate_max,translate_factor,scale_ranges,shear")

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
        return d

    def get_mode(self, x):
        dt = x.dtype
        if dt == torch.uint8:
            mode = "nearest"
        elif dt == torch.float32 or x.dtype == torch.float16:
            mode = "bilinear"
        return mode, dt

    def func(self, x):
        mode, dt = self.get_mode(x)

        if np.random.rand() < self.p:
            grid = get_affine_grid(
                x.shape,
                shear=self.shear,
                scale_ranges=self.scale_ranges,
                rotate_max=self.rotate_max,
                translate_factor=self.translate_factor,
                device=x.device,
            ).type(torch.float32)
            x = F.grid_sample(x.type(x.dtype), grid, mode=mode)
        return x.to(dt)


#


def fill_bbox(bbox, cropped_tensor):
    """
    bbox : 3-tuple of slices
    cropped_tensor: 4d CxWxHxD. It has metadata with spatial_shape key defining full tensor shape
    """

    n_ch = cropped_tensor.shape[0]
    shape = [n_ch] + list(cropped_tensor.meta["spatial_shape"])
    full = torch.zeros(shape)
    full[bbox] = cropped_tensor
    out_tensor = MetaTensor(full)
    out_tensor.copy_meta_from(cropped_tensor)
    return out_tensor


class FillBBoxPatchesd(Transform):
    """
    Based on size of original image and n_channels output by model, it creates a zerofilled tensor. Then it fills locations of input-bbox with data provided
    """

    def __call__(self, d):
        """
        d is a dict with keys: 'image','pred','bbox'
        """
        pred = d["pred"]
        bbox = d["bounding_box"]
        pred_out = fill_bbox(bbox, pred)
        d["pred"] = pred_out
        return d


class MaskLabelRemapd(MapTransform):
    def __init__(self, keys, src_dest_labels: tuple, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        if isinstance(src_dest_labels, str):
            src_dest_labels = ast.literal_eval(src_dest_labels)
        self.src_dest_labels = src_dest_labels

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.remapper(d[key])
        return d

    def remapper(self, mask):
        n_classes = len(self.src_dest_labels)
        mask_out = torch.zeros(mask.shape, dtype=mask.dtype)
        mask_tmp = one_hot(mask, n_classes, 0)
        mask_reassigned = torch.zeros(mask_tmp.shape, device=mask.device)
        for src_des in self.src_dest_labels:
            src, dest = src_des[0], src_des[1]
            mask_reassigned[dest] += mask_tmp[src]

        for x in range(n_classes):
            mask_out[torch.isin(mask_reassigned[x], 1.0)] = x
        return mask_out


# %%
if __name__ == "__main__":
    from fran.inference.base import load_dataset_params
    from fran.utils.common import *

    P = Project(project_title="lidc2")

    global_properties = load_dict(P.global_properties_filename)
# %%
