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
from label_analysis.helpers import get_labels
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Compose, MapTransform
from monai.transforms.io.array import SaveImage
from monai.transforms.transform import Transform
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from torch.utils.data import Dataset
from fran.data.dataloader import  img_mask_metadata_lists_collated

from fran.managers.project import get_ds_remapping
from fran.transforms.imageio import LoadSITKd
from fran.transforms.intensitytransforms import standardize
from fran.transforms.misc_transforms import AddMetadata, HalfPrecisiond, RemapSITKImage
from fran.transforms.spatialtransforms import *
from fran.utils.helpers import *
from fran.utils.imageviewers import ImageMaskViewer
from fran.utils.string import strip_extension

tr = ipdb.set_trace
import itertools as il
from collections.abc import Callable, Sequence
from pathlib import Path

import itk
import numpy as np
import SimpleITK as sitk
from fastcore.all import listify, store_attr
from fastcore.foundation import GetAttr
from lightning.fabric import Fabric
from monai.data.dataloader import DataLoader
from monai.data.dataset import  PersistentDataset

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
        self.E = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel") # this creates funny shapes mismatch
        self.S = Spacingd(keys=["image"], pixdim=self.dataset_params["spacings"])
        self.N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )

        self.O = Orientationd(keys=["image"], axcodes="RPS")  # nOTE RPS

        # tfms += [E,S,N]

        # self.transform=Compose(tfms)

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


#
# class InferenceDatasetCascade(InferenceDatasetNii):
#     '''
#     This creates two image formats, one low-res ,and one high-res
#     '''
#     def __init__(self,project, imgs,dataset_params_w, dataset_params_p):
#         super().__init__(project, imgs,dataset_params_p)
#         self.dataset_params_w= dataset_params_w
#
#     def set_transforms(self):
#         super().set_transforms()
#         self.transform_w = Resize(spatial_size=self.dataset_params_w["patch_size"])
#
#     def __getitem__(self, index):
#         dici = super().__getitem__(index)
#         dici['image_w'] = self.transform_w(dici['image'])
#         return dici
#
class ResamplerDataset(GetAttr,Dataset):
    _default = "project"

    def __init__(
        self,
        project,
        spacings,
        half_precision=False,
        clip_center=False,

        mean_std_mode: str = "dataset",
    ):

        assert mean_std_mode in [
            "dataset",
            "fg",
        ], "Select either dataset mean/std or fg mean/std for normalization"
        self.project = project
        self.df = self.filter_completed_cases()
        self.spacings = spacings
        self.half_precision = half_precision
        self.clip_center = clip_center
        super(GetAttr).__init__()
        self.set_normalization_values(mean_std_mode)
        self.create_transforms()

    def filter_completed_cases(self):
        df = self.project.df.copy() # speed up things
        return df
    def __len__(self): return len(self.df)   

    def __getitem__(self, index):
        cp = self.df.iloc[index]
        ds = cp['ds']
        remapping = get_ds_remapping(ds, self.global_properties)

        img_fname = cp["image"]
        mask_fname = cp["lm"]
        img = sitk.ReadImage(img_fname)
        mask = sitk.ReadImage(mask_fname)
        dici = {
            "image": img,
            "mask": mask,
            "image_fname": img_fname,
            "mask_fname": mask_fname,
            "remapping": remapping,
        }
        dici = self.transform(dici)
        return dici



    def create_transforms(self):
        R = RemapSITKImage(keys=["mask"])
        L = LoadSITKd(keys=["image", "mask"], image_only=True)

        Ai = AddMetadata(keys=['image'], meta_keys=['image_fname'],renamed_keys=['filename'])
        Am = AddMetadata(keys=['mask'], meta_keys=['mask_fname','remapping'],renamed_keys=['filename','remapping'])
        E = EnsureChannelFirstd(keys=["image","mask"], channel_dim="no_channel") # funny shape output mismatch
        Si = Spacingd(keys=["image"], pixdim=self.spacings,mode="trilinear")
        Rz = ResizeDynamicd(keys=["mask"], key_spatial_size='image', mode='nearest')

        #Sm = Spacingd(keys=["mask"], pixdim=self.spacings,mode="nearest")
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.global_properties["intensity_clip_range"],
            mean=self.mean,
            std=self.std,
        )
        tfms = [R,L,Ai,Am,E,Si, Rz]
        if self.clip_center == True:
            tfms.extend([N])
        if self.half_precision == True:
            H = HalfPrecisiond(keys=["image"])
            tfms.extend([H])
        self.transform = Compose(tfms)




    def set_normalization_values(self, mean_std_mode):
        if mean_std_mode == "dataset":
            self.mean = self.global_properties["mean_dataset_clipped"]
            self.std = self.global_properties["std_dataset_clipped"]
        else:
            self.mean = self.global_properties["mean_fg"]
            self.std = self.global_properties["std_fg"]



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
        PersistentDataset().__init__(
            data,
            self.transform,
            cache_dir,
            hash_func,
            pickle_module,
            pickle_protocol,
            hash_transform,
            reset_ops_id,
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


class SimpleDatasetPT(Dataset):
    def __init__(self, parent_folder, fnames, transform=None) -> None:
        """
        takes files_list, converts to pt format, and creates img/mask pair dataset
        fnames: files_list 
        """

        super().__init__()
        fnames = [strip_extension(fn) for fn in fnames]
        fnames = [fn + ".pt" for fn in fnames]
        fnames = fnames
        images_fldr = parent_folder / ("images")
        masks_fldr = parent_folder / ("masks")
        imgs = list(images_fldr.glob("*"))
        masks = list(masks_fldr.glob("*"))
        self.img_mask_pairs = []
        for fn in fnames:
            fn = Path(fn)
            img_mask_pair = [find_matching_fn(fn, imgs), find_matching_fn(fn, masks)]
            self.img_mask_pairs.append(img_mask_pair)

        self.transform = transform

    def __len__(self):
        return len(self.img_mask_pairs)

    def __getitem__(self, ind):
        img_fn, label_fn = self.img_mask_pairs[ind]
        # img = self.create_metatensor(img_fn)
        # label = self.create_metatensor(label_fn)
        dici = {"image": img_fn, "label": label_fn}
        if self.transform:
            dici = self.transform(dici)
        return dici


class ImageMaskBBoxDataset(Dataset):
    """
    takes a list of case_ids and returns bboxes image and label
    """

    def __init__(self, fnames, bbox_fn, class_ratios: list = None, transform=None):
        """
        class_ratios decide the proportionate guarantee of each class in the output including background. While that class is guaranteed to be present at that frequency, others may still be present if they coexist
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
        while self.mandatory_label not in self.label_info["labels_this_case"]:
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
            if self.contains_bg(bbox_stats):
                labels = [0] + labels
            if len(labels) == 0:
                labels = [0]  # background class only by exclusion
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
        """The class_ratios property."""
        return self._class_ratios

    @class_ratios.setter
    def class_ratios(self, raw_ratios):
        denom = reduce(operator.add, raw_ratios)
        self._class_ratios = [x / denom for x in raw_ratios]

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
        img, label = self.load_tensors(filename)
        dici = {"image": img, "label": label, "bbox": bbox}
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
class NormaliseClip(Transform):
    def __init__(self, clip_range, mean, std):
        # super().__init__(keys, allow_missing_keys)

        store_attr("clip_range,mean,std")

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        d = self.clipper(data)
        return d

    def clipper(self, img):
        img = torch.clip(img, self.clip_range[0], self.clip_range[1])
        img = standardize(img, self.mean, self.std)
        return img


class NormaliseClipd(MapTransform):
    def __init__(self, keys, clip_range, mean, std, allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.N = NormaliseClip(clip_range=clip_range, mean=mean, std=std)

    def __call__(self, d):
        for key in self.key_iterator(d):
            d[key] = self.N(d[key])
        return d


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
        bbox = d["bbox"]
        pred_out = fill_bbox(bbox, pred)
        d["pred"] = pred_out
        return d


class MaskLabelRemap2(MapTransform):
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

    P = Project(project_title="totalseg")

    global_properties = load_dict(P.global_properties_filename)
# %%

    R = ResamplerDataset(project=P, spacings=[3,3,3], half_precision=False)
    dl = DataLoader(dataset=R,num_workers=2,collate_fn = img_mask_metadata_lists_collated,batch_size=2)

    iteri = iter(dl)

    bb = next(iteri)
# %%
    n_items = dl.batch_size
    images,masks = bb['image'], bb['mask']
    image = images[0]
    fname = Path(image.meta['filename'])
    img_fn ="/s/fran_storage/datasets/preprocessed/fixed_spacings/totalseg/spc_300_300_300/images/totalseg_s1418.pt"

    img = torch.load(img_fn)
    img.shape
# %%
    P = Project(project_title="totalseg")
    img_fn =  Path('/s/xnat_shadow/totalseg/images/totalseg_s0627.nii.gz')
    mask_fn =  Path('/s/xnat_shadow/totalseg/masks/totalseg_s0627.nii.gz')

    img = sitk.ReadImage(img_fn)
    mask = sitk.ReadImage(mask_fn)

    remapping = get_ds_remapping('totalseg', P.global_properties)
    dici = {'image':img,'mask':mask, 'remapping':remapping , 
            "image_fname": img_fn,
            "mask_fname": mask_fn,}

# %%

    spacings = [3,3,3]
    R = RemapSITKImage(keys=["mask"])
    L = LoadSITKd(keys=["image", "mask"], image_only=True)

    Ai = AddMetadata(keys=['image'], meta_keys=['image_fname'],renamed_keys=['filename'])
    Am = AddMetadata(keys=['mask'], meta_keys=['mask_fname','remapping'],renamed_keys=['filename','remapping'])
    E = EnsureChannelFirstd(keys=["image","mask"], channel_dim="no_channel")
    Si = Spacingd(keys=["image"], pixdim=spacings,mode="trilinear")

    Rz = ResizeDynamicd(keys=["mask"], key_spatial_size='image', mode='nearest')

    tfms = [R,L,Ai,Am,E,Si, Rz]

    C = Compose(tfms)
# %%
    c = C(dici)
# %%


    print(c['image'].shape)
    print(c['mask'].meta)
# %%
    Lo = LoadImaged(keys=["image", "mask"])
# %%
    dici2 = {'image': img_fn, 'mask':mask_fn}
    a = Lo(dici2)

    print(a['image'].shape)
    print(a['image'].meta)
# %%

    print(c['image'].shape, c['mask'].shape)
    ImageMaskViewer([c['image'][0].permute(2,0,1),c['mask'][0]])
# %%

    image = c['image']
    data = c['mask']
    spatial_size = image[0].shape
    mode ='nearest'
    mask2 =fm.resize(
            img=data,
            out_size=spatial_size,
            mode=mode,
            lazy=False,
            align_corners=None,
            dtype=None,
            input_ndim=3,
            anti_aliasing=False,
            anti_aliasing_sigma=0.0,
            transform_info=None,
        )

    print(mask2.shape)
# %%
    R(dici)
    a = L(dici)
    b = Si(a)
    c2 = Sm(b)
    img = b['img']
    c= Rz(b)
# %%
    print(c['image'].shape)
    print(c['mask'].shape)
# %%

    print(c2['image'].shape)
    print(c2['mask'].shape)
# %%
    index = 0
  
# %%
