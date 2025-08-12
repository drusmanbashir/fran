# %%
from __future__ import annotations
from torch.serialization import safe_globals
from monai.data import ImageWriter
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import SimpleITK as sitk
import torch
from fastcore.basics import store_attr, warnings
from label_analysis.helpers import get_labels as gl
from monai.config import PathLike
from monai.config.type_definitions import KeysCollection
from monai.data.image_reader import (
    ImageReader,
    ITKReader,
    NibabelReader,
    NrrdReader,
    NumpyReader,
    PILReader,
    PydicomReader,
    _copy_compatible_dict,
    _stack_images,
)
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import is_supported_format, orientation_ras_lps
from monai.transforms.io.array import SUPPORTED_READERS, LoadImage
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.transform import MapTransform
from monai.transforms.utility.array import EnsureChannelFirst 
from monai.utils import ImageMetaKey as Key
from monai.utils import ensure_tuple, optional_import
from monai.utils.enums import MetaKeys, SpaceKeys
from monai.utils.module import optional_import, require_pkg

from fran.transforms.totensor import ToTensorT
from utilz.fileio import load_dict

nib, _ = optional_import("nibabel")
Image, _ = optional_import("PIL.Image")
nrrd, _ = optional_import("nrrd")

__all__ = ["LoadImage", "SaveImage", "SUPPORTED_READERS"]

SUPPORTED_READERS = {
    "pydicomreader": PydicomReader,
    "itkreader": ITKReader,
    "nrrdreader": NrrdReader,
    "numpyreader": NumpyReader,
    "pilreader": PILReader,
    "nibabelreader": NibabelReader,
}
import ipdb

from utilz.imageviewers import ImageMaskViewer

tr = ipdb.set_trace


class LoadSITKd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        reverse_indexing: bool = False,
        ensure_channel_first: bool = False,
        affine_lps_to_ras=True,
        lm_key=None,  # if provided it will store number of labels in LM
        channel_dim: str | int | None = None,
        simple_keys: bool = False,
        pattern=None,
        sep: str = ".",
        image_only: bool = False,
    ) -> None:
        """
        Multipurpose function to load images in SITK format. Can also directly take sitk.Image object.
        """

        super().__init__(keys, allow_missing_keys)
        store_attr(
            "reverse_indexing,ensure_channel_first,affine_lps_to_ras,channel_dim,simple_keys,pattern,sep,image_only,lm_key"
        )

    def _get_meta_dict(self, img) -> dict:
        """
        Get all the metadata of the image and convert to dict type.

        Args:
            img: an ITK image object loaded from an image file.

        """
        meta_dict = {}
        meta_dict["spacing"] = np.asarray(img.GetSpacing())
        return meta_dict

    def _get_affine(self, img, lps_to_ras: bool = True):
        """
        Get or construct the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: an ITK image object loaded from an image file.
            lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to True.

        """

        direction = img.GetDirection()
        if len(direction) != 9:
            raise NotImplemented
        else:
            direction = np.array(direction)
            direction = direction.reshape(3, 3)

        spacing = np.asarray(img.GetSpacing())
        origin = np.asarray(img.GetOrigin())

        sr = min(max(direction.shape[0], 1), 3)
        affine: np.ndarray = np.eye(sr + 1)
        affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
        affine[:sr, -1] = origin[:sr]
        if lps_to_ras:
            affine = orientation_ras_lps(affine)
        return affine

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of `img`.

        Args:
            img: a  SITK image object loaded from an image file.

        """
        sr = img.GetDimension()
        sr = max(min(sr, 3), 1)
        _size = list(img.GetSize())
        if isinstance(self.channel_dim, int):
            _size.pop(self.channel_dim)
        return np.asarray(_size[:sr])

    def _get_array_data(self, img):
        """
        Get the raw array data of the image, converted to Numpy array.

        Following PyTorch conventions, the returned array data has contiguous channels,
        e.g. for an RGB image, all red channel image pixels are contiguous in memory.
        The last axis of the returned array is the channel axis.

        See also:

            - https://github.com/InsightSoftwareConsortium/ITK/blob/v5.2.1/Modules/Bridge/NumPy/wrapping/PyBuffer.i.in

        Args:
            img: an ITK image object loaded from an image file.

        """
        np_img = sitk.GetArrayFromImage(img)
        if img.GetNumberOfComponentsPerPixel() == 1:  # handling spatial images
            return np_img if self.reverse_indexing else np_img.T
        # handling multi-channel images
        return np_img if self.reverse_indexing else np.moveaxis(np_img.T, 0, -1)

    def get_data(self, img, get_labels=False) -> tuple[np.ndarray, dict]:
        """
        Args:
            img: an SITK image object loaded from an image file or a list of ITK image objects.

        """

        img = self.maybe_recast_sitk(img)
        data = self._get_array_data(img)
        header = self._get_meta_dict(img)
        header[MetaKeys.ORIGINAL_AFFINE] = self._get_affine(img, self.affine_lps_to_ras)
        header[MetaKeys.SPACE] = (
            SpaceKeys.RAS if self.affine_lps_to_ras else SpaceKeys.LPS
        )
        header[MetaKeys.AFFINE] = header[MetaKeys.ORIGINAL_AFFINE].copy()
        header[MetaKeys.SPATIAL_SHAPE] = self._get_spatial_shape(img)
        if self.channel_dim is None:  # default to "no_channel" or -1
            header[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
                float("nan")
                if len(data.shape) == len(header[MetaKeys.SPATIAL_SHAPE])
                else -1
            )
        else:
            header[MetaKeys.ORIGINAL_CHANNEL_DIM] = self.channel_dim
        if get_labels == True:
            labels = gl(img)
            header["labels"] = labels

        return data, header

    def maybe_recast_sitk(self, img: sitk.Image):
        pixid = img.GetPixelID()
        if pixid in [2, 3, 5]:  # unsigned signed 16 bit, unsigned 32 int
            img = sitk.Cast(img, sitk.sitkInt32)
        return img

    def func(self, img: Union[str, Path, sitk.Image], get_labels=False):
        if isinstance(img, Path) or isinstance(
            img, str
        ):  # for compatibility with Slicerpython 3.9
            img_fn = img
            img = sitk.ReadImage(str(img))
            array_np, meta_data = self.get_data(img, get_labels)
            meta_data[Key.FILENAME_OR_OBJ] = str(img_fn)
        elif isinstance(img, sitk.Image):
            array_np, meta_data = self.get_data(img, get_labels)
        else:
            raise TypeError("img must be a str filename or a sitk.Image object")
        array_pt = ToTensorT()(array_np)

        img = MetaTensor.ensure_torch_and_prune_meta(
            array_pt, meta_data, self.simple_keys, pattern=self.pattern, sep=self.sep
        )
        if self.ensure_channel_first:
            img = EnsureChannelFirst()(img)
        return img

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.lm_key is not None and key == self.lm_key:
                d[key] = self.func(d[key], get_labels=True)
            else:
                d[key] = self.func(d[key], get_labels=False)
        return d


@require_pkg(pkg_name="SimpleITK")
class SITKReader(ITKReader):
    """
    MASSIVE FAIL
    Load medical images based on ITK library.
    All the supported image formats can be found at:
    https://github.com/InsightSoftwareConsortium/ITK/tree/master/Modules/IO
    The loaded data array will be in C order, for example, a 3D image NumPy
    array index order will be `CDWH`.

    Args:
        channel_dim: the channel dimension of the input image, default is None.
            This is used to set original_channel_dim in the metadata, EnsureChannelFirstD reads this field.
            If None, `original_channel_dim` will be either `no_channel` or `-1`.

                - Nifti file is usually "channel last", so there is no need to specify this argument.
                - PNG file usually has `GetNumberOfComponentsPerPixel()==3`, so there is no need to specify this argument.

        series_name: the name of the DICOM series if there are multiple ones.
            used when loading DICOM series.
        reverse_indexing: whether to use a reversed spatial indexing convention for the returned data array.
            If ``False``, the spatial indexing follows the numpy convention;
            otherwise, the spatial indexing convention is reversed to be compatible with ITK. Default is ``False``.
            This option does not affect the metadata.
        series_meta: whether to load the metadata of the DICOM series (using the metadata from the first slice).
            This flag is checked only when loading DICOM series. Default is ``False``.
        affine_lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to ``True``.
            Set to ``True`` to be consistent with ``NibabelReader``, otherwise the affine matrix remains in the ITK convention.
        kwargs: additional args for `itk.imread` API. more details about available args:
            https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itk/support/extras.py

    """

    def read(self, data: Sequence[PathLike] | PathLike, **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.
        If passing directory path instead of file path, will treat it as DICOM images series and read.
        Note that the returned object is ITK image object or list of ITK image objects.

        Args:
            data: file name or a list of file names to read,
            kwargs: additional args for `itk.imread` API, will override `self.kwargs` for existing keys.
                More details about available args:
                https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itk/support/extras.py

        """
        img_ = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            name = f"{name}"
            if Path(name).is_dir():
                raise Exception(f"{name} is a directory, not a file.")
            else:
                img_.append(sitk.ReadImage(name, **kwargs_))
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the metadata of the first image is used to represent the output metadata.

        Args:
            img: an ITK image object loaded from an image file or a list of ITK image objects.

        """
        img_array: list[np.ndarray] = []
        compatible_meta: dict = {}
        if not isinstance(img, Union[list, tuple]):
            img = [img]

        for i in img:  # ensure_tuple freezes with sitk images
            data = self._get_array_data(i)
            img_array.append(data)
            header = self._get_meta_dict(i)
            header[MetaKeys.ORIGINAL_AFFINE] = self._get_affine(
                i, self.affine_lps_to_ras
            )
            header[MetaKeys.SPACE] = (
                SpaceKeys.RAS if self.affine_lps_to_ras else SpaceKeys.LPS
            )
            header[MetaKeys.AFFINE] = header[MetaKeys.ORIGINAL_AFFINE].copy()
            header[MetaKeys.SPATIAL_SHAPE] = self._get_spatial_shape(i)
            if self.channel_dim is None:  # default to "no_channel" or -1
                header[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
                    float("nan")
                    if len(data.shape) == len(header[MetaKeys.SPATIAL_SHAPE])
                    else -1
                )
            else:
                header[MetaKeys.ORIGINAL_CHANNEL_DIM] = self.channel_dim
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> dict:
        """
        Get all the metadata of the image and convert to dict type.

        Args:
            img: an ITK image object loaded from an image file.

        """
        meta_dict = {}
        meta_dict["spacing"] = np.asarray(img.GetSpacing())
        #
        # img_meta_dict = img.GetMetaDataDictionary()
        #
        # meta_dict = {}
        # for key in img_meta_dict.GetKeys():
        #     if key.startswith("ITK_"):
        #         continue
        #     val = img_meta_dict[key]
        #     meta_dict[key] = np.asarray(val) if type(val).__name__.startswith("itk") else val

        return meta_dict

    def _get_affine(self, img, lps_to_ras: bool = True):
        """
        Get or construct the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: an ITK image object loaded from an image file.
            lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to True.

        """

        direction = img.GetDirection()
        if len(direction) != 9:
            raise NotImplemented
        else:
            direction = np.array(direction)
            direction = direction.reshape(3, 3)

        spacing = np.asarray(img.GetSpacing())
        origin = np.asarray(img.GetOrigin())

        sr = min(max(direction.shape[0], 1), 3)
        affine: np.ndarray = np.eye(sr + 1)
        affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
        affine[:sr, -1] = origin[:sr]
        if lps_to_ras:
            affine = orientation_ras_lps(affine)
        return affine

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of `img`.

        Args:
            img: a  SITK image object loaded from an image file.

        """
        sr = img.GetDimension()
        sr = max(min(sr, 3), 1)
        _size = list(img.GetSize())
        if isinstance(self.channel_dim, int):
            _size.pop(self.channel_dim)
        return np.asarray(_size[:sr])

    def _get_array_data(self, img):
        """
        Get the raw array data of the image, converted to Numpy array.

        Following PyTorch conventions, the returned array data has contiguous channels,
        e.g. for an RGB image, all red channel image pixels are contiguous in memory.
        The last axis of the returned array is the channel axis.

        See also:

            - https://github.com/InsightSoftwareConsortium/ITK/blob/v5.2.1/Modules/Bridge/NumPy/wrapping/PyBuffer.i.in

        Args:
            img: an ITK image object loaded from an image file.

        """
        np_img = sitk.GetArrayFromImage(img)
        if img.GetNumberOfComponentsPerPixel() == 1:  # handling spatial images
            return np_img if self.reverse_indexing else np_img.T
        # handling multi-channel images
        return np_img if self.reverse_indexing else np.moveaxis(np_img.T, 0, -1)


class TorchReader(ImageReader):

    def get_data(self, img) -> tuple[torch.Tensor, dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the metadata of the first image is used to represent the output metadata.

        Args:
            img: an ITK image object loaded from an image file or a list of ITK image objects.

        """
        img_array: list[torch.Tensor] = []
        for i in ensure_tuple(img):
            header = i.meta
            img_array.append(i)

        img_array = torch.stack(img_array, 0)
        img_array = np.array(img_array)
        return img_array, header

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
        Verify whether the specified file or files format is supported by Numpy reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.
        """
        suffixes: Sequence[str] = ["pt"]
        return is_supported_format(filename, suffixes)

    def read(self, data, **kwargs):
        img_: list = []
        filenames: Sequence[PathLike] = ensure_tuple(data)
        for name in filenames:
                img = torch.load(name,weights_only=False, **kwargs)
                img_.append(img)
        return img_ if len(img_) > 1 else img_[0]


class LoadTorchd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Multipurpose function to load images in SITK format. Can also directly take sitk.Image object.
        """

        super().__init__(keys, allow_missing_keys)

    def __call__(self, d):
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
        return d

    def func(self, fn):
        with safe_globals([MetaTensor]):
            img = torch.load(fn, weights_only=False)
        meta = img.meta
        meta["src_filename"] = meta["filename_or_obj"]
        meta["filename_or_obj"] = str(fn)
        img.meta = meta
        return img


class TorchWriter(ImageWriter):
    def set_data_array(self, data_array, **kwargs):
        self.data_obj = data_array

    def set_metadata(self, meta_dict, resample, **kwargs):
        if hasattr(self.data_obj, "meta"):
            self.data_obj.meta = meta_dict

    def write(self, filename, verbose: bool = False, **kwargs):
        super().write(filename, verbose, **kwargs)
        torch.save(self.data_obj, filename)


# %%
if __name__ == "__main__":

    import torch

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *

    project_title = "nodes"
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    )
    configuration_filename = None

    configs = ConfigMaker(
        proj, raytune=False, configuration_filename=configuration_filename
    ).config

    global_props = load_dict(proj.global_properties_filename)
    # %%
    fn = "/s/xnat_shadow/crc/images/crc_CRC261_20170322_AbdoPelvis1p5.nii.gz"
    fn_pt = "/s/fran_storage/datasets/preprocessed/fixed_spacings/lilu/spc_080_080_150/images/drli_001ub.pt"
    dici = {"image": fn}

    L = LoadImaged(keys=["image"])
    img = L(dici)
    # %%
    dici_pt = {"image": fn_pt}
    Lp = LoadTorchd(keys=["image"])
    img_pt = Lp(dici_pt)

    # %%
    dici = {"image": fn}
    from time import time

    L1 = LoadSITKd(keys=["image"])
    L = LoadImaged(keys=["image"])
    # %%
    start = time()
    for n in range(5):
        img = L1(dici)
    stop = time()
    spent = stop - start
    print(spent)

    # %%
    start = time()
    for n in range(5):
        img = L(dici)
    stop = time()
    spent = stop - start
    print(spent)
    # %%

    L2 = LoadImaged(keys=["image"], reader=SITKReader)
    img2 = L2(dici)

    img2["image"].meta
    # %%
    L3 = LoadSITKd(keys=["image"])
    im3 = L3(dici)

    # %%
    ImageMaskViewer([img2["image"], im3["image"]], data_types=["image", "image"])
    # %%
    fname = fn
    img = sitk.ReadImage(fname)
