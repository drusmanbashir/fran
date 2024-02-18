# %%
from __future__ import annotations
from pathlib import Path
from typing import Hashable, Mapping, Union
import SimpleITK as sitk
from fastcore.foundation import inspect
from monai.transforms.transform import MapTransform, Transform
from monai.data.utils import is_supported_format, orientation_ras_lps
from monai.transforms.compose import Compose
from monai.transforms.io.array import SUPPORTED_READERS, LoadImage, switch_endianness
from monai.transforms.io.dictionary import DEFAULT_POST_FIX, LoadImaged
from monai.utils.enums import MetaKeys, SpaceKeys
from monai.utils.module import optional_import, require_pkg
from neptune.metadata_containers.metadata_container import traceback
import torch
from monai.data.image_reader import ITKReader, _copy_compatible_dict, _stack_images
import numpy as np
from typing import Sequence
from monai.config.type_definitions import DtypeLike, KeysCollection, PathLike
from fastcore.basics import listify, store_attr, warnings
from monai.data.image_reader import ImageReader
from monai.utils.misc import ensure_tuple, ensure_tuple_rep
from torchvision.utils import Any
from fran.transforms.totensor import ToTensorT
from fran.utils.fileio import load_dict


import warnings
from collections.abc import Sequence
from pathlib import Path
from pydoc import locate
from typing import Callable

import numpy as np
import torch

from monai.config import DtypeLike,  PathLike
from monai.data.image_reader import (
    ImageReader,
    ITKReader,
    NibabelReader,
    NrrdReader,
    NumpyReader,
    PILReader,
    PydicomReader,
)
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import is_no_channel
from monai.transforms.transform import Transform
from monai.transforms.utility.array import EnsureChannelFirst
from monai.utils import GridSamplePadMode
from monai.utils import ImageMetaKey as Key
from monai.utils import OptionalImportError, convert_to_dst_type, ensure_tuple, look_up_option, optional_import

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
from fran.utils.helpers import folder_name_from_list
from fran.utils.imageviewers import ImageMaskViewer
import ipdb
tr = ipdb.set_trace



class SITKReaderFast(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False,  reverse_indexing: bool= False,ensure_channel_first: bool= False,affine_lps_to_ras=True,
        channel_dim: str | int | None = None,
        simple_keys: bool = False,
        pattern=None,
        sep: str = ".",
        image_only: bool = False,
                 ) -> None:
        super().__init__(keys, allow_missing_keys)
        store_attr('reverse_indexing,ensure_channel_first,affine_lps_to_ras,channel_dim,simple_keys,pattern,sep,image_only')


    def _get_meta_dict(self, img) -> dict:
        """
        Get all the metadata of the image and convert to dict type.

        Args:
            img: an ITK image object loaded from an image file.

        """
        meta_dict  = {}
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
        if len(direction)!=9:
            raise NotImplemented
        else:
            direction = np.array(direction)
            direction = direction.reshape(3,3)

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
        # img_np = sitk.GetArrayFromImage(img)
        # 
        # if img.GetNumberOfComponentsPerPixel() == 1:  # handling spatial images
        #     img_np = img_np if self.reverse_indexing else img_np.T
        # # handling multi-channel images
        # img_np = img_np if self.reverse_indexing else np.moveaxis(img_np.T, 0, -1)
        # # return img_np
        #
        #
        np_img = sitk.GetArrayFromImage(img)
        if img.GetNumberOfComponentsPerPixel() == 1:  # handling spatial images
            return np_img if self.reverse_indexing else np_img.T
        # handling multi-channel images
        return np_img if self.reverse_indexing else np.moveaxis(np_img.T, 0, -1)

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        """
        Args:
            img: an SITK image object loaded from an image file or a list of ITK image objects.

        """

        data = self._get_array_data(img)
        header = self._get_meta_dict(img)
        header[MetaKeys.ORIGINAL_AFFINE] = self._get_affine(img, self.affine_lps_to_ras)
        header[MetaKeys.SPACE] = SpaceKeys.RAS if self.affine_lps_to_ras else SpaceKeys.LPS
        header[MetaKeys.AFFINE] = header[MetaKeys.ORIGINAL_AFFINE].copy()
        header[MetaKeys.SPATIAL_SHAPE] = self._get_spatial_shape(img)
        if self.channel_dim is None:  # default to "no_channel" or -1
            header[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
                float("nan") if len(data.shape) == len(header[MetaKeys.SPATIAL_SHAPE]) else -1
            )
        else:
            header[MetaKeys.ORIGINAL_CHANNEL_DIM] = self.channel_dim

        return data,header



    def func(self,fname):
        img= sitk.ReadImage(fname)

        array_np = self._get_array_data(img)

        array_np,  meta_data = self.get_data(img)
        meta_data[Key.FILENAME_OR_OBJ] =str(fname)
        array_pt = ToTensorT()(array_np)

        img = MetaTensor.ensure_torch_and_prune_meta(
                array_pt, meta_data, self.simple_keys, pattern=self.pattern, sep=self.sep
        )
        if self.ensure_channel_first:
            img = EnsureChannelFirst()(img)
        return img

    def __call__(self, data ):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
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
        if not isinstance(img,Union[list,tuple]): 
            img = [img]

        for i in img: # ensure_tuple freezes with sitk images
            data = self._get_array_data(i)
            img_array.append(data)
            header = self._get_meta_dict(i)
            header[MetaKeys.ORIGINAL_AFFINE] = self._get_affine(i, self.affine_lps_to_ras)
            header[MetaKeys.SPACE] = SpaceKeys.RAS if self.affine_lps_to_ras else SpaceKeys.LPS
            header[MetaKeys.AFFINE] = header[MetaKeys.ORIGINAL_AFFINE].copy()
            header[MetaKeys.SPATIAL_SHAPE] = self._get_spatial_shape(i)
            if self.channel_dim is None:  # default to "no_channel" or -1
                header[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
                    float("nan") if len(data.shape) == len(header[MetaKeys.SPATIAL_SHAPE]) else -1
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
        meta_dict  = {}
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
        if len(direction)!=9:
            raise NotImplemented
        else:
            direction = np.array(direction)
            direction = direction.reshape(3,3)

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


class LoadSITK(Transform):


    def __init__(
        self,
        image_only: bool = True,
        dtype: DtypeLike | None = np.float32,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: str | None = None,
        prune_meta_sep: str = ".",
        expanduser: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            image_only: if True return only the image MetaTensor, otherwise return image and header dict.
            dtype: if not None convert the loaded image to this data type.
            ensure_channel_first: if `True` and loaded both image array and metadata, automatically convert
                the image array shape to `channel first`. default to `False`.
            simple_keys: whether to remove redundant metadata keys, default to False for backward compatibility.
            prune_meta_pattern: combined with `prune_meta_sep`, a regular expression used to match and prune keys
                in the metadata (nested dictionary), default to None, no key deletion.
            prune_meta_sep: combined with `prune_meta_pattern`, used to match and prune keys
                in the metadata (nested dictionary). default is ".", see also :py:class:`monai.transforms.DeleteItemsd`.
                e.g. ``prune_meta_pattern=".*_code$", prune_meta_sep=" "`` removes meta keys that ends with ``"_code"``.
            expanduser: if True cast filename to Path and call .expanduser on it, otherwise keep filename as is.
            args: additional parameters for reader if providing a reader name.
            kwargs: additional parameters for reader if providing a reader name.

        """

        self.auto_select = False
        self.image_only = image_only
        self.dtype = dtype
        self.ensure_channel_first = ensure_channel_first
        self.simple_keys = simple_keys
        self.pattern = prune_meta_pattern
        self.sep = prune_meta_sep
        self.expanduser = expanduser
        self.reader = SITKReader()
    def __call__(self, filename: Sequence[PathLike] | PathLike):
        """

        Args:
            filename: path file or file-like object or a list of files.
                will save the filename to meta_data with key `filename_or_obj`.
                if provided a list of files, use the filename of first file to save,
                and will stack them together as multi-channels data.
                if provided directory path instead of file path, will treat it as
                DICOM images series and read.
            reader: runtime reader to load image file and metadata.

        """
        filename = tuple(
            f"{Path(s).expanduser()}" if self.expanduser else s for s in ensure_tuple(filename)  # allow Path objects
        )
        img, err = None, []
        img =self.reader.read(data= filename)
        if img is None :
            if isinstance(filename, tuple) and len(filename) == 1:
                filename = filename[0]
            msg = "\n".join([f"{e}" for e in err])
            raise RuntimeError(
                f"{self.__class__.__name__} cannot find a suitable reader for file: {filename}.\n"
                "    Please install the reader libraries, see also the installation instructions:\n"
                "    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies.\n"
                f"   The current registered: {self.readers}.\n{msg}"
            )

        img_array, meta_data = self.reader.get_data(img)

        if not isinstance(meta_data, dict):
            raise ValueError(f"`meta_data` must be a dict, got type {type(meta_data)}.")
        # make sure all elements in metadata are little endian
        meta_data = switch_endianness(meta_data, "<")

        meta_data[Key.FILENAME_OR_OBJ] = f"{ensure_tuple(filename)[0]}"  # Path obj should be strings for data loader
        img = MetaTensor.ensure_torch_and_prune_meta(
            img_array, meta_data, self.simple_keys, pattern=self.pattern, sep=self.sep
        )
        if self.ensure_channel_first:
            img = EnsureChannelFirst()(img)
        if self.image_only:
            return img
        return img, img.meta if isinstance(img, MetaTensor) else meta_data


class LoadSITKd(LoadImaged):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LoadImage`,
    It can load both image data and metadata. When loading a list of files in one key,
    the arrays will be stacked and a new dimension will be added as the first dimension
    In this case, the metadata of the first image will be used to represent the stacked result.
    The affine transform of all the stacked images should be same.
    The output metadata field will be created as ``meta_keys`` or ``key_{meta_key_postfix}``.

    If reader is not specified, this class automatically chooses readers
    based on the supported suffixes and in the following order:

        - User-specified reader at runtime when calling this loader.
        - User-specified reader in the constructor of `LoadImage`.
        - Readers from the last to the first in the registered list.
        - Current default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader),
          (npz, npy -> NumpyReader), (dcm, DICOM series and others -> ITKReader).

    Please note that for png, jpg, bmp, and other 2D formats, readers by default swap axis 0 and 1 after
    loading the array with ``reverse_indexing`` set to ``True`` because the spatial axes definition
    for non-medical specific file formats is different from other common medical packages.

    Note:

        - If `reader` is specified, the loader will attempt to use the specified readers and the default supported
          readers. This might introduce overheads when handling the exceptions of trying the incompatible loaders.
          In this case, it is therefore recommended setting the most appropriate reader as
          the last item of the `reader` parameter.

    See also:

        - tutorial: https://github.com/Project-MONAI/tutorials/blob/master/modules/load_medical_images.ipynb


    """

    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = np.float32,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = True,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: str | None = None,
        prune_meta_sep: str = ".",
        allow_missing_keys: bool = False,
        expanduser: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadSITK(
            image_only = image_only,
            dtype = dtype,
            ensure_channel_first=ensure_channel_first,
            simple_keys= simple_keys,
            prune_meta_pattern= prune_meta_pattern,
            prune_meta_sep= prune_meta_sep,
            expanduser=expanduser,
            *args,
            **kwargs,
        )
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError(
                f"meta_keys should have the same length as keys, got {len(self.keys)} and {len(self.meta_keys)}."
            )
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def __call__(self, data):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key])
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError(
                        f"loader must return a tuple or list (because image_only=False was used), got {type(data)}."
                    )
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError(f"metadata must be a dict, got {type(data[1])}.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        return d




class Reader(ImageReader):
    def __init__(self,channel_dim=None,**kwargs) -> None:
        self.kwargs=  kwargs
        self.channel_dim = float("nan") if channel_dim == "no_channel" else channel_dim
        super().__init__()

    def read_func(self,fn,**kwargs):
        '''
        return torch tensor
        '''
        pass
      
    def read(self, data , **kwargs):
        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        img_ = []
        for fn in filenames:
            fn = f"{fn}"
            img_.append(self.read_func(fn, **kwargs_))
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> tuple[torch.Tensor, dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is torch.Tensor of image data, second is dict of metadata.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the metadata of the first image is used to represent the output metadata.

        Args:
            img: a torch.Tensor loaded from a file or a list of torch.Tensors.

        """

        img_array: list[torch.Tensor] = []
        compatible_meta: dict = {}
        if isinstance(img, torch.Tensor):
            img = (img,)

        for i in ensure_tuple(img):
            header: dict[MetaKeys, Any] = {}
            if isinstance(i, torch.Tensor):
                # if `channel_dim` is None, can not detect the channel dim, use all the dims as spatial_shape
                spatial_shape = np.asarray(i.shape)
                if isinstance(self.channel_dim, int):
                    spatial_shape = np.delete(spatial_shape, self.channel_dim)
                header[MetaKeys.SPATIAL_SHAPE] = spatial_shape
                header[MetaKeys.SPACE] = SpaceKeys.RAS
            img_array.append(i)
            header[MetaKeys.ORIGINAL_CHANNEL_DIM] = (
                self.channel_dim if isinstance(self.channel_dim, int) else float("nan")
            )
            
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def verify_suffix(self, filename) -> bool:
        suffixes = ['pt']
        return is_supported_format(filename,suffixes)

class TorchReader(Reader):
    def read_func(self,fn,**kwargs):
        return torch.load(fn)


# %%
if __name__ == "__main__":

    import torch
    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *

    project_title = "nodes"
    proj = Project(project_title=project_title)

    configuration_filename="/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    configuration_filename=None

    configs = ConfigMaker(
        proj,
        raytune=False,
        configuration_filename=configuration_filename
    ).config

    global_props = load_dict(proj.global_properties_filename)
# %%
    fn = "/s/xnat_shadow/crc/images/crc_CRC261_20170322_AbdoPelvis1p5.nii.gz"
    dici = {'image':fn}

    L = LoadImaged(keys= ['image'])
    img = L(dici)

# %%
    dici = {'image':fn}
    from time import time
    L1 = LoadSITKd(keys=['image'])
    L = LoadImaged(keys= ['image'])
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

 
    L2 = LoadImaged(keys=['image'], reader=SITKReader)
    img2 = L2(dici)

    img2['image'].meta
# %%
    L3 = SITKReaderFast(keys=['image'])
    im3 = L3(dici)

# %%
    ImageMaskViewer([img2['image'],im3['image']],data_types=['image','image'])
# %%
    fname = fn
    img = sitk.ReadImage(fname)
# %%

