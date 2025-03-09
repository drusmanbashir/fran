
from monai.utils import ImageMetaKey as Key
from monai.data.meta_obj import get_track_meta
from typing import List, Optional, Sequence, Union
from monai.transforms.inverse import TraceableTransform
from monai.transforms.transform import Randomizable
from monai.transforms.utils import generate_label_classes_crop_centers, map_classes_to_indices
from monai.utils.misc import fall_back_tuple
import numpy as np
from fastcore.all import ItemTransform, delegates
from monai.transforms.croppad.array import *
import ipdb
from typing import List

from torch.functional import Tensor
tr = ipdb.set_trace
# %%

class RandCropImgMaskByLabelClasses(Randomizable, TraceableTransform):
    backend = SpatialCrop.backend

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        ratios: Optional[List[Union[float, int]]] = None,
        label: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
        num_samples: int = 1,
        image: Optional[torch.Tensor] = None,
        image_threshold: float = 0.0,
        indices: Optional[List[Tensor]] = None,
        allow_smaller: bool = False,
    ) -> None:
        self.spatial_size = spatial_size
        self.ratios = ratios
        self.label = label
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[int]]] = None
        self.indices = indices
        self.allow_smaller = allow_smaller

    def randomize(
        self, label: torch.Tensor, indices: Optional[List[Tensor]] = None, image: Optional[torch.Tensor] = None
    ) -> None:
        indices_: Sequence[Tensor]
        if indices is None:
            if self.indices is not None:
                indices_ = self.indices
            else:
                indices_ = map_classes_to_indices(label, self.num_classes, image, self.image_threshold)
        else:
            indices_ = indices
        self.centers = generate_label_classes_crop_centers(
            self.spatial_size, self.num_samples, label.shape[1:], indices_, self.ratios, self.R, self.allow_smaller
        )

    def __call__(
        self,
        img: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        indices: Optional[List[Tensor]] = None,
        randomize: bool = True,
    ) -> List[torch.Tensor]:
        """
        Args:
            img: input data to crop samples from based on the ratios of every class, assumes `img` is a
                channel-first array.
            label: the label image that is used for finding indices of every class, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``image > image_threshold`` to select the centers only in valid region. if None, use `self.image`.
            indices: list of indices for every class in the image, used to randomly select crop centers.
            randomize: whether to execute the random operations, default to `True`.

        """
        if label is None:
            label = self.label
        if label is None:
            raise ValueError("label should be provided.")
        if image is None:
            image = self.image

        mask = label.clone()
        if randomize:
            self.randomize(label, indices, image)
        results: List[torch.Tensor] = []
        orig_size = img.shape[1:]
        if self.centers is not None:
            for i, center in enumerate(self.centers):
                roi_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
                cropped_img = SpatialCrop(roi_center=tuple(center), roi_size=roi_size)(img)
                cropped_mask= SpatialCrop(roi_center=tuple(center), roi_size=roi_size)(mask)
                if get_track_meta():
                    ret_: MetaTensor = cropped_img  # type: ignore
                    ret_.meta[Key.PATCH_INDEX] = i
                    ret_.meta["crop_center"] = center
                    self.push_transform(ret_, orig_size=orig_size, extra_info=self.pop_transform(ret_, check=False))
                results.append([cropped_img,cropped_mask])
        imgs, masks = zip(*results)
        return imgs,masks

# %%
    def infer_num_samples(spatial_size, ds_median_shape, oversampling_factor):
            spatial_dim2 = spatial_size[2] * (1-oversampling_factor)
            ds_median_dim2 = ds_median_shape[2]
            num_samples =  int(np.ceil(ds_median_dim2/spatial_dim2))
            print("Each image will be sampled {} times per epoch based on dataset shape / patch_size / overlap combo.".format(num_samples))
            return num_samples


class RandomCropped(ItemTransform):
        @delegates(RandCropImgMaskByLabelClasses)
        def __init__(self,spatial_size,ratios,num_classes,ds_median_shape=None,oversampling_factor=0.5 , num_samples=None,*args,**kwargs):
            assert num_samples is not None or ds_median_shape is not None, "Provide either num_samples or ds_median_shape to infer samples to randomize per case"
            if not num_samples:
                num_samples = self.infer_num_samples(spatial_size,ds_median_shape,oversampling_factor)
            self.cropper = RandCropImgMaskByLabelClasses(spatial_size=spatial_size,ratios=ratios,num_classes=num_classes,num_samples=num_samples,*args,**kwargs)
        def encodes(self,x):
            imgs,masks= self.cropper(img=x[0],label=x[1])
            shapes = [im.shape[1:] for im in imgs]
            if any([list(s)!=[128,128,128] for s in shapes]):
                tr()
            return torch.stack(imgs),torch.stack(masks)
        def collate_fn(self,x):
            imgs=[]
            masks=[]
            for img,mask in x:
                imgs.append(img)
                masks.append(mask)
                
            return torch.cat(imgs,0), torch.cat(masks,0)
        def infer_num_samples(self,spatial_size, ds_median_shape, oversampling_factor):
            spatial_dim2 = spatial_size[2] * (1-oversampling_factor)
            ds_median_dim2 = ds_median_shape[2]
            num_samples =  int(np.ceil(ds_median_dim2/spatial_dim2))
            print("Each image will be sampled {} times per epoch based on dataset shape / patch_size / overlap combo.".format(num_samples))
            return num_samples

