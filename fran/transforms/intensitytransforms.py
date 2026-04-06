# %%
from functools import wraps
from typing import Hashable, Mapping

import numpy as np
import torch
from fran.transforms.base import (
    ItemTransform,
    MonaiDictTransform,
    Transform,
)
from monai.config.type_definitions import DtypeLike, NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms import MapTransform, RandomizableTransform
from monai.transforms.intensity.array import RandGaussianNoise
from monai.utils.type_conversion import convert_to_tensor
from scipy.ndimage.filters import gaussian_filter
from torch.functional import Tensor


class NormaliseClip(Transform):
    def __init__(self, clip_range, mean, std):
        # super().__init__(keys, allow_missing_keys)

        self.clip_range = clip_range
        self.mean = mean
        self.std = std

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


class RandRandGaussianNoised(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys,
        std_limits,
        prob: float = 1,
        do_transform: bool = True,
        dtype: DtypeLike = np.float32,
    ):
        MapTransform.__init__(self, keys, False)
        RandomizableTransform.__init__(self, prob)
        self.std_limits = std_limits
        self.dtype = dtype

    def randomize(self):
        super().randomize(None)
        rand_std = self.R.uniform(low=self.std_limits[0], high=self.std_limits[1])
        self.rand_gaussian_noise = RandGaussianNoise(
            mean=0, std=rand_std, prob=1.0, dtype=self.dtype
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random noise
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.rand_gaussian_noise.randomize(d[first_key])
        for key in self.key_iterator(d):
            d[key] = self.rand_gaussian_noise(img=d[key], randomize=False)
        return d


class MakeBinary(MonaiDictTransform):
    def func(self, x):
        x[x > 0] = 1
        return x


def zero_to_one(func):
    @wraps(func)
    def _inner(img, *args, **kwargs):
        min = img.min()
        range = img.max() - min
        if min < 0:
            img = img - min
        img = img / (range + 1e-5)
        img = func(img, *args, **kwargs)
        return img

    return _inner


def standardize(img, mn, std):
    return (img - mn) / std

# %%
if __name__ == "__main__":
    from fran.data.dataset import *
    from fran.transforms.spatialtransforms import *
    from fran.utils.common import *
    from matplotlib import pyplot as plt
    from utilz.fileio import *
    from utilz.helpers import *
    from utilz.imageviewers import *

    P = Project(project_title="litsmc")
    proj_defaults = P
    # %%

    import torchvision

    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.eval()
    # %%
    fn = Path(
        "/s/fran_storage/datasets/preprocessed/fixed_spacing/litsmc/spc_080_080_150/images/drli_024.pt"
    )
    im = torch.load(fn)

    im1 = im.mean(0)
    im1 = im.mean(2)
    im2 = im.mean(1)
    plt.imshow(im2)

    # %%
    x = torch.tensor([1, 2, 3])
    x.repeat(4, 2)
    x.repeat(4, 2, 1).size()
    # %%
    im1 = im1.repeat(3, 1, 1)

    confidence_threshold = 0.8
    pred = model([im1])
    bbox, scores, labels = pred[0]["boxes"], pred[0]["scores"], pred[0]["labels"]
    indices = torch.nonzero(scores > confidence_threshold).squeeze(1)

    filtered_bbox = bbox[indices]
    filtered_scores = scores[indices]
    filtered_labels = labels[indices]

    # %%
    import cv2

    def draw_boxes_and_labels(image, bbox, labels):
        img_copy = image.copy()

        for i in range(len(bbox)):
            x, y, w, h = bbox[i].astype("int")
            cv2.rectangle(img_copy, (x, y), (w, h), (0, 0, 255), 5)

            class_index = labels[i].numpy().astype("int")
            # class_detected = class_names[class_index - 1]

            class_index = str(class_index)
            cv2.putText(
                img_copy,
                class_index,
                (x, y + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return img_copy

    # %%

    plt.imshow(img)
    im1 = im1.detach().cpu()
    img = np.array(im1)
    img = np.transpose(img, (1, 2, 0))
    cv2_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = bbox.detach().cpu().numpy()
    result_img = draw_boxes_and_labels(cv2_image, bbox, labels)
    cv2.imshow("image", result_img)
    # %%
    mask_fn = Path("/home/ub/datasets/preprocessed/kits23/masks/kits23_00088.npy")
    img_fn = Path("/home/ub/datasets/preprocessed/kits23/images/kits23_00088.npy")
    bb = load_dict(bboxes_21)
    bboxes = [b for b in bb if b["filename"] == mask_fn][0]
    patch_size = [64, 256, 256]
    # %%
    x, y, _ = train_ds[0]
    # %%
    xx, yy = invert([x, y], factor_range=[0.6, 1.2])
    xx, yy = contrast([x, y], factor_range=[0, 1])
    plt.hist(x.flatten())
    plt.hist(xx.flatten())
    # %%
    b = partial(brightness, factor_range=[1.2, 1.3])
    c = partial(brightness, factor_range=[1.2, 1.3])
    xx, yy = b([x, y])


# %%
#
#     A = AffineTrainingTransform3D(0.99,pi/8)
#     a,b = A.encodes([xx,yy])
#     C = CropBatch(patch_size)
#     aa,bb= C.encodes([a,b])
#     # ImageMaskViewer([a[n,0],b[n,0]])
#
# %%
#     xxx,yyy = P2([xx,yy])
# %%
#     n=0
#     ImageMaskViewer([xxx[n,0],yyy[n,0]])
#     ImageMaskViewer([x,y])
# %%
#     xl,_ = power_transform([x,y],scale=1)
#     ImageMaskViewer([x,xl],cmap_mask="Greys_r")
#     xx, yy = T.encodes([x,y])
#
#     ImageMaskViewer([x,xx],cmap_mask="Greys_r")

# %%
