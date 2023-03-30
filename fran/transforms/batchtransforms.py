
import ipdb
tr = ipdb.set_trace

import torch.nn.functional as F
from fran.transforms.basetransforms import *
# class AffineTrainingTransform3D(ItemTransform):
#     '''
#     to-do: verify if nearestneighbour method preserves multiple mask labels
#     '''
#     
#
#     def __init__(self, p=0.5, rotate_max=pi / 6, translate=torch.tensor([0, 0, 0]), rescale_max=0.2):
#         store_attr()
#
#     def encodes(self, x):
#         img, mask = x
#         if np.random.rand() < self.p:
#             grid = get_affine_grid(img.shape,
#                                    shear=True,
#                                    rescale_max=self.rescale_max,
#                                    rotate_max=self.rotate_max,
#                                    translate=self.translate,
#                                    device=img.device).type(img.dtype)
#             img = F.grid_sample(img, grid)
#             mask = F.grid_sample(mask.type(img.dtype), grid, mode='nearest')
#             return img, mask.to(torch.uint8)
#         return img, mask
#


class CropBatch(ItemTransform):

    def __init__(self, patch_size):
        self.dim = len(patch_size)
        self.patch_halved = [int(x / 2) for x in patch_size]

    def encodes(self, x):
        img, mask = x
        center = [int(x / 2) for x in img.shape[-self.dim:]]
        slices = [
            slice(None),
        ] * 2  # batch and channel dims
        for ind in range(self.dim):
            slc = slice(center[ind] - self.patch_halved[ind], center[ind] + self.patch_halved[ind])
            slices.append(slc)
        img, mask = img[slices], mask[slices]
        return img, mask

class ResizeBatch(ItemTransform):
    def __init__(self,target_size):
        self.target_size=target_size
    def encodes(self,x):
        img,mask=x
        if list(img.shape[2:]) !=self.target_size:
            img = F.interpolate(img,size=self.target_size,mode='trilinear')
            mask = F.interpolate(mask,size=self.target_size,mode='nearest')
        return img,mask


