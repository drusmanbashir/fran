
# %%
from typing import Union
from fastai.vision.augment import typedispatch
import numpy as np
from torch.functional import Tensor
import torchio as tio
import ipdb

from fran.transforms.basetransforms import KeepBBoxTransform

from fran.inference.helpers import get_amount_to_pad
tr = ipdb.set_trace

import torch
from torch.nn import functional as F
from fastcore.basics import store_attr
from fastcore.transform import ItemTransform, Transform
from fran.inference.inference_base import get_scale_factor_from_spacings, rescale_bbox
from fran.transforms.spatialtransforms import slices_from_lists

from fran.utils.helpers import multiply_lists

class BBoxesToLists(Transform):

    def encodes(self,bounding_boxes):
        slices_as_num=[]
        for bb in bounding_boxes:
            cc = [[a.start,a.stop] for a in bb]
            slices_as_num.append(cc)
        return slices_as_num
    def decodes(self,bboxes):
        bboxes_out=[]
        for bb in bboxes:
            slices=[]
            for b in bb:
                slc = slice(b[0],b[1])
                slices.append(slc)
            bboxes_out.append(slices)
        return bboxes_out

class BBoxesToPatchSize(ItemTransform):
    def __init__(self, patch_size,sz_dest,expand_bbox):
        store_attr()
    def encodes(self,x):
        img,bboxes = x
        stride=[1,1,1]
        bboxes_out=[]
        for bbox in bboxes:
            bbox_size = [a.stop - a.start for a in bbox]
            bbox_size = [int(np.round(a+a*self.expand_bbox)) for a in bbox_size]
            input_shape = self.sz_dest
            # input_shape, target_patch_size,centroid = [multiply_lists(arr, stride) for arr in [input_shape,target_patch_size,centroid]]
            centroid = [np.floor((a.stop+a.start)/2) for a in bbox]
            target_patch_size= multiply_lists(self.patch_size,stride)
            target_patch_size = [int(np.ceil(np.maximum(a,b)/2)*2) for a,b in zip(target_patch_size,bbox_size)] # turns to even number
            patch_halved = [x / 2 for x in target_patch_size]
            center_moved = np.maximum(centroid,patch_halved)
            slc_start = np.floor(np.maximum(0,centroid - np.floor(patch_halved).astype(np.int32)))
            # slc_stop = np.floor(np.minimum(input_shape,center_moved + (np.ceil(patch_halved)).astype(np.int32)))
            slc_stop = center_moved + np.ceil(patch_halved).astype(np.int32)
            shift_back = np.minimum(0, input_shape - slc_stop)
            shift_forward = np.minimum(0, slc_start)
            shift_final = shift_back - shift_forward
            slc_start, slc_stop = slc_start + shift_final, slc_stop + shift_final
            slices = tuple(slices_from_lists(slc_start, slc_stop, stride))
            bboxes_out.append(slices)
        return img,bboxes_out


class ResampleToStage0(ItemTransform):
    # resize entire image to patch_size
    def __init__(self, img_sitk, resample_spacing,mode='trilinear'):
        store_attr()

        self.sz_source , spacing_source = img_sitk.GetSize(), img_sitk.GetSpacing()

        self.scale_factor = [a / b for a, b in zip(spacing_source, resample_spacing)]
        self.sz_dest,_ = get_scale_factor_from_spacings(self.sz_source,spacing_source,resample_spacing)
    def encodes (self,x):
        img,bboxes = x
        bboxes_out=[]
        for bbox in bboxes:
            bbox = rescale_bbox(self.scale_factor,bbox)
            bboxes_out.append(bbox)

        is_numpy = isinstance(img,np.ndarray)
        img = torch.tensor(img,dtype=torch.float32)
        img= img.unsqueeze(0).unsqueeze(0)
        img = F.interpolate(img,self.sz_dest,mode=self.mode)
        img= img.squeeze(0).squeeze(0)
        if is_numpy==True:
            img = img.numpy()
        x = img,bboxes_out
        return x

class Backsample(ItemTransform):
        def __init__(self,img_sitk):
            self.sz_source  = img_sitk.GetSize()
        def encodes(self,x):
            x_out=[]
            modes  = ['trilinear','nearest']
            for xx , mode in zip(x,modes):
                xx= xx.unsqueeze(0).unsqueeze(0)
                xx = F.interpolate(xx,self.sz_source,mode=mode)
                xx= xx.squeeze(0).squeeze(0)
                x_out.append(xx)
            return x_out
class BacksampleMask(Transform):
        def __init__(self,img_sitk):
            self.sz_source  = img_sitk.GetSize()
        def encodes(self,x):
                x= x.unsqueeze(0).unsqueeze(0)
                x = F.interpolate(x,self.sz_source,mode='nearest')
                x= x.squeeze(0).squeeze(0)
                return x

class CreateDataLoaderAggregator(ItemTransform):
        def __init__(self,patch_size,patch_overlap,grid_mode,batch_size):
            store_attr()
        def encodes(self,x):
            img,bboxes = x
            dls=[]
            for bbox in bboxes:
                img_cropped = img[bbox]
                img_tio= tio.ScalarImage(tensor=np.expand_dims(img_cropped,0))
                subject = tio.Subject(image=img_tio)
                grid_sampler = tio.GridSampler(subject=subject,patch_size=self.patch_size,patch_overlap=self.patch_overlap) 
                aggregator = tio.inference.GridAggregator(grid_sampler,overlap_mode=self.grid_mode) 
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size)
                dls.append([patch_loader,aggregator])
            return dls

# %%
class PadNpArray(KeepBBoxTransform):

        def __init__(self, patch_size, mode='constant'):
            store_attr()
            super().__init__()
        def func(self, img:np.ndarray)->np.ndarray:
            patch_size_vs_img_size = [x > y for x, y in zip(self.patch_size, img.shape)]
            if any(patch_size_vs_img_size):  # check if any dim of image is smaller than patch_size
                self.padding = get_amount_to_pad(img.shape, self.patch_size)
                img= np.pad(img, self.padding, self.mode)
            return img
        def decodes(self,img):
            '''
            note: this will crash if img.dim()>3
            :param img:
            :return:
            '''
            if hasattr(self,'padding'):
                s = [slice(p[0],s-p[1]) for p,s in zip(self.padding,img.shape)]
                img = img[tuple(s)]
            return img

# %%
        # dest = [200,300,400]
        # x  = np.random.rand(50,100,200)
        # R = ResizeNP(dest)
        # P = PadNpArray(patch_size=[160,160,160])
        # y = P.encodes([x])
        # R.encodes([x])
# %%
class Resize(KeepBBoxTransform):
    '''
    Strictly an inference transform. Requires img and bbox
    '''
    
    def __init__(self,dest_size,mode='trilinear'):
        store_attr()
    def func(self,img):
        self.org_size = img.shape
        img = torch.tensor(img,dtype=torch.float32)
        img= img.unsqueeze(0).unsqueeze(0)
        img = F.interpolate(img,self.dest_size,mode=self.mode)
        img= img.squeeze(0).squeeze(0)
        img = img.numpy()
        return img
    def decodes(self,img):
        if isinstance(img,Union[list,tuple]):
            img, bboxes=img
            has_bbox=True
        else: has_bbox=False
        img = torch.tensor(img)
        mode = 'nearest' if 'int' in str(img.dtype) else 'trilinear'
        img= img.unsqueeze(0).unsqueeze(0)
        img = F.interpolate(img,self.org_size,mode=mode)
        img= img.squeeze(0).squeeze(0)
        img = img.numpy()
        if has_bbox==True:
            return img, bboxes
        else: return img
# %%

class ApplyBBox(ItemTransform):
    '''
    param bbox: 3-tuple of slices
    param x: input image of 3 or greater dims. BBox is repeated over every extra dim (e.g., channel or batch dims)
    '''
    
    def __init__(self,bboxes):
        self.bboxes = tuple(bboxes)
    def encodes(self, x):
            self.org_size = x.shape
            x_out = [x[bbox] for bbox in self.bboxes]
            return x_out

    def decodes(self,x):
            bbox = self.bboxes[0] # havent figured out a multi-bbox version
            img,anything = x
            self.bbox_equate_dims(img)
            output_img = torch.zeros(self.org_size)
            output_img[bbox]= img
            return output_img,anything

    def bbox_equate_dims(self,img):
        first_dims = img.dim()- len(self.bbox)
        first_dims_output_img = img.dim() - len(self.org_size)
        if first_dims >0:
            sizes = list(img.shape[:first_dims])
            slices_added = [slice(0,end) for end in sizes]
            self.bbox =tuple(slices_added)+self.bbox
        if first_dims_output_img>0:
            sizes = list(img.shape[:first_dims])
            self.org_size = sizes+self.org_size

# A = ApplyBBox([400,400,400],self.bboxes_transformed[0])
# y = A.encodes([x,self.bboxes_transformed])
class AddBatchChannelDims(ItemTransform):
    def __init__(self,preserve_dims_on_decode=[0,1]): # (batch,channel)
        self.preserve_dims_on_decode = preserve_dims_on_decode
        super().__init__()
    def encodes(self,x):
        img,bboxes = x
        img = img.unsqueeze(0).unsqueeze(0)
        return img,bboxes
    def decodes(self,img):
        img = img[self.preserve_dims_on_decode[0],self.preserve_dims_on_decode[1]]
        return img




    
        
# %%
class TransposeSITKToNp(ItemTransform):
    '''
    typedispatched class which handles arrays +/- bboxes. Bboxes, if present, MUST be in a list even if len-1
    '''

    def decodes(self,x): return self.encodes(x)

@TransposeSITKToNp
def encodes(self,x:Union[list,tuple]):
    img,bboxes  = x
    img= img.transpose(2,1,0)# not training is done on images facing up, and inference has to do the same.
    bboxes_out=[]
    for bbox in bboxes:
        bbox=bbox[2],bbox[1],bbox[0]
        bboxes_out.append(bbox)
    return img,bboxes_out

@TransposeSITKToNp
def encodes(self,img:np.ndarray):
    img= img.transpose(2,1,0)# not training is done on images facing up, and inference has to do the same.
    return img

@TransposeSITKToNp
def encodes(self,img:Tensor):
    img= img.permute(2,1,0)# not training is done on images facing up, and inference has to do the same.
    return img

# %%
    # bb = [slice(0,10),slice(0,20),slice(0,30)]
    # x = np.random.rand(50,100,200)
    #
    # T = TransposeSITKToNp()
    # a = T([x,[bb]])
    # b = T.decodes(a)
# %%
class Resample(Transform):
    def __init__(self,org_size,org_spacing,dest_spacing,order=None):
        self.sz_dest, self.scale_factor = get_scale_factor_from_spacings(org_size,org_spacing,dest_spacing)
        store_attr()

    def encodes(self,x):
            x =  resize(x, self.sz_dest,order=self.order)
            return x

    def decodes(self,x):
        x = x.unsqueeze(0).unsqueeze(0)
        x = F.interpolate(input= x,size=self.org_size)
        x = x.squeeze(0).squeeze(0)
        return x

class Stride(Transform):
    def __init__(self,stride=[1,1,1]):
        store_attr()
    def encodes(self,x):
        if self.stride != [1,1,1]:
            self.prestride_shape= x.shape
            x =  x[::self.stride[0],::self.stride[1],::self.stride[2]]
        return x
    def decodes(self,x):
        if self.stride !=[1,1,1]:
            x = x.unsqueeze(0).unsqueeze(0)
            x = F.interpolate(input= x,size=self.prestride_shape)
            x = x.squeeze(0).squeeze(0)
        return x


class ToTensorBBoxes(ItemTransform):

    def __init__(self, img_dtype=torch.float, mask_dtype=torch.uint8):
        store_attr()

    def encodes(self, x):
        img, bboxes= x

        img = torch.tensor(img.copy(), dtype=self.img_dtype)
        return img, bboxes
    def decodes(self,img):
        return img.detach().cpu().numpy()
        


