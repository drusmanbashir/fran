
# %%
from typing import Union
import numpy as np
from torch.functional import Tensor
import ipdb

from fran.transforms.basetransforms import KeepBBoxTransform
from fran.utils.sitk_utils import *

from fran.inference.helpers import get_amount_to_pad, get_scale_factor_from_spacings, rescale_bbox

tr = ipdb.set_trace

import torch
from torch.nn import functional as F
from fastcore.basics import store_attr
from fastcore.transform import ItemTransform, Transform
# from fran.inference.inference_base import get_scale_factor_from_spacings, rescale_bbox
from fran.transforms.spatialtransforms import MaskLabelRemap, slices_from_lists

from fran.utils.helpers import multiply_lists
from fastcore.test import is_close
from fastcore.all import GetAttr
class PredictorTransform(ItemTransform,GetAttr):
    _default = 'predictor'
    def __repr__(self): return type(self).__name__

orientations = {
    'LAS': (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
    'LPS': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    'PRS': (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
}

def reorient_sitk(img:sitk.Image,direction:tuple)->sitk.Image:
    try:
        orn = [key for key,val in orientations.items() if is_close(val ,direction,eps=1e-5)][0]
    except:
        tr()
    return sitk.DICOMOrient (img, orn)
class ArrayToSITKF(Transform):

    def __init__(self,sitk_props=None,img_sitk=None): 
        assert any([sitk_props,img_sitk]), "Either provide sitk_properties or a sitk_image to serve as templt"
        if not sitk_props: 
            sitk_props = img_sitk.GetOrigin(),img_sitk.GetSpacing(),img_sitk.GetDirection()
        self.sitk_props= sitk_props
    def encodes(self,pred_pt)->list:
            assert pred_pt.ndim==4, "This requires 4d array, NxDxWxH"
            preds_out = []
            for pred in pred_pt: 
                pred_ = array_to_sitk(pred)
                pred_ = reorient_sitk(pred_,self.sitk_props[-1])
                pred_ = set_sitk_props(pred_,self.sitk_props)
                preds_out.append(pred_)
            return preds_out

class ArrayToSITKI(Transform):
    def __init__(self,sitk_props=None,img_sitk=None): 
        assert any([sitk_props,img_sitk]), "Either provide sitk_properties or a sitk_image to serve as templt"
        if not sitk_props: 
            sitk_props = img_sitk.GetOrigin(),img_sitk.GetSpacing(),img_sitk.GetDirection()
        self.sitk_props= sitk_props
    def encodes(self,pred):
                assert all([pred.ndim==3,'int' in str(pred.dtype)]), "This requires 3d int array, DxWxH"
                pred_ = array_to_sitk(pred)
                pred_ = reorient_sitk(pred_,self.sitk_props[-1])
                pred_ = set_sitk_props(pred_,self.sitk_props)
                return pred_


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
    '''
    If the BBox is smaller than the patch size, it helps to enlarge it (if BBox is larger, the grid_sampler takes care of it).
    '''
    
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
class ChangeDType(KeepBBoxTransform):
    def __init__(self,target_dtype):store_attr()
    def func(self,img):
        return img.to(self.target_dtype)
        # dest = [200,300,400]
        # x  = np.random.rand(50,100,200)
        # R = ResizeNP(dest)
        # P = PadNpArray(patch_size=[160,160,160])
        # y = P.encodes([x])
        # R.encodes([x])
        
class DICOMOrientSITK(ItemTransform):
        def __init__(self): self.orientation = (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)
        def encodes(self,x):
            img,thing= x
            org_direction = img.GetDirection()
            if not org_direction==self.orientation:
                self.org_direction = org_direction
                return img.DICOMOrient(img,"LPS")
            return img,thing
        def decodes (self,x):
            if not hasattr(self,'org_direction'): return x
            img,thing = x
            img = img.SetDirection(self.org_direction)
            return img,thing

# %%

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

class ResampleToStage0(PredictorTransform):
    # Applies resample_spacing used to obtain training dataset corresponding to this model
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

        img = resize_tensor(img,self.sz_dest,self.mode)
        return img,bboxes_out
    def decodes(self,x):
        return resize_tensor(x,self.sz_source,mode='trilinear')

class Resize(KeepBBoxTransform):
    '''
    Strictly an inference transform. Requires img and bbox
    '''
    
    def __init__(self,dest_size,mode='trilinear'):
        store_attr()
    def func(self,img:Tensor)->Tensor:
        self.org_size = img.shape
        img = resize_tensor(img,self.dest_size,self.mode)
        return img
    def decodes(self,img:Tensor):
        if isinstance(img,list) or isinstance(img,tuple) :
            if len(img)>=2:
                img, bboxes=img
                has_bbox=True
            else: 
                img = img[0]
        mode = 'nearest' if 'int' in str(img.dtype) else 'trilinear'
        img = resize_tensor(img,self.org_size,mode)

        try :
            has_bbox==True
            return img, bboxes
        except: return img
# %%
def resize_tensor(img,target_size,mode):
        unsqueeze_times = 5-img.dim()
        for times in range(unsqueeze_times):img= img.unsqueeze(0)
        img = F.interpolate(img,target_size,mode=mode)
        for times in range(unsqueeze_times): img= img.squeeze(0)
        return img

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
class TransposeSITK(PredictorTransform):
    '''
    typedispatched class which handles arrays +/- bboxes. Bboxes, if present, MUST be in a list even if len-1
    '''

    def decodes(self,x): return self.encodes(x)

@TransposeSITK
def encodes(self,x:Union[list,tuple]):
    img,bboxes  = x
    func = torch.permute if isinstance(img,Tensor) else np.transpose
    img= func(img,[2,1,0])# not training is done on images facing up, and inference has to do the same.
    bboxes_out=[]
    for bbox in bboxes:
        bbox=bbox[2],bbox[1],bbox[0]
        bboxes_out.append(bbox)
    return img,bboxes_out

@TransposeSITK
def encodes(self,img:Union[Tensor,np.ndarray]):
    func = torch.permute if isinstance(img,Tensor) else np.transpose
    if img.ndim== 3: tp_list = [2,1,0]
    elif img.ndim==4: tp_list = [0,3,2,1]
    else: raise NotImplementedError
    img = func(img,tp_list)
    return img


# %%
    # bb = [slice(0,10),slice(0,20),slice(0,30)]
    # x = np.random.rand(50,100,200)
    #
    # T = TransposeSITK()
    # a = T([x,[bb]])
    # b = T.decodes(a)
# %%
# class Resample(Transform):
#     def __init__(self,org_size,org_spacing,dest_spacing,order=None):
#         self.sz_dest, self.scale_factor = get_scale_factor_from_spacings(org_size,org_spacing,dest_spacing)
#         store_attr()
#
#     def encodes(self,x):
#             x =  resize(x, self.sz_dest,order=self.order)
#             return x
#
#     def decodes(self,x):
#         x = x.unsqueeze(0).unsqueeze(0)
#         x = F.interpolate(input= x,size=self.org_size)
#         x = x.squeeze(0).squeeze(0)
#         return x
#
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
        

class LabelMapToBinary(Transform):
    '''
    outputs a binary mask of 1s and 0s.
    label: label which should be mapped to 1.
    list in merge_labels will merge mentioned labels into label
    '''
    def __init__(self,label,n_classes,return_type='numpy',merge_labels=[]):

        maps = [[x,0] for x in range(n_classes)]
        info= [label,1]
        altered= [[m,1] for m in merge_labels]
        altered.append(info)
        self.mapping=[]
        for m in maps:
            if not  m[0] in [x[0] for x in altered]:
                self.mapping.append(m)
        self.mapping.extend(altered)
        self.mapping.sort()
        self.remapper= MaskLabelRemap(self.mapping)
        self.return_type=return_type
        
   
    
    def encodes(self,x):
        _,mask = self.remapper([None,x])
        if self.return_type=='numpy': mask = np.array(mask)
        return mask


#
# # # %%
# # #
#                 pred_ = sitk.GetImageFromArray(pred)
# # #                 dd = self.sitk_props[-1]
# # #                 pred_ = reorient_sitk(pred_,self.sitk_props[-1])
# # #                 pred_ = set_sitk_props(pred_,self.sitk_props)
# # # # %%
# #                 pred2 = sitk.DICOMOrient(pred_,'LAS')
# # #                 pred2 = sitk.DICOMOrient(pred_,'PLS')
# # #                 pred2 = sitk.DICOMOrient(pred_,'PRS')
#                  pred2 = sitk.DICOMOrient(pred_,'PrS')
#                  pred2 = set_sitk_props(pred2,self.sitk_props)
# # # # %%
#                  sitk.WriteImage(pred2,"/home/ub/temp_prs.nrrd")
# # #                 preds_out.append(pred_)
# # # %%
