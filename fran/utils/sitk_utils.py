from pathlib import Path
import ast
from fastai.callback.tracker import Union
from fastcore.basics import store_attr
import numpy as np
from fastai.vision.augment import test_eq
from torch.functional import Tensor
from fran.utils.fileio import str_to_path

import SimpleITK as sitk
from fran.utils.helpers import abs_list
import ipdb
tr = ipdb.set_trace
from fastcore.transform import Transform, ItemTransform
import itertools


# %%

def array_to_sitk(arr:Union[Tensor,np.ndarray]):
    '''
    converts cuda to cpu. Rest is as sitk.GetImageFromArray
    '''
    if isinstance(arr,Tensor) and arr.device.type=='cuda':
        arr = arr.detach().cpu()
    return sitk.GetImageFromArray(arr)
    

class ReadSITK(Transform):
    def encodes(self,x): return sitk.ReadImage(x)

class ReadSITKImgMask(ItemTransform):
    '''
    Applied to tuple(img, mask)
    '''
    
    def encodes(self,x):
        return list(map(sitk.ReadImage, x))


class SITKDICOMOrient(Transform):    
    '''
    Re-orients SITK Images to DICOM. Allows all other datatypes to pass
    '''

    def __init__(self):
        self.dicom_orientation =  (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)
    def encodes(self,x:sitk.Image):
            if isinstance(x,sitk.Image):
                if x.GetDirection!=self.dicom_orientation:
                    x = sitk.DICOMOrient(x,"LPS")
            return x

class SITKImageMaskFixer():
    @str_to_path([1,2])
    def __init__(self,img_fn, mask_fn): 
        store_attr()
        self.img,self.mask = map(sitk.ReadImage,[img_fn,mask_fn])
        self.dicom_orientation =  (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)
    def process(self,fix=True,outname=None):
        self.essential_sitk_props()
        self.verify_img_mask_match()
        if self.match_string!="Match" and fix==True: 
            self.mask = align_sitk_imgs(self.img,self.mask)
            self.match_string="Repaired"
            self.to_DICOM_orientation()
        self.save_altered_sitk(outname)


    def save_altered_sitk(self,outname=None):
        if self.match_string =="Repaired":
            if outname:
                sitk.WriteImage(self.mask, self.mask_fn.parent/(f"{outname}_mask.nii"))
            else:
                sitk.WriteImage(self.mask,self.mask_fn)
        elif "changed to DICOM" in self.match_string:
            if not outname:
                itertools.starmap(sitk.WriteImage,[(self.img,self.img_fn),(self.mask,self.mask_fn)])
            else:
                img_fn = self.img_fn.parent/(f"{outname}_img.nii")
                mask_fn= self.img_fn.parent/(f"{outname}_mask.nii")
                itertools.starmap(sitk.WriteImage,[(self.img,img_fn),(self.mask,mask_fn)])
            

    def to_DICOM_orientation(self):
        direction = np.array(abs_list(ast.literal_eval(self.pairs[0][0])))
        try:
            test_eq(direction,self.dicom_orientation)
        except: 
            print(f"Changing img/mask orientation from {direction} to {np.eye(3)}")
            self.img, self.mask = map(
                lambda x: sitk.DICOMOrient(x, "LPS"), [self.img, self.mask]
            )
            self.match+=", changed to DICOM"
            self.pairs.insert(1,str(self.dicom_orientation)*2)

    def verify_img_mask_match(self ):
            matches = [a==b for a,b in self.pairs]
            if all(matches):         self.match_string = 'Match' 
            elif not matches[0]: raise Exception("Irreconciable difference in sizes. Check img/mask pair {}".format(self.mask_fn))
            else:
                self.match_string= 'Mismatch'
           
    def essential_sitk_props(self):
            directions,sizes,spacings = [],[],[]
            for arr in self.img,self.mask:
                directions.append(str(abs_list(arr.GetDirection())))
                sizes.append(str(arr.GetSize()))
                _spacing = arr.GetSpacing()
                _spacing = list(map(lambda x: round(x,2), _spacing))
                spacings.append(str(_spacing))
            self.pairs =[directions] + [sizes] + [spacings]

    @property 
    def log(self):
        return [self.match_string]+[self.img_fn,self.mask_fn]+self.pairs

def set_sitk_props(img:sitk.Image,sitk_props:Union[list,tuple])->sitk.Image:
        origin,spacing,direction = sitk_props
        img.SetOrigin(origin)
        img.SetSpacing(spacing)
        img.SetDirection(direction)
        return img
def align_sitk_imgs(img,img_template):
                    img = set_sitk_props(img,[img_template.GetOrigin(),img_template.GetSpacing(),img_template.GetDirection()])
                    # img.CopyInformation(img_template)
                    return img


def create_sitk_as(img:sitk.Image,arr:Union[np.array,Tensor]=None)->sitk.Image:
    if arr is not None:
        img_new = sitk.GetImageFromArray(arr)
    else:
        img_new = sitk.Image(*img.GetSize())
    img_new = align_sitk_imgs(img_new,img)
    return img_new

def get_meta(img:sitk.Image)->list   :
    res = img.GetSize(), img.GetSpacing(), img.GetDirection()
    return  res
# %%
if __name__ == "__main__":
    img_fn =  Path('/media/ub/datasets_bkp/litq/complete_cases/images/litq_0014389_20190925.nii')
    mask_fn = "/media/ub/UB11/datasets/lits_short/segmentation-51.nii"
# %%
    F = SITKImageMaskFixer(img_fn,mask_fn)
    F.process(fix=True,outname="jackson")
