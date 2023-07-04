
# %%
from pathlib import Path
from fastcore.all import is_close

from fastcore.test import test_close
import ast
from fastai.callback.tracker import Union, shutil
from fastcore.basics import store_attr
import numpy as np
from fastai.vision.augment import test_eq
from torch.functional import Tensor
from fran.utils.imageviewers import ImageMaskViewer
from fran.utils.fileio import maybe_makedirs, str_to_path

import SimpleITK as sitk
from fran.utils.helpers import abs_list, get_extension
import ipdb

from fran.utils.string import cleanup_fname
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
            matches =[] 
            for l in self.pairs:
                l = [ast.literal_eval(la) for la in l]
                match = is_close(*l,eps=1e-5)
                matches.append(match)

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

def fix_slicer_labelmap(mask_fn,img_fn):
    '''
    slicer output labelmaps are not full sized but just a bbox of the labels
    this function zero-fills outside the bbox to match imge size
    
    '''
    print("Processing {}".format(mask_fn))
    img = sitk.ReadImage(img_fn)
    mask = sitk.ReadImage(mask_fn)
    m = sitk.GetArrayFromImage(mask)
    i_shape = sitk.GetArrayFromImage(img).shape
    m_shape = m.shape
    if i_shape==m_shape:
        print("Identical shaped image and mask. Nothing done")
    else:
        print("Image shape is {0}. Mask shape is {1}. Creating mask backup in /tmp folder and fixing..".format(i_shape,m_shape))
        mask_bk_fn = Path("/tmp")/mask_fn.name
        shutil.copy(mask_fn,mask_bk_fn)

        distance =[a-b for a,b in zip(mask.GetOrigin(),img.GetOrigin())]
        tr()
        ad = [int(d/s) for d,s in zip(distance,img.GetSpacing())]
        ad.reverse()
        shp = list(img.GetSize())
        shp.reverse()
        zers = np.zeros(shp)
        zers[ad[0]:ad[0]+m_shape[0],ad[1]:ad[1]+m_shape[1],ad[2]:ad[2]+m_shape[2]] = m
        mask_neo = create_sitk_as(img,zers)
        sitk.WriteImage(mask_neo,mask_fn)

@str_to_path(0)
def compress_img(img_fn):
    '''
    if img_fn is a symlink. This will alter the target file
    '''
    img_fn = img_fn.resolve()
    
    e = get_extension(img_fn)
    fn_neo = img_fn.str_replace(e,"nii.gz")
    fn_old_bkp = img_fn.str_replace("/s","/s/tmp")
    fn = Path("/s/xnat_shadow/litq/images/litq_48_20200107.nii.gz")
    
    for parent in fn_old_bkp.parents:
        maybe_makedirs(parent)
    if e!="nii.gz":
        print("Converting {0}  -----> {1}".format(img_fn,fn_neo))
        img = sitk.ReadImage(img_fn)
        sitk.WriteImage(img,fn_neo)
        shutil.move(img_fn,fn_old_bkp)

        
    else:
        print("File {} already nii.gz format. Nothing to do.".format(img_fn))
    
@str_to_path(0)
def compress_fldr(fldr:Path, recursive=True):
    if recursive==True:
        files = fldr.rglob("*")
    else:
        files = fldr.glob("*")
    for fn in files:
        if fn.is_file():
            compress_img(fn)

# %%
if __name__ == "__main__":
# %%
    fldr =Path("/home/ub/Desktop/capestart/liver/masks")
    compress_fldr(fldr)
# %%


    imgs
# %%
    masks_fldr = Path("/home/ub/Desktop/capestart/liver/masks")
    images_fldr = Path("/s/datasets_bkp/litq/sitk/images/")
    masks = list(masks_fldr.glob("*"))
    imgs = list(images_fldr.glob("*"))
    for mask_fn in masks:
        mt = cleanup_fname(mask_fn.name)
        img_fn =[fn for fn in imgs if cleanup_fname(fn.name)==mt]
        if len(img_fn)>1 or len(img_fn)==0:
            tr()
        else:
            fix_slicer_labelmap(mask_fn,img_fn[0])
# %%
     


    fldr = Path("/s/datasets_bkp/drli/masks/")
    for fn in fldr.glob("*"):
        imgfn = fn.str_replace("masks","images")
        if not imgfn.exists():
            tr()
# %%

    img_fn =  Path('/media/ub/datasets_bkp/litq/complete_cases/images/litq_0014389_20190925.nii')
    mask_fn = Path("/media/ub/UB11/datasets/lits_short/segmentation-51.nii")
    compress_img(img_fn)
# %%


    img_fn = Path("/s/fran_storage/datasets/raw_data/lits2/images/litq_77_20210306.nii.gz")
    mask_fn = "/s/fran_storage/datasets/raw_data/lits2/masks/litq_77_20210306.nrrd"
    mask_outfn = "/s/fran_storage/datasets/raw_data/lits2/masks/litq_77_20210306_fixed.nrrd"
    img = sitk.ReadImage(img_fn)
    np_a = sitk.GetArrayFromImage(img)
    np_a = np_a.transpose(2,1,0)
    
    np2 = np.mean(np_a,1)
    np2 = np2.transpose(0,1)
    plt.imshow(np2)
    ImageMaskViewer([a,a])

