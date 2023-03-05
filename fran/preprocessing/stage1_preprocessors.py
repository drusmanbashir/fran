
# %%
from batchgenerators.augmentations.utils import center_crop_3D_image
import torchio as tio
import cc3d

from fran.transforms.spatialtransforms import PadDeficitImgMask, get_amount_to_pad_torch, get_slices_shifted_from_centroid, get_bbox_size

from fran.utils.image_utils import resize_tensor_3d
if 'get_ipython' in globals():
        print("setting autoreload")
        from IPython import get_ipython
        ipython = get_ipython()
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
from fastcore.basics import itertools, listify, store_attr
import nibabel as nib
from fastai.vision.augment import mask_tensor
from skimage.transform import resize
from functools import partial
from pathlib import Path
from typing import Union
import h5py
import sys,os, json
from fran.preprocessing.datasetanalyzers import BBoxesFromMask, bboxes_function_version, get_cc3d_stats
from fran.utils.fileio import *
from fran.utils.imageviewers import *
from fran.utils.fileio import save_np
from fran.utils.helpers import   *
import ipdb
import SimpleITK as sitk
import numpy as np
from typing import List
from torch.nn.functional import pad
from multiprocessing import Pool
import os, sys
from fastai.basics import L
# from fastai.vision.all import *
# export
import ipdb, re
tr = ipdb.set_trace
import torch.nn.functional as F
from fran.preprocessing.stage0_preprocessors import generate_bboxes_from_masks_folder
# %%

def tensors_from_dict_file(filename):
        img_mask= torch.load(filename)
        img = img_mask['img']
        mask = img_mask['mask']
        return img,mask

def view_sitk(img_mask_pair):
    img,mask = img_mask_pair
    img,mask=map(sitk.GetArrayFromImage,[img,mask])
    img, mask = img.transpose(2,1,0), mask.transpose(2,1,0)
    ImageMaskViewer([img,mask])



def resample_tensor_dict(in_filename,out_filename,output_size,overwrite=True):
    if write_file_or_not(out_filename, overwrite) == True:
        img_mask= torch.load(filename)
        resized_tensor={}
        resized_tensor = {{img_type: resize_tensor_3d(tensr)} for img_type,tensr in img_mask.items()}
        torch.save(resized_tensor,out_filename)



def resample_img_mask_tensors(in_filename,out_filename,output_size,overwrite=True):
    if write_file_or_not(out_filename, overwrite) == True:
        img,mask = tensors_from_dict_file(in_filename)
        resized_tensor={}
        for x, key in zip([img,mask],['img','mask']):
            mode = 'nearest' if 'int' in str(x.dtype) else 'trilinear'
            x= resize_tensor_3d(x,output_size,mode)
            resized_tensor.update({key:x})
        torch.save(resized_tensor,out_filename)

def calculate_patient_bbox(img,threshold):
    ii = img.clone()
    ii[img<threshold]= 0
    ii[img>=threshold]=1
    stats_patient = cc3d.statistics(ii.numpy().astype(np.uint8))
    patient_bb = stats_patient['bounding_boxes'][1]
    return patient_bb


def files_exist(filename,any_or_all="any"):
    if not isinstance(filename,list):
        filename = listify(filename)
    return any([fn.exists() for fn in filename])

class CropToPatientTorchToTorch(object):
    def __init__(self,output_parent_folder,spacings,pad_each_side: str='4cm',overwrite=True):
        '''
        params:
        target_length : 20 cm by default 
        '''
        self.output_folders=output_parent_folder
        assert 'cm' in pad_each_side, "Must give length in cm for clarity" 
        maybe_makedirs(self.output_folders)
        pad_each_side = float(pad_each_side[:-2])
        self.overwrite = overwrite 
        self.pad_voxels_each_side= int(pad_each_side*10/spacings[0])


    @property
    def output_folders(self):
        return self._output_folders

    @output_folders.setter
    def output_folders(self,output_parent_folder):
        self._output_folders= [output_parent_folder/subfld for subfld in ["images","masks"]]
 
    @property
    def output_filenames(self):
        return self._output_filenames       

    @output_filenames.setter
    def output_filenames(self,filename):
        self._output_filenames = [fldr/filename.name for fldr in self._output_folders]

    def _save_to_file(self):
                argss = zip([self.img_cropped,self.mask_cropped],self.output_filenames)
                [torch.save(a,b) for a,b in argss]
    def process_case(self,img_fn,  mask_fn: Path,threshold=-0.4):
            self.output_filenames = img_fn
            if files_exist(self.output_filenames) and self.overwrite==False:
                print("File(s) {} exists. Skipping.".format(self.output_filenames))
                return 0
            else:
                self.img_cropped,self.mask_cropped = self._load_and_crop(img_fn,mask_fn,threshold)
                self._save_to_file()
                return 1

    def _load_and_crop(self, img_fn,mask_fn,threshold):
                img,mask = map(torch.load,[img_fn,mask_fn])
                # organ_z_center, organ_length_voxels= self._get_organ_stats(mask)
                # pad_total_each_side=int(organ_length_voxels/2+self.pad_voxels_each_side)
                # slices_craniocaudal= slice(int(np.maximum(0,organ_z_center-pad_total_each_side)),int(np.minimum(mask.shape[0],organ_z_center+pad_total_each_side)))
                patient_bb = calculate_patient_bbox(img.clone(),threshold=threshold)
                cropped_bb = tuple([patient_bb[0],patient_bb[1],patient_bb[2]])
                img_cropped,mask_cropped = img[cropped_bb].clone(),mask[cropped_bb].clone()
                return img_cropped,mask_cropped

    def _get_organ_stats(self,mask):
        mask_binary = mask.clone().numpy()
        mask_binary[mask_binary>1]=1
        stats_mask = cc3d.statistics(mask_binary)
        mask_centroid= stats_mask['centroids'][-1]
        organ_z_center= int(mask_centroid[0])
        organ_length_voxels = stats_mask['bounding_boxes'][1][0].stop - stats_mask['bounding_boxes'][1][0].start
        return organ_z_center,organ_length_voxels


class CropToPatientTorchToNifty(CropToPatientTorchToTorch):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)
        pass

    @property
    def output_folders(self):
        return self._output_folder

    @output_folders.setter
    def output_folders(self,output_folder):
        self._output_folder= [output_folder/"images_nii"/"images", output_folder/"images_nii"/"masks"]


    @property
    def output_filenames(self):
        return self._output_filename

    @output_filenames.setter
    def output_filenames(self,filename):
        self._output_filename =[folder/(str(filename.name).replace(".pt",".nii.gz")) for folder in self._output_folder]

    def _save_to_file(self,img_cropped,mask_cropped):
        for im, fn in zip([img_cropped,mask_cropped],self.output_filenames):
                    im = im.numpy()
                    save_to_nii(im,fn)

    def _get_organ_stats(self,mask):
        mask_binary = mask.clone().numpy()
        mask_binary[mask_binary>1]=1
        stats_mask = cc3d.statistics(mask_binary)
        mask_centroid= stats_mask['centroids'][-1]
        organ_z_center= int(mask_centroid[0])
        organ_length_voxels = stats_mask['bounding_boxes'][1][0].stop - stats_mask['bounding_boxes'][1][0].start
        return organ_z_center,organ_length_voxels

class WholeImageTensorMaker():
    def __init__(self,proj_defaults,source_spacings, output_size,num_processes):
        store_attr('proj_defaults, source_spacings,output_size,num_processes')
        resampling_configs_fn = proj_defaults.fixed_spacings_folder/("resampling_configs")
        resampling_configs = load_dict(resampling_configs_fn)
        self.set_files_folders(resampling_configs)
        if any([not fn.exists() for fn in self.mask_files]): raise "Some file(s) do not exist. Dataset corrupt"
        print("Run process_tensors()")


    def set_files_folders(self,resampling_configs):
        self.input_folder = [conf['resampling_output_folder'] for conf in resampling_configs if conf['spacings']== self.source_spacings][0]
        self.output_parent_folder = self.proj_defaults.whole_images_folder/("dim_{0}_{1}_{2}".format(*self.output_size))
        self.output_folder_imgs = self.output_parent_folder/("images")
        self.output_folder_masks= self.output_parent_folder/("masks")
        self.img_files = list((self.input_folder/("images")).glob("*pt"))
        self.mask_files = [self.input_folder/("masks/{}".format(fn.name)) for fn in self.img_files]

    def get_args_for_resizing(self):
        maybe_makedirs([self.output_folder_imgs,self.output_folder_masks])
        arglist_imgs = [[img_filename,self.output_size,'trilinear',self.output_folder_imgs/img_filename.name] for img_filename in self.img_files]
        arglist_masks= [[mask_filename,self.output_size,'nearest',self.output_folder_masks/mask_filename.name] for mask_filename in self.mask_files]
        return arglist_imgs, arglist_masks

    # def generate_bboxes_from_masks_folder(self,debug=False,num_processes=8):
    #     generate_bboxes_from_masks_folder(self.output_folder_masks,self.proj_defaults,0.2,debug,num_processes)

def resize_and_save_tensors(input_filename,output_size, mode,output_filename):
        input_tensor = torch.load(input_filename)
        resized_tensor = resize_tensor_3d(input_tensor,output_size,mode)
        torch.save(resized_tensor.to(torch.float32 if mode=='trilinear' else torch.uint8),output_filename)
def cropper_wrapper_nifty(filename,args):
        C = CropToPatientTorchToNifty(*args)

        return C.process_case(filename)

def cropper_wrapper_torch(filename,args):
        C = CropToPatientTorchToTorch(*args)
        return C.process_case(filename)

def get_cropped_label_from_bbox_info(outfolder, bbox_info,label="tumour",label_index=2):
    filename = bbox_info['filename']
    out_filename = outfolder/filename.name
    img_mask= torch.load(filename)


    img = img_mask['img']
    mask = img_mask['mask']

    bbox_stats = bbox_info['bbox_stats']
    ref = [ b for b in bbox_stats if b['tissue_type']==label]
    try:
        refo = ref[0]
        slcs =  refo['bounding_boxes']
        slc = slcs[1]
        img_tmr = img[slc]
        mask_tmr = mask[slc]
        mask_tmr[mask_tmr!=label_index]=0
        mask_tmr[mask_tmr==label_index]=1
        img_tmr = img_tmr*mask_tmr
        mask_tmr[mask_tmr==1]=label_index
        out_tensr ={'img':img_tmr,'mask':mask_tmr}
        torch.save(out_tensr,out_filename)
        return out_filename
    except:
        print("Label {0} not in this case {1}".format(label,bbox_info['case_id']))
        return 0,filename


def pad_bbox(bbox,padding_torch_style): # padding is reverse order Torch style
    padding = padding_torch_style[::-1]
    assert len(padding)==6, "Padding must be 6-tuple"
    out_slcs=[]
    for indx in range(len(bbox)):
        slcs = bbox[indx]
        slc_new = slice(int(slcs.start-padding[indx*2]),int(slcs.stop+padding[indx*2+1]))
        out_slcs.append(slc_new)
    return tuple(out_slcs)
        


class PatchGenerator(DictToAttr):
    def __init__(self,dataset_properties:dict, output_folder,output_patch_size,info, patch_overlap=(0,0,0) ,expand_by=None):
        '''
        generates function from 'all_fg' bbox associated wit the given case
        expand_by is specified in mm, i.e., 30 = 30mm. 
        spacings are essential to compute number of array elements to add in case expand_by is required. Default: None
        '''
        store_attr('output_folder,output_patch_size,info,patch_overlap')
        self.output_masks_folder = output_folder/("masks")
        self.output_imgs_folder = output_folder/("images")
        self.mask_fn = info['filename']
        self.img_fn = Path(str(self.mask_fn).replace("masks","images"))
        self.assimilate_dict(dataset_properties)
        bbs= info['bbox_stats']

        b = [b for b in bbs if b['tissue_type']=="all_fg"][0]
        self.bboxes = b['bounding_boxes'][1:]
        if expand_by:
            self.add_to_bbox=  [ int(expand_by/sp) for sp in self.dataset_spacings]
        else: self.add_to_bbox=[0.,]*3


    def load_img_mask_padding(self):
        mask = torch.load(self.mask_fn)
        img = torch.load(self.img_fn)
        self.img,self.mask,self.padding = PadDeficitImgMask(patch_size=self.output_patch_size,input_dims=3, pad_values = [self.dataset_min,0],return_padding_array=True).encodes([img,mask])
        self.shift_bboxes_by = list(self.padding[::2])
        self.shift_bboxes_by.reverse()


    def maybe_expand_bbox(self,bbox):
        bbox_new=[]
        # for s,ps in zip(bbox,min_sizes):
        #     sz = int(s.stop-s.start)
        #     diff  = np.maximum(0,ps-sz*stride)
        #     start_new= int(s.start-np.ceil(diff/2))
        #     stop_new = int(s.stop+np.floor(diff/2))
        #     s_new = slice(start_new,stop_new,stride)
        #     bbox_new.append(s_new)
        for s,shift, ps,imsize,exp_by in zip(bbox,self.shift_bboxes_by,self.output_patch_size,self.img.shape,self.add_to_bbox):
                s = slice(int(np.maximum(0,s.start+shift-exp_by)), int(np.minimum(imsize,s.stop+shift+exp_by)))
                sz = int(s.stop-s.start)
                ps_larger_by= np.maximum(0,ps-sz)
                start_tentative= int(s.start-np.ceil(ps_larger_by/2))
                stop_tentative= int(s.stop+np.floor(ps_larger_by/2))
                shift_back = np.minimum(0,imsize-stop_tentative)
                shift_forward = abs(np.minimum(0,start_tentative))
                shift_final = shift_forward+shift_back
                start_new = start_tentative+shift_final
                stop_new = stop_tentative+shift_final
                s_new = slice(start_new,stop_new,None)
                bbox_new.append(s_new)
        return bbox_new
    def create_grid_sampler_from_patchsize(self,bbox_final):
        img_f = self.img[tuple(bbox_final)].unsqueeze(0)
        mask_f= self.mask[tuple(bbox_final)].unsqueeze(0)
        img_tio = tio.ScalarImage(tensor=img_f)
        mask_tio = tio.ScalarImage(tensor=mask_f)
        subject= tio.Subject(image=img_tio,mask=mask_tio)
        self.grid_sampler = tio.GridSampler(subject=subject,patch_size=self.output_patch_size,patch_overlap=self.patch_overlap)

    def create_patches_from_grid_sampler(self):
        for i, a in enumerate(self.grid_sampler):
            out_fname = self.mask_fn.name.split(".")[0]+"_"+str(i)+".pt"
            out_mask_fname = self.output_masks_folder/out_fname
            out_img_fname = self.output_imgs_folder/out_fname
            print("Saving to files {0} and {1}".format(out_img_fname, out_mask_fname))
            img= a['image'][tio.DATA].squeeze(0)
            mask= a['mask'][tio.DATA].squeeze(0)
            torch.save(img,out_img_fname)
            torch.save(mask,out_mask_fname)
    def create_patches_from_all_bboxes(self):
        print("Creating patches for case {}".format(self.info['case_id']))
        self.load_img_mask_padding()
        for bbx in self.bboxes:
            bbox_new=self.maybe_expand_bbox(bbx)
            self.create_grid_sampler_from_patchsize(bbox_new)
            self.create_patches_from_grid_sampler()


# %%
def to_even(input_num, lower=True):
    np.fnc = np.subtract if lower==True else np.add
    output_num = np.fnc(input_num,input_num%2)
    return int(output_num)

def patch_generator_wrapper(output_folder,output_patch_size, info,oversampling_factor=0,expand_by=None):
    # make sure output_folder already has been created
    if not oversampling_factor: oversampling_factor=0.
    assert oversampling_factor<0.9 , "That will create a way too large data folder. Choose an oversampling_factor between [0, 0.9)"
    patch_overlap = [int(oversampling_factor*ps) for ps in output_patch_size]
    patch_overlap=map(to_even,patch_overlap)

    dataset_properties_fn = info['filename'].parent.parent/("resampled_dataset_properties")
    dataset_properties=load_dict(dataset_properties_fn)
    P= PatchGenerator(dataset_properties,output_folder,output_patch_size, info,patch_overlap,expand_by)
    P.create_patches_from_all_bboxes()
    return 1,info['filename']
# %%

if __name__ == "__main__":
    ######################################################################################
    # %% [markdown]
    ## Creates low res images
    # %%

# %%
    
    from fran.utils.common import *
    P = Project(project_title="lits"); proj_defaults= P.proj_summary
    output_shape=[128,128,96]
    overs = .25
    fixed_folder = proj_defaults.fixed_spacings_folder/("spc_077_077_100/images")
    fixed_files = list(fixed_folder.glob("*.pt"))
# %%
    n=0
    img_fn = fixed_files[n]
    mask_fn =img_fn.str_replace('images','masks')
# %%

    spacings= load_dict( proj_defaults.resampled_dataset_properties_filename)['preprocessed_dataset_spacings']
    global_props =  load_dict(proj_defaults.raw_dataset_properties_filename)[-1]
# %%


    C = CropToPatientTorchToTorch(output_parent_folder=Path("tmp"),spacings=[.77,.77,1],pad_each_side='0cm')
    C.process_case(img_fn=img_fn,mask_fn = mask_fn )

# %%
    img,mask = map(torch.load,[img_fn,mask_fn])
    ImageMaskViewer([img,mask])
# %%

    ImageMaskViewer([C.img_cropped,C.mask_cropped])



# %%
    stage1_fldr = Path("/home/ub/datasets/preprocessed/lits/patches/spc_077_077_100/dim_320_320_256/")
    img_fldr = stage1_fldr/('images')
    mask_fldr = stage1_fldr/('masks')
    stage1_img_fn= list(img_fldr.glob("*"))
    stage1_mask_fn= list(mask_fldr.glob("*"))
# %%
    n=100
    filenames = [stage1_img_fn[n], stage1_mask_fn[n]]
    img,mask = list(map(torch.load,filenames))
    x = [x.permute(2,1,0) for x in [img,mask]]
    ImageMaskViewer([x[0],x[1]],intensity_slider_range_percentile=[0,100])
    # ImageMaskViewer([img[lims],mask[lims]])
# %%
######################################################################################
# %% [markdown]
## Trialling torch to nibabel format for rapid loading 
# 

# %%
# %%
    # stage0_few=stage0_files[:20]
    multiprocess_multiarg(cropper_wrapper_nifty,args,debug=False)
    args = [[filename,[output_folder,spacings,'3.5cm',True]] for filename in stage0_files]
# %%
    n=0
    filename =stage0_files[0]
    filename_cropped = output_folder/filename.name
    filename='/home/ub/datasets/preprocessed/kits21/stage0//images/kits21_00147.pt'
    filename0='/home/ub/datasets/preprocessed/kits21/stage1/cropped/images/kits21_00147.pt'
    t = torch.load(filename)
    img,mask = t['img'],t['mask']

    ImageMaskViewer([img,mask])

    os.path.getsize(filename),os.path.getsize(filename0)

######################################################################################
######################################################################################
######################################################################################
# %% [markdown]
##  Creating tumour examples at original stage0 resolution
# %%
    
    folder = proj_defaults.stage0_folder/"images"
    outfolder = folder.parent/("tumour_only")
    maybe_makedirs(outfolder)
    label = "tumour"
    bboxes = load_dict(folder.parent/"bboxes_info")
    args=[[outfolder,info,"tumour"] for info in bboxes]

    a= multiprocess_multiarg(get_cropped_label_from_bbox_info,args,multiprocess=True,debug=False)

# %%

    fn = '/home/ub/datasets/preprocessed/kits21/stage0/tumours_only/kits21_00000.pt'
    im = torch.load(fn)
    img,mask = im['img'],im['mask']
    ImageMaskViewer([img,mask])
# %%
######################################################################################
# %% [markdown]
## Creating tumour examples for augmentation
# %%

    outfolder = folder.parent/("tumour_only")
    maybe_makedirs(outfolder)
    label = "tumour"
    bboxes = load_dict(folder.parent/"bboxes_info")

    a= multiprocess_multiarg(get_cropped_label_from_bbox_info,args,multiprocess=True,debug=True)
    args=[[outfolder,info,"tumour"] for info in bboxes]
# %%
######################################################################################
# %% [markdown]
## Cropped to kidney dataset
# %%
    outfolder = proj_defaults.stage1_folder/("cropped_separate_kidneys")
# %%
    caseid = "00002"
    b = [bb for bb in stage0_bbox if bb['case_id']==caseid ][0]
# %%
    P = PatchGenerator(output_folder,patch_size,info=b)
    P.create_patches_from_all_bboxes()
# %%
######################################################################################
# %% [markdown]
## Creating Patches separate kidney
# %%
######################################################################################
# %% [markdown]
## BBoxes need to only get 1 largest component per kidney , otherwise it introduces bugs ifusing default 2
# %%
    dusting_threshold_factor=1 # as no subsampling in this dataset
    filenames = list(output_folder.glob("*.pt"))
    label_settings = proj_defaults.mask_labels
    label_settings[0]['k_largest']=1

    arguments =[[x,proj_defaults,label_settings,dusting_threshold_factor] for x in filenames]
    res_cropped= multiprocess_multiarg(func=bboxes_function_version,arguments=arguments,num_processes=16,debug=False)
    save_dict(res_cropped,output_folder.parent/"bboxes_info")
# %%

# %%
    fn = '/home/ub/datasets/preprocessed/kits21/stage0/images1/kits21_00011.pt'
    im = torch.load(fn)
    img,mask = im['img'],im['mask']
    ImageMaskViewer([img,mask])
# %%
# %%
    mask_fn = '/s/datasets/raw_database/raw_data/kits21/masks/kits21_00011.nii.gz'
    tumour_fn ='/s/datasets/raw_database/raw_data/kits21/images/kits21_00011.nii.gz' 
    img = sitk.ReadImage(tumour_fn)
    mask =sitk.ReadImage(mask_fn)
    view_sitk([img,mask])

# %% [markdown]
## Creating bboxes from nifty masks
# %%
    folder = proj_defaults.stage1_folder/("cropped/images_nii/masks")
    cropped_masks_filenames = get_fileslist_from_path(folder,ext=".nii.gz")

    filename = cropped_masks_filenames[0]

    mask = nib.load(filename)
    dat = mask.get_fdata()
    dat.dtype

    arguments =[[x,proj_defaults] for x in cropped_masks_filenames]
# %% [markdown]
## Splitting dict files into separate img mask tensor files
    in_fldr = stage1_subfolder
    tnsrs = list(in_fldr.glob("*pt"))
    im = tensors_from_dict_file(tnsrs[0])
    outfldrs =[in_fldr]
# %%

    res_cropped= multiprocess_multiarg(func=bboxes_function_version,arguments=arguments,num_processes=16,debug=True)
# %%
