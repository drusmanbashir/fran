# nvidia measure command
# timeout -k 1 2700 nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > results-file.csv
# short version below
# timeout -k 1 2700 nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > results-file.csv
# %%

# from torch.utils.data import DataLoader
# from fastai.data.transforms import DataLoader as DL2
from collections.abc import Hashable, Mapping
from fastcore.basics import Dict
from monai.data.dataset import PersistentDataset
from monai.data.image_writer import ITKWriter
from monai.transforms.transform import Transform
from monai.transforms.croppad.dictionary import   ResizeWithPadOrCropd
from monai.transforms.intensity.dictionary import NormalizeIntensityd, RandAdjustContrastd, RandScaleIntensityd, RandShiftIntensityd, ScaleIntensityRanged, ThresholdIntensityd,RandGaussianNoised
from monai.transforms.spatial.dictionary import RandFlipd
from monai.transforms.utility.dictionary import  EnsureChannelFirstd
from monai.transforms import Compose, MapTransform
from monai.utils.enums import TransformBackends
import numpy as np
import operator 
from functools import reduce

import itertools
from functools import partial

from fran.transforms.intensitytransforms import standardize
from torch.utils.data import Dataset
from fran.utils.helpers import *
from fran.utils.imageviewers import ImageMaskViewer
from monai.transforms import CropForegroundd

from fran.transforms.spatialtransforms import *
import ipdb

from fran.utils.string import strip_extension

tr = ipdb.set_trace
# path=  proj_default_folders.preprocessing_output_folder
# imgs_folder =  proj_default_folders.preprocessing_output_folder/("images")
# masks_folder=  proj_default_folders.preprocessing_output_folder/("masks")
#
from fran.utils.fileio import *

# %%
# export
def foldername_from_shape(parent_folder, shape):
    shape = str(shape).strip("[]").replace(",", "_").replace(" ", "")
    output = Path(parent_folder) / shape
    return output




def maybe_set_property(func):
        def inner(cls,*args,**kwargs):
            prop_name = "_"+func.__name__
            if not hasattr(cls,prop_name):
                prop =  func(cls, *args,**kwargs)
                setattr(cls,prop_name, prop)
            return getattr(cls,prop_name)
        return inner
            

class ImageMaskBBoxDataset(Dataset):
    """
    takes a list of case_ids and returns bboxes image and label
    """

    def __init__(self,fnames, bbox_fn , class_ratios:list=None,transform=None):

        """
        class_ratios decide the proportionate guarantee of each class in the output including background. While that class is guaranteed to be present at that frequency, others may still be present if they coexist
        """
        self.transform=transform
        if not class_ratios: 
            self.enforce_ratios = False
        else: 
            self.class_ratios = class_ratios
            self.enforce_ratios = True

        print("Loading dataset from BBox file {}".format(bbox_fn))
        bboxes_unsorted = load_dict(bbox_fn)
        self.bboxes_per_id = []
        for fn in fnames:
            bboxes = self.match_raw_filename(bboxes_unsorted, fn)
            bboxes.append(self.get_label_info(bboxes))
            self.bboxes_per_id.append(bboxes)


    def match_raw_filename(self,bboxes,fname:str):
        bboxes_out=[]
        fname = strip_extension(fname)
        for bb in bboxes:
            fn = bb['filename']
            fn_no_suffix=cleanup_fname(fn.name)
            if fn_no_suffix==fname:
                bboxes_out.append(bb)
        if len(bboxes_out) == 0:
                print("Missing filename {0} from bboxfile".format(fn))
                tr()
        return bboxes_out



    def __len__(self):
        return len(self.bboxes_per_id)

    def __getitem__(self, idx):
        self.set_bboxes_labels(idx)
        if self.enforce_ratios == True:
             self.mandatory_label = self.randomize_label() 
             self.maybe_randomize_idx()

        filename, bbox = self.get_filename_bbox()
        img,label = self.load_tensors(filename)
        if self.transform is not None:
            img,label,bbox= self.transform([img,label,bbox])
      
        return img,label,bbox

    def load_tensors(self,filename:Path):
        label = torch.load(filename)
        if isinstance(label, dict):
            img, label = label["img"], label["label"]
        else:
            img_folder = filename.parent.parent / ("images")
            img_fn = img_folder / filename.name
            img = torch.load(img_fn)
        return img,label

    def set_bboxes_labels(self,idx):
         self.bboxes = self.bboxes_per_id[idx][:-1]
         self.label_info =self.bboxes_per_id[idx][-1]
    def get_filename_bbox(self):
        if self.enforce_ratios==True:
            candidate_indices= self.get_inds_with_label()
        else:
            candidate_indices = range(0,len(self.bboxes))
        sub_idx = random.choice(candidate_indices)
        bbox = self.bboxes[sub_idx]
        fn = bbox["filename"]
        return fn, bbox

    def maybe_randomize_idx(self):
            while self.mandatory_label not in self.label_info['labels_this_case']:
                idx =  np.random.randint(0, len(self))
                self.set_bboxes_labels(idx)


    def get_inds_with_label(self):
        labels_per_file = self.label_info['labels_per_file']
        inds_label_status = [self.mandatory_label in labels for labels in labels_per_file]
        indices = self.label_info['file_indices']
        inds_with_label= list(itertools.compress(indices,inds_label_status))
        return inds_with_label

    def randomize_label(self):
        mandatory = np.random.multinomial(1,self.class_ratios,1)
        _,mandatory_label= np.where(mandatory==1)
        return mandatory_label.item()

    def shape_per_id(self,id):
            bb = self.bboxes_per_id[id]
            bb_stats = bb[0]['bbox_stats']
            bb_any = bb_stats[0]['bounding_boxes'][0]
            shape = [sl.stop for sl in bb_any]
            return shape

    def get_label_info(self,case_bboxes):
            indices = []
            labels_per_file = []
            for indx, bb in enumerate(case_bboxes):
                bbox_stats  = bb['bbox_stats']
                labels = [(a['label']) for a in bbox_stats if not a['label']=='all_fg']
                if self.contains_bg(bbox_stats): labels = [0]+labels 
                if len(labels)==0 : labels =[0] # background class only by exclusion
                indices.append(indx)
                labels_per_file.append(labels)
            labels_this_case = list(set(reduce(operator.add,labels_per_file)))
            return {'file_indices':indices,'labels_per_file':labels_per_file, 'labels_this_case': labels_this_case}


        
    @property
    def class_ratios(self):
        """The class_ratios property."""
        return self._class_ratios

    @class_ratios.setter
    def class_ratios(self, raw_ratios):
        denom = reduce(operator.add,raw_ratios)
        self._class_ratios= [x/denom for x in raw_ratios]

    @property
    @maybe_set_property
    def median_shape(self):
            aa = []
            for i in range(len(self)):
                aa.append(self.shape_per_id(i))
            return np.median(aa,0).astype(int)

    @property
    @maybe_set_property
    def parent_folder(self):
            fn , _ = self.get_filename_bbox(0)
            return fn.parent.parent


    @property
    @maybe_set_property
    def dataset_min(self):
        try:
            data_properties= load_dict(self.parent_folder.parent/("resampled_dataset_properties"))
        except: raise FileNotFoundError
        return data_properties['dataset_min']
    
    def contains_bg(self,bbox_stats):
        all_fg_bbox = [bb for bb in bbox_stats if bb['label']=='all_fg'][0]
        bboxes = all_fg_bbox['bounding_boxes']
        if len(bboxes) == 1 : return True
        if bboxes[0]!=bboxes[1]: return True
            

class ImageMaskBBoxDatasetd(ImageMaskBBoxDataset):
        def __getitem__(self, idx):
            self.set_bboxes_labels(idx)
            if self.enforce_ratios == True:
                 self.mandatory_label = self.randomize_label() 
                 self.maybe_randomize_idx()

            filename, bbox = self.get_filename_bbox()
            img,label = self.load_tensors(filename)
            dici={'image':img,'label':label,'bbox':bbox}
            if self.transform is not None:
                dici= self.transform(dici)
          
            return dici

class CropImgMaskd(MapTransform):

    def __init__(self, patch_size,input_dims):
        self.dim = len(patch_size)
        self.patch_halved = [int(x / 2) for x in patch_size]
        self.input_dims=input_dims

    def func(self, x):
        img, label = x
        center = [x / 2 for x in img.shape[-self.dim:]]
        slices = [
            slice(None),
        ] * (self.input_dims-3 ) # batch and channel dims if its a batch otherwise empty
        for ind in range(self.dim):
            source_sz = center[ind]
            target_sz = self.patch_halved[ind]
            if source_sz > target_sz:
                slc = slice(int(source_sz - target_sz), int(source_sz + target_sz))
            else:
                slc = slice(None)
            slices.append(slc)
        img, label = img[slices], label[slices]
        return img, label


# %%
class Affine3D(MapTransform):
    '''
    to-do: verify if nearestneighbour method preserves multiple mask labels
    '''

    def __init__(self,
                 keys,
                 p=0.5,
                 rotate_max=pi / 6,
                 translate_factor=0.0,
                 scale_ranges=[0.75, 1.25],
                 shear: bool = True,
                 allow_missing_keys=False):
        '''
        params:
        scale_ranges: [min,max]
        '''
        super().__init__(keys, allow_missing_keys)
        store_attr('p,rotate_max,translate_factor,scale_ranges,shear')

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
        return d

    def get_mode(self,x):
        dt = x.dtype
        if dt == torch.uint8:
            mode='nearest'
        elif dt==torch.float32 or x.dtype==torch.float16:
            mode ='bilinear'
        return mode,dt

    def func(self, x):
        mode,dt = self.get_mode(x)

        if np.random.rand() < self.p:
            grid = get_affine_grid(x.shape,
                                   shear=self.shear,
                                   scale_ranges=self.scale_ranges,
                                   rotate_max=self.rotate_max,
                                   translate_factor=self.translate_factor,
                                   device=x.device).type(torch.float32)
            x = F.grid_sample(x.type(x.dtype), grid,mode=mode)
        return x.to(dt)
#
class NormaliseClip(Transform):
    def __init__(self,clip_range,mean,std):
        # super().__init__(keys, allow_missing_keys)

        store_attr('clip_range,mean,std')

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) :
            d = self.clipper(data)
            return d

    def clipper(self, img):
        img = torch.clip(img,self.clip_range[0],self.clip_range[1])
        img = standardize(img,self.mean,self.std)
        return img

class NormaliseClipd(MapTransform):
    def __init__(self,keys,clip_range,mean,std,allow_missing_keys=False):
        MapTransform.__init__(self,keys, allow_missing_keys)
        self.N = NormaliseClip(clip_range=clip_range,mean=mean,std=std)

    def __call__(self,d):
        for key in self.key_iterator(d):
            d[key] = self.N(d[key])
        return d



class FillBBoxPatches(Transform):
    """
    Based on size of original image and n_channels output by model, it creates a zerofilled tensor. Then it fills locations of input-bbox with data provided
    """


    def __call__(self,d):
        '''
        d is a dict with keys: 'image','pred','bbox'
        '''

        full= torch.zeros(d['image'].shape)
        bbox = d['bbox']
        full[bbox]=d['pred']
        d['pred']=full
        return d

class MaskLabelRemap2(MapTransform):
    def __init__(self,keys,src_dest_labels:tuple,allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        if isinstance(src_dest_labels,str): src_dest_labels = ast.literal_eval(src_dest_labels)
        self.src_dest_labels=src_dest_labels

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.remapper(d[key])
        return d

    def remapper(self,mask):
            n_classes=len(self.src_dest_labels)
            mask_out = torch.zeros(mask.shape,dtype=mask.dtype)
            mask_tmp = one_hot(mask,n_classes,0)
            mask_reassigned = torch.zeros(mask_tmp.shape,device=mask.device)
            for src_des in self.src_dest_labels:
                src,dest = src_des[0],src_des[1]
                mask_reassigned[dest]+=mask_tmp[src]

            for x in range(n_classes):
                mask_out[torch.isin(mask_reassigned[x],1.0)]=x
            return mask_out

# %%
if __name__ == "__main__":
    from fran.utils.common import *
    P = Project(project_title="lits");
    configs_excel = ConfigMaker(P,raytune=False).config

    train_list, valid_list = P.get_train_val_files(0)
    fldr =Path("/s/fran_storage/datasets/preprocessed/fixed_spacings/lax/spc_080_080_150/") 


    bboxes_fname = fldr/ ("bboxes_info")
    glob_props=load_dict(P.global_properties_filename)


    pp(glob_props)
   
# %%
    train_ds = ImageMaskBBoxDatasetd(
            train_list,
            bboxes_fname,
            [0,0,1],transform=None

        )
    a= train_ds[1]
    a['label'].dtype
    bb ="/home/ub/datasets/preprocessed/lax/patches/spc_080_080_150/dim_192_192_96/images/lits_0ub_2.pt"
    img = torch.load(bb)
    img.dtype
# %%
    scale_ranges=[.75,1.25]
    contrast_ranges=[.7,1.3]
    shift=[-1.0,1.0]
    noise=.1
    src_dest_labels=[[0,2],[1,1],[2,1]]
    tfm_keys=['image','label']
    tfms=Compose([
            MaskLabelRemap2(keys=['label'],src_dest_labels=src_dest_labels),
            EnsureChannelFirstd(keys=tfm_keys,channel_dim='no_channel'),
            NormaliseClipd(keys=['image'],clip_range= glob_props['intensity_clip_range'],mean=glob_props['mean_fg'],std=glob_props['std_fg']),
            RandFlipd(keys=["image", "label"], prob=0.99, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=1.0, spatial_axis=1),
            RandScaleIntensityd(keys="image", factors=scale_ranges, prob=1.0),
            RandGaussianNoised(keys=['image'],prob=0.9,std=noise),
            RandShiftIntensityd(keys="image", offsets=shift, prob=1.0),
            RandAdjustContrastd(['image'],prob=1.0,gamma=contrast_ranges),
            ResizeWithPadOrCropd(keys=['image','label'],source_key='image',spatial_size=[160,160,96]),
    ])
# %%
    Af=Affine3D(tfm_keys)
    A = EnsureChannelFirstd(keys=tfm_keys,channel_dim='no_channel')

# %%
    tfms.insert(1,A)
    Ra=RandFlipd(keys=["image", "label"], prob=1.0, spatial_axis=0)

    E = EnsureChannelFirstd(keys=tfm_keys,channel_dim='no_channel')

    aaa = tfms(a)
    a4 = E(aaa)
# %%
    a5 = Af(a4)
# %%
# %%
    # a = a.permute(*axes)
    ImageMaskViewer([a4['image'][0,0],a4['label'][0,0]])
    ImageMaskViewer([a5['image'][0,0],a5['label'][0,0]])
# %%
    import pywt
    wavelet = pywt.Wavelet('haar')
    d = pywt.wavedec(a, wavelet, mode='zero', level=2)
    dd = torch.tensor(d[0])
    dd = dd.permute(*axes)
    a2 = a.permute(*axes)
    org_shape = a.shape
    a2 = resize_tensor(a,dd.shape,mode='trilinear')
    a3 = resize_tensor(a2,org_shape,mode='trilinear')
    a = a.permute(*axes)
    a3 = a3.permute(*axes)
    a=a.permute(*axes)
    ImageMaskViewer([a,b],data_types=['img','img'])

     
# %%
    from time import time
    start = time()

    a2 = [resize_tensor(a,dd.shape,mode='trilinear') for x in range(100)]
    end = time()
    print(end-start)

# %% [markdown]
    start = time()
    d = [pywt.wavedec(a, wavelet, mode='zero', level=1) for x in range(100)]
    end = time()
    print(end-start)
## Versus torch resize
# %%
    resize_tensor


# Create geometries and projector.
# %%
    a = a.numpy()
    b = b.numpy()
    np.save("/home/ub/code/aug/im_liver.npy",a)
    np.save("/home/ub/code/aug/ma_liver.npy",b)
