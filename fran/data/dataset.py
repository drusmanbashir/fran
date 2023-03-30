# nvidia measure command
# timeout -k 1 2700 nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > results-file.csv
# short version below
# timeout -k 1 2700 nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > results-file.csv
# %%

# from torch.utils.data import DataLoader
# from fastai.data.transforms import DataLoader as DL2
import numpy as np
import operator 
from functools import reduce

import itertools
from functools import partial

from fastai.data.transforms import *
from fran.utils.helpers import *


from fastai.data.transforms import FileGetter
from fran.transforms.spatialtransforms import *
import ipdb

tr = ipdb.set_trace
# path=  proj_default_folders.preprocessing_output_folder
# imgs_folder =  proj_default_folders.preprocessing_output_folder/("images")
# masks_folder=  proj_default_folders.preprocessing_output_folder/("masks")
#
get_img_files = partial(FileGetter(extensions=".npy", folders=["images"]))
get_msk_files = partial(FileGetter(extensions=".npy", folders=["masks"]))
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
            

class ImageMaskBBoxDataset():
    """
    takes a list of case_ids and returns bboxes image and mask
    """

    def __init__(self,proj_defaults, case_ids, bbox_fn , class_ratios:list=None):
        store_attr('proj_defaults')

        """
        class_ratios decide the proportionate guarantee of each class in the output including background. While that class is guaranteed to be present at that frequency, others may still be present if they coexist
        """
        if not class_ratios: 

            self.enforce_ratios = False
        else: 
            assert len(class_ratios) == self.num_classes, "All classes must be represented"
            self.class_ratios = class_ratios
            self.enforce_ratios = True

        print("Loading dataset from BBox file {}".format(bbox_fn))
        bboxes_unsorted = load_dict(bbox_fn)
        self.bboxes_per_id = []

        for cid in case_ids:
            bboxes = [bb for bb in bboxes_unsorted if bb["case_id"] == cid]
            if len(bboxes) == 0:
                print("Missing case id {0} from bboxfile".format(cid))

            bboxes.append(self.get_label_info(bboxes))
            self.bboxes_per_id.append(bboxes)

    def __len__(self):
        return len(self.bboxes_per_id)

    def __getitem__(self, idx):
        self.set_bboxes_labels(idx)
        if self.enforce_ratios == True:
             self.mandatory_label = self.randomize_label() 
             self.maybe_randomize_idx()

        filename, bbox = self.get_filename_bbox()
        img,mask = self.load_tensors(filename)
        return img, mask, bbox

    def load_tensors(self,filename:Path):
        mask = torch.load(filename)
        if isinstance(mask, dict):
            img, mask = mask["img"], mask["mask"]
        else:
            img_folder = filename.parent.parent / ("images")
            img_fn = img_folder / filename.name
            img = torch.load(img_fn)
        return img,mask

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
    def num_classes(self):
        if not hasattr(self,'_num_classes'):
          self._num_classes =len(self.proj_defaults.mask_labels )+1 # +1 for the bg class
        return self._num_classes
        
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
            

# %%
if __name__ == "__main__":
    from fran.utils.common import *
    P = Project(project_title="lits"); proj_defaults= P.proj_summary
    configs_excel = ConfigMaker(proj_defaults.configuration_filename,raytune=False).config

    train_list, valid_list, test_list = get_fold_case_ids(
            fold=configs_excel['metadata']["fold"],
            json_fname=proj_defaults.validation_folds_filename,
        )
    fldr =Path("/home/ub/datasets/preprocessed/lits/patches/spc_100_100_200/dim_220_220_110") 


    bboxes_fname = fldr/ ("bboxes_info")
    dd = load_dict(bboxes_fname)


# %%
    for d in dd:
        print("-----------")
        for stats in d['bbox_stats']:
            pp(stats['label'])
    
# %%
    train_ds = ImageMaskBBoxDataset(
            proj_defaults,
            train_list,
            bboxes_fname,
            [0,1,0]
        )
# %%
    for indx in range(len(train_ds)):
        _,_,c = train_ds[indx]
        pp([a['label'] for a  in c['bbox_stats']])
# %%
