# %%
import itertools, random
from fran.architectures.gan import create_augmentations
from fran.transforms.spatialtransforms import *
from monai.data.grid_dataset import PatchDataset
from monai.transforms.croppad.array import RandCropByLabelClasses, RandSpatialCropSamples
from fran.data.dataset import *
from monai.data import GridPatchDataset,  PatchIter, Dataset
from monai.transforms import RandShiftIntensity, RandSpatialCrop

from fran.transforms.misc_transforms import DropBBoxFromDataset

from fran.transforms.monaitransforms import RandCropImgMaskByLabelClasses, RandomCropped

class PermuteImageMask(ItemTransform):

    def __init__(self,p=0.3):
        self.p=p
    def encodes(self,x):
        if np.random.rand() < self.p:
            img,mask=x
            sequence =(0,)+ tuple(np.random.choice([1,2],size=2,replace=False)   ) #if dim0 is different, this will make pblms
            img_permuted,mask_permuted = torch.permute(img,dims=sequence),torch.permute(mask,dims=sequence)
            return img_permuted,mask_permuted
        else: return x


# %%
if __name__ == "__main__":
    common_paths_filename=os.environ['FRAN_COMMON_PATHS']
    P = Project(project_title="lits"); proj_defaults= P.proj_summary
    spacings = [1,1,1]
    src_patch_size = [192,192,192]
    images_folder = proj_defaults.patches_folder/("spc_100_100_100")/("dim_{0}_{0}_{2}".format(*src_patch_size))/("images")
    bboxes_fn =images_folder.parent/"bboxes_info"  
    bboxes = load_dict(bboxes_fn)
    fold = 0

    json_fname=proj_defaults.validation_folds_filename

    imgs =list((proj_defaults.raw_data_folder/("images")).glob("*"))
    masks =list((proj_defaults.raw_data_folder/("masks")).glob("*"))
    img_fn = imgs[0]
    mask_fn = masks[0]
    train_ids,val_ids,_ =  get_fold_case_ids(fold=0, json_fname=proj_defaults.validation_folds_filename)
    train_ds = ImageMaskBBoxDataset(proj_defaults,train_ids, bboxes_fn,[0,1,2])
    valid_ds = ImageMaskBBoxDataset(proj_defaults,val_ids, bboxes_fn,[0,1,2])

    train_list_w,valid_list_w,_ = get_train_valid_test_lists_from_json(project_title=proj_defaults.project_title,fold=fold,image_folder =images_folder, json_fname=json_fname)
    
# %%
    for x in range(len(train_ds)):
        a,b,c = train_ds[x]

# %%
    aa = train_ds.median_shape
# %%
    bb = train_ds.bboxes_per_id[0]

# %%
    mask_labs = proj_defaults.mask_labels
    tissue_label_dict = {entry['name']:entry['label'] for entry in mask_labs  }
# %%
    bboxes = train_ds.bboxes_per_id
    excluded =['all_fg']
# %%

# %%
    for n in range(len(self.bboxes_per_id)):
        case_bboxes  = bboxes[n]
        case_id = case_bboxes[0]['case_id']
        indices = []
        labels_per_file = []
        for indx, bb in enumerate(case_bboxes):
            bbox_stats  = bb['bbox_stats']

            tissues= [bbox_stat['tissue_type'] for bbox_stat in bbox_stats if bbox_stat['tissue_type'] not in excluded]
            labels = [tissue_label_dict[tissue] for tissue in tissues]
            if self.contains_bg(bbox_stats): labels = [0]+labels 
            if len(labels)==0 : labels =[0] # background class only by exclusion
            fn = bb['filename']
            indices.append(indx)
            labels_per_file.append(labels)
        labels_this_case = list(set(reduce(operator.add,labels_per_file)))
        label_info ={'case_id':case_id,'file_indices':indices,'labels_per_file':labels_per_file, 'labels_this_case': labels_this_case}
        truths = [labels_this_case==a for a in labels_per_file]
# %%

# %%
# %%

# %%
    candidate_indices = [ensure_label in labels for labels in labels_per_file]
    indices = label_info['indices']
    inds_with_label= list(itertools.compress(indices,candidate_indices))
    sub_indx = random.choice(inds_with_label)
# %%


# %%
    after_item_intensity=    {'brightness': [[0.7, 1.3], 0.1],
     'shift': [[-0.2, 0.2], 0.1],
     'noise': [[0, 0.1], 0.1],
     'brightness': [[0.7, 1.5], 0.01],
     'contrast': [[0.7, 1.3], 0.1]}
    after_item_spatial = {'flip_random':0.5}
    intensity_augs,spatial_augs = create_augmentations(after_item_intensity,after_item_spatial)

    probabilities_intensity,probabilities_spatial = 0.1,0.5
    after_item_intensity = TrainingAugmentations(augs=intensity_augs, p=probabilities_intensity)
    after_item_spatial = TrainingAugmentations(augs=spatial_augs, p=probabilities_spatial)
# %%
    sampler = RandSpatialCropSamples(roi_size=[128,128,128],random_size=False,random_center=True,num_samples=3) # %% for i in range(self.__len__()):
        #     tr()
        #     a,b,c = self[i]
        #     aa.append(a.shape)
        # self._median_shape = np.median(aa,0)

# %%


# %%
    patch_size=[128,128,128]

    crop= RandomCropped(spatial_size=patch_size,ratios=[1,1,2],num_classes=3,ds_median_shape=train_ds.median_shape)

# %%

    dl = TfmdDL(train_ds,
                after_item=
                Pipeline([
                DropBBoxFromDataset(),

                PermuteImageMask(p=0.3),
                after_item_spatial,
                PadDeficitImgMask(patch_size,3),
                Unsqueeze,
                crop

                ]),
                before_batch=
                [

    ],
                bs = 3,

                )
# %%
    for i, b  in enumerate(dl):
        print(b[0].shape, b[1].shape)

# %%
    n=14
    ImageMaskViewer([aa[0][n,0],aa[1][n, 0]])
    
# %%
    from itertools import *
# %%
    for iti in (chain('abc','cde')):
        print(iti)
# %%

