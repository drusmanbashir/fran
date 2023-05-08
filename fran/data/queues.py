from fastai.data.transforms import delegates
from torch.utils.data import DataLoader as DL_torch
import torchio as tio
from torchio.data import Subject, subject
from fran.data.dataset import ImageMaskBBoxDataset
#
from fran.utils.helpers import get_fold_case_ids
#


# %%
def reader_tensor_dict(path:Path):
    tnsr = torch.load(path)
    dic={}
    for key,val in tnsr.items():
        dic.update({key:tio.ScalarImage(tensor=val.unsqueeze(0))})
    s = Subject(dic)
    im1 = tio.ScalarImage(tensor)

# %%
    
# %%
# class SubjectsDatasetFromFastaiDataset(ImageMaskBBoxDataset)
#
#     def create_subjects_dataset(self):
#         for bbox in self.bboxes_per_id:
#
#         self.subjects_list = [
#         Subject(images=tio.ScalarImage())
#     ]
#         fn =bbox['filename']
#         tensr = torch.load(fn)
#         img,mask = tensr['img'],tensr['mask']
#         return img,mask,bbox
#
#
#
#     s = Subject(images=tio.ScalarImage(tensor = a[0].unsqueeze(0)),mask=tio.LabelMap(tensor = a[1].unsqueeze(0)))
#     sampler = tio.data.UniformSampler([1,128,128])
#     ds = tio.SubjectsDataset(10*[s])
# %%
#
# class QueueWrapper(tio.Queue):
#
#     @delegates(tio.Queue)
#     def __init__(self,fastai_ds):
#
#         subject_dict = {'image':tio.ScalarImage,'mask':tio.LabelMap,'bbox':None}
#     s = Subject(images=tio.ScalarImage(tensor = a[0].unsqueeze(0)),mask=tio.LabelMap(tensor = a[1].unsqueeze(0)))
#     sampler = tio.data.UniformSampler([1,128,128])
#     ds = tio.SubjectsDataset(10*[s])
#     q = tio.Queue(
#
#         ds,
#         20,
#         10,
#         sampler,
#         num_workers=4
#     )
# class QueueImageMaskBBox(QueueWrapper):
#     @delegates(QueueWrapper)
#     def __init__(self, fastai_ds, *args,**kwargs):
#         subject_dict = {'image':tio.ScalarImage,'mask':tio.LabelMap,'bbox':None}
#         super().__init__(subject_dict, fastai_ds, *args,**kwargs)
# %%

# %%
if __name__ == "__main__":

    dl = DataLoader(q)
    iteri = iter(dl)
    a = next(iteri)
    
    from fran.utils.fileio import *
    common_vars_filename=os.environ['FRAN_COMMON_PATHS']
    P = Project(project_title="lits"); proj_defaults= P
    images_folder = proj_defaults.stage2_folder/("60_164_164")
    bboxes_fn =images_folder/"bboxes_info"  
    bboxes = load_dict(bboxes_fn)
    fold = 0
# %%
