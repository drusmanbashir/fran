# from fastai.data.core import TfmdDL
import torch
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter
_loaders = (_MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter)
import ipdb
tr = ipdb.set_trace

def img_lm_bbox_collate( batch):
        imgs=[]
        lms= []
        bboxes=[]
        for i , item in enumerate(batch):
            imgs.append(item[0])
            lms.append(item[1])
            bboxes.append(item[2])
        return torch.stack(imgs,0),torch.stack(lms,0),bboxes

def img_lm_bbox_collated( batch):
        imgs=[]
        lms= []
        bboxes=[]
        for i , item in enumerate(batch):
            imgs.append(item['image'])
            lms.append(item['lm'])
            bboxes.append(item['bbox'])
        output = {'image':torch.stack(imgs,0),'lm':torch.stack(lms,0),'bbox':bboxes}
        return output

def img_lm_metadata_lists_collated(batch):
        images=[]
        lms=[]
        images_meta=[]
        lms_meta=[]
        for i , item in enumerate(batch):
            images.append(item['image'])
            images_meta.append(item['image'].meta)
            lms.append(item['lm'])
            lms_meta.append(item['lm'].meta)
        output = {'image': images, 'lm':lms, 'images_meta':images_meta, 'lms_meta':lms_meta}
        return output

def dict_list_collated(keys):
    def _inner(batch):
        output ={key:[] for key in keys}
        for i , item in enumerate(batch):
            for key in keys:
                output[key].append(item[key])
        return output
    return _inner



#
# class TfmdDLKeepBBox(TfmdDL):
#
#     def create_batch(self, batch):
#         imgs=[]
#         lms= []
#         bboxes=[]
#         for i , item in enumerate(batch):
#             imgs.append(item[0])
#             lms.append(item[1])
#             bboxes.append(item[2])
#         return torch.stack(imgs,0),torch.stack(lms,0),bboxes
#
#
#
#
