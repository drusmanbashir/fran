# from fastai.data.core import TfmdDL
import torch
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter
_loaders = (_MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter)
import ipdb
tr = ipdb.set_trace

def img_mask_bbox_collate( batch):
        imgs=[]
        masks= []
        bboxes=[]
        for i , item in enumerate(batch):
            imgs.append(item[0])
            masks.append(item[1])
            bboxes.append(item[2])
        return torch.stack(imgs,0),torch.stack(masks,0),bboxes

def img_mask_bbox_collated( batch):
        imgs=[]
        labels= []
        bboxes=[]
        for i , item in enumerate(batch):
            imgs.append(item['image'])
            labels.append(item['label'])
            bboxes.append(item['bbox'])
        output = {'image':torch.stack(imgs,0),'label':torch.stack(labels,0),'bbox':bboxes}
        return output

def img_mask_metadata_lists_collated(batch):
        images=[]
        masks=[]
        images_meta=[]
        masks_meta=[]
        for i , item in enumerate(batch):
            images.append(item['image'])
            images_meta.append(item['image'].meta)
            masks.append(item['mask'])
            masks_meta.append(item['mask'].meta)
        output = {'image': images, 'mask':masks, 'images_meta':images_meta, 'masks_meta':masks_meta}
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
#         masks= []
#         bboxes=[]
#         for i , item in enumerate(batch):
#             imgs.append(item[0])
#             masks.append(item[1])
#             bboxes.append(item[2])
#         return torch.stack(imgs,0),torch.stack(masks,0),bboxes
#
#
#
#
