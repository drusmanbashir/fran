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

def img_metadata_collated(batch):
        imgs=[]
        org_sizes=[]
        for i , item in enumerate(batch):
            imgs.append(item['image'])
            org_sizes.append(item['org_size'])
        output = {'image':torch.stack(imgs,0),'org_size':org_sizes}
        return output





