from fastai.torch_core import to_device
import torch
from fastai.data.load import *
from fastai.callback.fp16 import TfmdDL
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter,_DatasetKind
_loaders = (_MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter)
import ipdb
tr = ipdb.set_trace


class TfmdDLKeepBBox(TfmdDL):

    def create_batch(self, batch):
        imgs=[]
        masks= []
        bboxes=[]
        for i , item in enumerate(batch):
            imgs.append(item[0])
            masks.append(item[1])
            bboxes.append(item[2])
        return torch.stack(imgs,0),torch.stack(masks,0),bboxes

        # BELOW FUNCTION DOESNT WORK
    # def __iter__(self):
    #     self.randomize()
    #     self.before_iter()
    #     self.__idxs=self.get_idxs() # called in context of main process (not workers/subprocesses)
    #     for b in _loaders[self.fake_l.num_workers==0](self.fake_l):
    #         # pin_memory causes tuples to be converted to lists, so convert them back to tuples
    #         tr()
    #         if self.pin_memory and type(b) == list: b = tuple(b)
    #         if self.device is not None: 
    #             b,bboxes = b[:-1],b[-1]
    #             b = to_device(b, self.device)
    #             b = *b,bboxes
    #         yield self.after_batch(b)
    #     self.after_iter()
    #     if hasattr(self, 'it'): del(self.it)
    #
# %%



