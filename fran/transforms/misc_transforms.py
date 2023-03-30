from functools import partial
import ipdb

tr = ipdb.set_trace

from fastcore.basics import store_attr
from fastcore.transform import ItemTransform

import fran.transforms.intensitytransforms as intensity
import fran.transforms.spatialtransforms as spatial

from fastai.vision.augment import ItemTransform, store_attr



def create_augmentations(after_item_intensity:dict, after_item_spatial:dict):
        intensity_augs = []
        spatial_augs = []
        probabilities_intensity = []
        probabilities_spatial = []
        for key, value in after_item_intensity.items():
            func = getattr(intensity, key)
            out_fnc = partial(func, factor_range=value[0])
            intensity_augs.append(out_fnc)
            probabilities_intensity.append(value[1])

        for key, value in after_item_spatial.items():
            spatial_augs.append(getattr(spatial, key))
            probabilities_spatial.append(value)
        return intensity_augs, spatial_augs

class FilenameFromBBox(ItemTransform):
    def encodes(self,x):
        img,mask,bbox = x
        fname = str(bbox['filename'])
        return img,mask, fname

class Squeeze(ItemTransform):

    def __init__(self, dim):
        store_attr()
    def encodes(self,x):
        outputs =[]
        for tensr in x:
            tensr= tensr.squeeze(self.dim)
            outputs.append(tensr)
        return outputs

    def decodes(self,x):
        outputs =[]
        for tensr in x:
            tensr= tensr.unsqueeze(self.dim)
            outputs.append(tensr)
        return outputs
 
class DropBBoxFromDataset(ItemTransform):
    def encodes(self,x):
        if len(x)==3:
            x= x[:2]
        return x

class BGToMin(ItemTransform):
    def __init__(self,min_val=-0.49):
        store_attr()
    def encodes(self,x):
        img, mask =x
        img[mask<1] =self.min_val
        return img,mask
        return outputs
