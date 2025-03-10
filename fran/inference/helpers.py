import numpy as np 
import torch
import math
def get_sitk_target_size_from_spacings(sitk_array,spacing_dest):
            sz_source , spacing_source = sitk_array.GetSize(), sitk_array.GetSpacing()
            sz_dest,_ = get_scale_factor_from_spacings(sz_source,spacing_source,spacing_dest)
            return sz_dest

def get_scale_factor_from_spacings (sz_source, spacing_source, spacing_dest):
            scale_factor = [a / b for a, b in zip(spacing_source, spacing_dest)]
            sz_dest= [round(a*b )for a,b in zip(sz_source,scale_factor)]
            return sz_dest, scale_factor

def rescale_bbox(scale_factor,bbox):
        bbox_out=[]
        for a,b in zip(scale_factor,bbox):
            bbox_neo = slice(int(b.start*a),int(np.ceil(b.stop*a)),b.step)
            bbox_out.append(bbox_neo)
        return tuple(bbox_out)


def apply_threshold(input_img,threshold):
    input_img[input_img<threshold]=0
    input_img[input_img>=threshold]=1
    return input_img

def get_amount_to_pad(img_shape, patch_size):


        pad_deficits = np.maximum(0, np.array(patch_size) - img_shape)
        padding = (math.floor(pad_deficits[0] / 2),
                   math.ceil(pad_deficits[0] / 2)), (math.floor(pad_deficits[1] / 2),
                                                     math.ceil(pad_deficits[1] / 2)), (math.floor(pad_deficits[2] / 2),
                                                                                       math.ceil(pad_deficits[2] / 2))
        return padding




def get_scale_factor_from_spacings (sz_source, spacing_source, spacing_dest):
            scale_factor = [a / b for a, b in zip(spacing_source, spacing_dest)]
            sz_dest= [round(a*b )for a,b in zip(sz_source,scale_factor)]
            return sz_dest, scale_factor


