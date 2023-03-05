import SimpleITK as sitk
from fastcore.transform import Transform
import numpy as np 
import torch
import cc3d
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

def mask_from_predictions(predictions_array, connected_components, threshold=0.95 ):
            '''
            params: predictionary_array shape C,D,W,H
            '''
            
            assert (len(predictions_array.shape)==4), "Input array has to be 4D (C,D,W,H). Found instead {}D".format(len(predictions_array))
            if (predictions_array.shape[0])> 1: # i.e., multi-channel softmax output_tensor
                mask = torch.argmax(predictions_array,dim=0)
            else: 
                mask = apply_threshold(predictions_array, threshold)
                mask = mask.squeeze(0)
            mask = np.array(mask, dtype=np.uint16)
            mask2 = cc3d.largest_k(mask,k=connected_components,connectivity=26,delta=0,return_N=False)
            mask[mask>0]=1
            return mask
# %%
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


