# %%
from functools import partial
from fastcore.transform import Pipeline
import math
from fran.utils.image_utils import get_bbox_from_mask
from fran.utils.helpers import *
from fran.utils.fileio import *
import ast


import skimage.transform as tf
import ipdb
import torch.nn.functional as F
from torch import sin, cos, pi

tr = ipdb.set_trace

# %%
from fran.transforms.basetransforms import *
    
class PermuteImageMask(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys,
        prob: float = 1,
        do_transform: bool = True,
    ):
        MapTransform.__init__(self, keys, False)
        RandomizableTransform.__init__(self, prob)
        store_attr()

    def func(self,x):
        if np.random.rand() < self.p:
            img,mask=x
            sequence =(0,)+ tuple(np.random.choice([1,2],size=2,replace=False)   ) #if dim0 is different, this will make pblms
            img_permuted,mask_permuted = torch.permute(img,dims=sequence),torch.permute(mask,dims=sequence)
            return img_permuted,mask_permuted
        else: return x




def one_hot(tnsr,classes, axis, fnc=torch.stack):
        '''
        takes tnsr, splits it into n_labels, then either stacks the results list on axis (creating a new axis), or concatenates it
        '''
        
        tnsr  = [torch.where(tnsr ==c, 1, 0) for c in range(classes)]
        tnsr = fnc(tnsr,axis)
        return tnsr


def rotate90_np(x):
    '''
    swaps last two dims of an np array, thus rotating by 90
    '''
    ndim = len(x[0].shape)
    first_dims = np.arange(ndim - 2)
    last_dims = np.array([ndim - 1, ndim - 2])
    final_perm = np.concatenate([first_dims, last_dims])
    x = [np.transpose(arr, final_perm)
         for arr in x]  # the z-dimension (typically different from other two) is not touched
    return x


def flip_horizontal_np(x):
    flip_axis = len(x[0].shape) - 1
    x = [np.flip(arr, axis=flip_axis) for arr in x]
    return x


def mirror_torch(x):
    ndim = len(x[0].shape)
    flip_axes = [ndim - 3, ndim - 1]
    x = [torch.flip(arr, flip_axes) for arr in x]
    return x


def flip_vertical_np(x):
    flip_axis = len(x[0].shape) - 2
    x = [np.flip(arr, axis=flip_axis) for arr in x]
    return x


def flip_random_np(x):
    '''
    3d array, can be horizontal flip, vertical flip or craniocaudal
    '''

    dim = random.randint(a=0, b=2)
    x = [np.flip(arr, axis=dim) for arr in x]
    return x


def flip_random(x):
    '''
    3d array, can be horizontal flip, vertical flip or craniocaudal
    '''

    dims = np.random.choice(size=2, a=[0, 1, 2], replace=False)
    x = [torch.flip(arr, dims=[dims[0], dims[1]]) for arr in x]
    return x

def get_bbox_size(bbox): return [int(a.stop - a.start) for a in bbox]

def get_slices_from_centroid(centroid, bbox, random_shift, input_shape, target_patch_size, stride=1):
    if isinstance(stride, int): stride = [stride] * len(input_shape)

    bbox_size = get_bbox_size(bbox) 
    # input_shape, target_patch_size,centroid = [multiply_lists(arr, stride) for arr in [input_shape,target_patch_size,centroid]]
    target_patch_size= multiply_lists(target_patch_size,stride)

    flexibility_eitherside = np.array([np.maximum(0, (a - b) / 2) for a, b in zip(bbox_size, target_patch_size)])
    jitter_max = np.array([random_shift * a for a in target_patch_size])
    flexibility_plus_jitter = flexibility_eitherside + jitter_max
    shift_amount = [np.random.randint(-a, +a) if a>10 else 0 for a in flexibility_plus_jitter ]

    center_moved = [int(a + b) for a, b in zip(centroid, shift_amount)]

    patch_halved = [x / 2 for x in target_patch_size]
    slc_start = np.maximum(0,center_moved - np.floor(patch_halved).astype(np.int32))
    slc_stop = np.minimum(input_shape,center_moved + np.ceil(patch_halved).astype(np.int32))
    # shift_back = np.minimum(0, input_shape - slc_stop)
    # shift_forward = np.minimum(0, slc_start)
    # shift_final = shift_back - shift_forward
    # slc_start, slc_stop = slc_start + shift_final, slc_stop + shift_final
    slices = slices_from_lists(slc_start, slc_stop, stride)
    return slices

class AffineTrainingTransform3D(KeepBBoxTransform):
    '''
    to-do: verify if nearestneighbour method preserves multiple mask labels
    '''

    def __init__(self,
                 p=0.5,
                 rotate_max=pi / 6,
                 translate_factor=0.0,
                 scale_ranges=[0.75, 1.25],
                 shear: bool = True):
        '''
        params:
        scale_ranges: [min,max]
        '''
        store_attr()

    def func(self, x):
        img, mask = x
        if np.random.rand() < self.p:
            grid = get_affine_grid(img.shape,
                                   shear=self.shear,
                                   scale_ranges=self.scale_ranges,
                                   rotate_max=self.rotate_max,
                                   translate_factor=self.translate_factor,
                                   device=img.device).type(img.dtype)
            img = F.grid_sample(img, grid)
            mask = F.grid_sample(mask.type(img.dtype), grid, mode='nearest')
            return img, mask.to(torch.uint8)
        return img, mask
#
# def expand_lesion(lesion_start_ind,lesion_end_ind,all_locs,expand_factor=0.2):
#         lesion_start_loc = all_locs[lesion_start_ind]
#         lesion_end_loc = all_locs[lesion_end_ind-1]
#         lesion_span = lesion_end_ind-lesion_start_ind
#         lesion_to_total_frac = lesion_span/all_locs.numel()
#         if lesion_to_total_frac>0.5:
#             return all_locs
#         if lesion_to_total_frac> 0.4:
#             expand_factor = np.minimum(0.2,expand_factor)
#         if lesion_to_total_frac> 0.3:
#             expand_factor = np.minimum(0.3,expand_factor)
#         if lesion_to_total_frac> 0.2:
#             expand_factor = np.minimum(0.4,expand_factor)
#         expand_by = int(expand_factor*lesion_span)
#         lesion_span_new= int(lesion_span+ np.minimum(expand_by,lesion_start_ind)+np.minimum(all_locs.numel()-lesion_end_loc,expand_by))
#         smooth_out_zone=expand_by
#         smooth_out_before_ind= int(lesion_start_ind-expand_by-smooth_out_zone)
#         truncate_before = int(np.minimum(lesion_start_ind-expand_by-smooth_out_zone,0))
#         truncate_after = np.minimum(0,all_locs.numel()-5-(lesion_end_ind + expand_by+smooth_out_zone))# shorten the after_zone if it exceeds total image dim
#         smooth_out_after_ind = (int(lesion_end_ind + expand_by+smooth_out_zone+truncate_after))
#         smooth_out_before_loc = all_locs[smooth_out_before_ind]
#         smooth_out_after_loc = all_locs[smooth_out_after_ind-1]
#
#         smooth_out_before_zone = torch.linspace(smooth_out_before_loc,lesion_start_loc,np.maximum(0,smooth_out_zone+1+truncate_before))
#         smooth_out_after_zone = torch.linspace(lesion_end_loc,smooth_out_after_loc,smooth_out_zone+1+truncate_after)
#         lesion_new_locs = torch.linspace(lesion_start_loc,lesion_end_loc,lesion_span_new)
#         lesion_segment_new = torch.cat([smooth_out_before_zone[:-1],lesion_new_locs,smooth_out_after_zone[1:]])
#         all_locs_modified = all_locs.clone()
#         all_locs_modified[smooth_out_before_ind:smooth_out_after_ind]= lesion_segment_new
#         return all_locs_modified
#
#

def expand_lesion(lesion_start_ind,lesion_end_ind,all_locs,expand_factor=0.2):

    num_inds = len(all_locs)
    lesion_start_x = all_locs[lesion_start_ind]
    lesion_end_x = all_locs[lesion_end_ind-1]
    lesion_span = lesion_end_ind-lesion_start_ind
    lesion_end_x-lesion_start_x
    lesion_center = (lesion_end_x+lesion_start_x)/2
    lesion_span/len(all_locs)
    lesion_locs_shifted = [lesion_start_x-lesion_center, lesion_end_x-lesion_center]
    lesion_start_y = expand_factor*lesion_locs_shifted[0]+lesion_center
    lesion_end_y = expand_factor*lesion_locs_shifted[1]+lesion_center
    smooth_before_ind= np.maximum(0,int(lesion_start_ind-lesion_span))
    smooth_before_x = all_locs[smooth_before_ind]
    smooth_after_ind = np.minimum(num_inds, int(lesion_end_ind+lesion_span))
    smooth_after_x = all_locs[smooth_after_ind]
    np.minimum(lesion_span,lesion_start_ind)
    np.minimum(lesion_span,lesion_start_ind)
    [-1.0,smooth_before_x,lesion_start_x,lesion_end_x,smooth_after_x,1.0]
    seg_y_lims = [-1,smooth_before_x,lesion_start_y,lesion_end_y,smooth_after_x,1.0]
    seg_lengths=[smooth_before_ind,lesion_start_ind-smooth_before_ind, lesion_end_ind-lesion_start_ind,smooth_after_ind-lesion_end_ind,num_inds-smooth_after_ind]
    
    y0 = torch.linspace(seg_y_lims[0],seg_y_lims[1],int(seg_lengths[0]+1))[:-1]
    y1 =  torch.linspace(seg_y_lims[1],seg_y_lims[2],int(seg_lengths[1]+1))[:-1]
    y2 =  torch.linspace(seg_y_lims[2],seg_y_lims[3],int(seg_lengths[2]+1))[:-1]
    y3 =   torch.linspace(seg_y_lims[3],seg_y_lims[4],int(seg_lengths[3]+1))[:-1]
    y4 =   torch.linspace(seg_y_lims[4],seg_y_lims[5],int(seg_lengths[4]))
    y = torch.cat([y0,y1,y2,y3,y4])
    return y


class GrowTumour(ItemTransform):
            def __init__(self, p = 0.3,grow_max=0.5):
                store_attr()
            def encodes(self,x):
                if np.random.uniform() > self.p:
                    return x
                else:
                    img,mask,bboxes = x
                    expand_factor = np.minimum(np.random.uniform(),self.grow_max)
                    print(expand_factor)
                    bbox_all = bboxes['bbox_stats']
                    try:
                        tumour_bb = [a for a in bbox_all if a['tissue_type']=='tumour'][0]['bounding_boxes'][1]
                    except IndexError:
                        return img,mask,bboxes
                    lins=[]
                    for sh in img.shape:
                        lins.append(np.linspace(-1,1,sh))
                    lins_out = []
                    for slices,linss in zip(tumour_bb,lins):
                        lins_out.append (expand_lesion(slices.start,slices.stop,linss, expand_factor=expand_factor))
                    x ,y, z= torch.meshgrid(*lins_out)
                    expanded_mesh = torch.stack([z,y,x],3)
                    expanded_mesh.unsqueeze_(0)
                    img_= img.unsqueeze(0).unsqueeze(0)
                    img_warped= F.grid_sample(img_,expanded_mesh)
                    mask_ = mask.unsqueeze(0).unsqueeze(0)

                    mask_warped= F.grid_sample(mask_.float(),expanded_mesh)
                    img,mask = img_warped.squeeze(0).squeeze(0),mask.squeeze(0).squeeze(0)
                    return img_warped,mask_warped.to(torch.uint8),bboxes,expanded_mesh






class CropImgMask(KeepBBoxTransform):

    def __init__(self, patch_size,input_dims):
        self.dim = len(patch_size)
        self.patch_halved = [int(x / 2) for x in patch_size]
        self.input_dims=input_dims

    def func(self, x):
        img, mask = x
        center = [x / 2 for x in img.shape[-self.dim:]]
        slices = [
            slice(None),
        ] * (self.input_dims-3 ) # batch and channel dims if its a batch otherwise empty
        for ind in range(self.dim):
            source_sz = center[ind]
            target_sz = self.patch_halved[ind]
            if source_sz > target_sz:
                slc = slice(int(source_sz - target_sz), int(source_sz + target_sz))
            else:
                slc = slice(None)
            slices.append(slc)
        img, mask = img[slices], mask[slices]
        return img, mask

class CropExtra(ItemTransform):
    def __init__(self,patch_size):

        store_attr()
    def encodes(self,x):
        img,mask = x
        if list(img.shape)>self.patch_size:
            img = img[:self.patch_size[0],:self.patch_size[1],:self.patch_size[2]]
            mask = mask[:self.patch_size[0],:self.patch_size[1],:self.patch_size[2]]
        return img,mask


class PadDeficitImgMask(KeepBBoxTransform):
    order = 0
    def __init__(self, patch_size, input_dims, pad_values:Union[list,tuple,int]=[0,0], mode='constant', return_padding_array=False):
        '''
        pad_values is a tuple/list 0 entry for img, 1 entry for mas
        '''
        
        if isinstance (pad_values,int):
            pad_values =[pad_values,int(pad_values)]
        assert isinstance (pad_values[1],int), "Provide integer pad value for the mask"
        self.first_dims = input_dims - len(patch_size)
        self.last_dims = input_dims - self.first_dims
        self.patch_size = np.array(patch_size)
        self.pad_values = pad_values
        self.return_padding_array = return_padding_array

    def func(self, x):
        pad_deficits = []
        if any(self.patch_size - x[0].shape[-self.last_dims:]):
            difference = np.maximum(0, self.patch_size - x[0].shape[-self.last_dims:])
            for diff in difference:
                pad_deficits += [math.ceil(diff / 2), math.floor(diff / 2)]
            pad_deficits = pad_deficits[-1::-1]  # torch expects flipped backward
            pad_deficits+=[0,]*self.first_dims
            padded_arrays = []
            for arr,pv in zip(x, self.pad_values):
                arr = F.pad(arr, pad_deficits,value=pv)
                padded_arrays.append(arr)
            x = padded_arrays
        if self.return_padding_array == True:
            x.append(pad_deficits)
        return x


def crop_to_bbox(arr, bbox, crop_axes, crop_padding=0., stride=[1, 1, 1]):
    '''
    param arr: torch tensor or np array to be cropped 
    param bbox: Bounding box (3D only supported)
    param crop_axes:  any combination of 'xyz' may be used (e.g., 'xz' will crop in x and z axes)
    param crop_padding: add crop_padding [0,1] fraction to all the planes of cropping. 
    param stride: stride in each plane
    '''
    assert len(arr.shape) == 3, "only supports 3d images"
    bbox_extra_pct = [int((bbox[i][1] - bbox[i][0]) * crop_padding / 2) for i in range(len(bbox))]
    bbox_mod = [[
        np.maximum(0, bbox[j][0] - bbox_extra_pct[j]),
        np.minimum(bbox[j][1] + bbox_extra_pct[j], arr.shape[j])
    ] for j in range(arr.ndim)]
    slices = []
    for dim, axis in zip([0, 1, 2], ['z', 'y', 'x']):  # tensors are opposite arrranged to numpy
        if axis in crop_axes:
            slices.append(slice(bbox_mod[dim][0], bbox_mod[dim][1], stride[dim]))
        else:
            slices.append(slice(0, arr.shape[dim], stride[dim]))
    return arr[tuple(slices)]


def warp(x):
    img, mask = x
    scale = np.random.uniform(low=0.5, high=1.5, size=3)
    output_shape = (scale * np.array(img.shape)).astype(int)
    coords0, coords1, coords2 = np.mgrid[:output_shape[0], :output_shape[1], :output_shape[2]]
    coords = np.array([coords0, coords1, coords2])
    for n in range(len(img.shape)):
        coords[n] = (coords[n] + 0.5) / scale[n] - 0.5
    img = tf.warp(img, coords)
    mask = tf.warp(mask, coords, order=0)
    return img, mask


class Subsample(ItemTransform):  # TRAINING TRANSFORM
    # Use for any number of dims as long as sample factor matches
    def __init__(self, sample_factor=[1, 2, 2], dim=2):
        store_attr()

    def encodes(self, x):
        img, mask = x
        if len(img.shape) != len(self.sample_factor): print("Sample factor shape does not match that of img")
        slices = []
        for dim, stride in zip(img.shape, self.sample_factor):
            slices.append(slice(0, dim, stride))
        return img[tuple(slices)], mask[tuple(slices)]

    def decodes(self, x):
        img, mask = x
        upsample_factor = self.sample_factor[-self.dim:]
        img = F.interpolate(img, scale_factor=upsample_factor, mode='trilinear')
        mask = F.interpolate(mask, scale_factor=upsample_factor, mode='nearest')
        return img, mask


def get_random_shear_matrix(scale=torch.rand(1) * .5 + 0.5, device='cpu'):
    shear_matrix = torch.eye(3, 3, device=device)
    shear_index = torch.randperm(3)[:2]
    shear_matrix[shear_index[0], shear_index[1]] = scale
    return shear_matrix


def get_random_rotation3d(angle_max=pi / 6, device='cpu'):
    angles = torch.rand(3) * angle_max
    alpha, beta, gamma = angles[0], angles[1], angles[2]
    rot_3d = torch.tensor(
        [[
            cos(beta) * cos(gamma),
            sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma),
            cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)
        ],
         [
             cos(beta) * sin(gamma),
             sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
             cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma)
         ], [-sin(beta), sin(alpha) * cos(beta), cos(alpha) * cos(beta)]],
        device=device)

    return rot_3d


def get_affine_grid(input_shape,
                    shear=True,
                    scale_ranges=[0.75, 1.25],
                    rotate_max=pi / 6,
                    translate_factor=0.0,
                    device='cpu'):
    output_shape = torch.tensor(input_shape)
    output_shape_last3d = torch.tensor(input_shape[-3:])

    bs = int(output_shape[0])
    scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
    output_shape[-3:] = (output_shape_last3d * scale).int()
    theta = torch.zeros(bs, 3, 4, device=device)
    translate = (torch.rand(3,device=device)-0.5)*translate_factor
    theta[:, :, 3] = translate
    if shear:
        shear_matrices = torch.stack([get_random_shear_matrix(device=device) for b in range(bs)])
    else:
        shear_matrices = torch.stack([torch.eye(3, 3, device=device) for x in range(bs)])
    if rotate_max > 0:
        rotation_matrices = torch.stack([get_random_rotation3d(angle_max=rotate_max, device=device) for b in range(bs)],
                                        dim=0)
    else:
        rotation_matrices = torch.stack([torch.eye(3, 3, device=device) for x in range(bs)])
    final_transform = shear_matrices @ rotation_matrices
    theta[:, :, :3] = final_transform
    grid = F.affine_grid(theta, tuple(output_shape))
    return grid


class ToTensorImageMask(ItemTransform):
    '''
    
    numpy image,mask to tensor
    '''
    

    def __init__(self, img_dtype=torch.float, mask_dtype=torch.uint8):
        store_attr()

    def encodes(self, x):
        img, mask = x

        img = torch.tensor(img.copy(), dtype=self.img_dtype)
        mask = torch.tensor(mask.copy(), dtype=self.mask_dtype)
        return img, mask


#


def get_bboxes_matching_labels(labels_list, bboxes):
    bboxes_output = []
    for label in labels_list:
        bbox = [bbox for bbox in bboxes if bbox['tissue_type'] == label]
        if len(bbox) > 0:
            bboxes_output.append(bbox[0])
    return bboxes_output


class WholeImageBinaryMask_skimage(ItemTransform):
    '''
    resizes entire image and mask. Beware and do not pass mask as any other than binary
    '''

    def __init__(self, output_size):
        store_attr()

    def encodes(self, x):
        img, mask = x[0], x[1]
        img = tf.resize(img, self.output_size, preserve_range=True)
        mask = tf.resize(mask.astype(bool), self.output_size)
        mask = mask.astype(np.uint8)
        return img, mask

class WholeImageBinaryMask(ItemTransform):
    '''
    takes 3d image/mask, converts them to tensors and then adds channel dimension before sending out
    resizes entire image and mask. Beware and do not pass mask as any other than binary
    '''

    def __init__(self, output_size):
        store_attr()

    def encodes(self, x):
        output_img = []
        for im in [x[0], x[1]]:
            mode = 'nearest' if 'int' in str(im.dtype) else 'trilinear'
            im = im.unsqueeze(0).unsqueeze(0)
            im = F.interpolate(im, self.output_size, mode=mode)
            im = im.squeeze(0).squeeze(0)
            output_img.append(im)
        return output_img

class PermuteImageMaskBBox(ItemTransform):
    def __init__(self,p=0.5):
        self.p=p
    def encodes(self,x):
        if np.random.rand() < self.p:
            sequence = tuple(np.random.choice([0,1,2],size=3,replace=False)   )
            img,mask,bbox_info=x
            bboxes ,centroids = bbox_info['bboxes'],bbox_info['centroids']
            bboxes_new=[]
            for bbox in bboxes:
                bboxes_new.append([bbox[sequence[0]],bbox[sequence[1]],bbox[sequence[2]]])
            centroids_new = np.array([centroids[:,sequence[0]],centroids[:,sequence[1]],centroids[:,sequence[2]]]).transpose()
            img_permuted,mask_permuted = torch.permute(img,dims=sequence),torch.permute(mask,dims=sequence)
            bbox_info_permuted = {'centroids':centroids_new,'bboxes':bboxes_new}
            return img_permuted,mask_permuted,bbox_info_permuted
        else:
            return x
class PermuteImageMask(KeepBBoxTransform):
    def __init__(self,p=0.3):
        self.p=p
        super().__init__()
    def func(self,x):
        if np.random.rand() < self.p:
            img,mask=x
            sequence =(0,)+ tuple(np.random.choice([1,2],size=2,replace=False)   ) #if dim0 is different, this will make pblms
            img_permuted,mask_permuted = torch.permute(img,dims=sequence),torch.permute(mask,dims=sequence)
            return img_permuted,mask_permuted
        else: return x



class CenteredPatch2(ItemTransform):
    '''
        tech-debt - applies variable stride
        Takes a 3-list of img,mask,bbox_stats
        Pads array with zeros if array is smaller than patch_size
        params: 
        patch_size: can be of three types, 3d array [slices,width,height] or  [1,width,height], which returns a singleslice patch, or [width,height] returns all_slices_in_image (variable number) x wigth x height
        crop_center: for kits is any combo in 'kidney','tumour','cyst'
        random_sample : gives pct of times a completely random location is selected instead
        '''

    def __init__(self,
                 patch_size,
                 expand_by=0.3,
                 minimum_num_slices=50,
                 random_shift=0.25,
                 random_sample=0.3,
                 stride_max=[4, 2, 2]):

        store_attr(but='patch_size')
        self.mod_patch_size = [int(x + x * expand_by) for x in patch_size]

    def encodes(
        self,
        x,
    ):
        stride = [np.random.randint(low=1,high=x+1) for x in self.stride_max]
        img, mask, case_info = x

        centroids, bboxes = case_info['centroids'], case_info['bboxes']
        if len(centroids) == 0 or np.random.uniform() < self.random_sample:
            target_patch_size = self.set_3d_patchsize(num_slices=0, img_shape=img.shape)
            slices, padding = create_random_bbox(target_patch_size, img.shape, stride)

        else:
            idx = torch.randint(low=0, high=len(centroids), size=[1])
            centroid = centroids[idx]
            bbox = bboxes[idx]
            num_slices = int(bbox[0].stop - bbox[0].start)
            target_patch_size = self.set_3d_patchsize(num_slices=num_slices, img_shape=img.shape)
            slices ,padding= get_slices_shifted_from_centroid(centroid, bbox, self.random_shift, img.shape, target_patch_size,stride)


        img = F.pad(input=img,pad=tuple(padding),value=0)
        mask = F.pad(input=mask,pad=tuple(padding),value=0)
        img_out, mask_out = img[tuple(slices)], mask[tuple(slices)]
        return img_out, mask_out

    def set_3d_patchsize(self, num_slices, img_shape):
        target_patch_size = self.mod_patch_size
        if len(target_patch_size) == 2:
            if num_slices == 0:
                target_patch_size = [img_shape[0]] + target_patch_size
            else:
                num_sl = np.maximum(self.minimum_num_slices, num_slices)
                target_patch_size = [num_sl] + target_patch_size
        return target_patch_size


def get_slices_shifted_from_centroid(centroid, bbox, random_shift, input_shape, target_patch_size, stride):

    if isinstance(stride, int): stride = [stride] * len(input_shape)

    bbox_size = [a.stop - a.start for a in bbox]
    # input_shape, target_patch_size,centroid = [multiply_lists(arr, stride) for arr in [input_shape,target_patch_size,centroid]]
    target_patch_size= multiply_lists(target_patch_size,stride)
    flexibility_eitherside = np.array([np.maximum(0, (a - b) / 2) for a, b in zip(bbox_size, target_patch_size)])
    jitter_max = np.array([random_shift * a for a in target_patch_size])
    flexibility_plus_jitter = flexibility_eitherside + jitter_max
    shift_amount = [np.random.randint(-a, +a) if a>10 else 0 for a in flexibility_plus_jitter ]
    center_moved = [int(a + b) for a, b in zip(centroid, shift_amount)]
    patch_halved = [x / 2 for x in target_patch_size]
    slc_start =center_moved - np.floor(patch_halved).astype(np.int32)
    slc_stop = center_moved + np.ceil(patch_halved).astype(np.int32)

    pad_before = np.abs(np.minimum(0,slc_start))[::-1]
    pad_after = np.maximum(0,slc_stop- input_shape)[::-1]
    pad_final = [None,]*6
    pad_final[::2] = pad_before
    pad_final[1::2] = pad_after


    shift_start = np.abs(np.minimum(0,slc_start))
    slc_start_sh = slc_start+shift_start
    slc_stop_sh = slc_stop+shift_start
    slices = slices_from_lists(slc_start_sh, slc_stop_sh, stride)

    return  slices,pad_final
    


    

class CenteredPatch(ItemTransform):
    '''
        Takes a 3-list of img,mask,bbox_stats
        Pads array with zeros if array is smaller than patch_size
        params: 
        patch_size: can be of three types, 3d array [slices,width,height] or  [1,width,height], which returns a singleslice patch, or [width,height] returns all_slices_in_image (variable number) x wigth x height
        crop_center: for kits is any combo in 'kidney','tumour','cyst'
        random_sample : gives pct of times a completely random location is selected instead
        '''

    def __init__(self,
                 patch_size,
                 expand_by=0.3,
                 minimum_num_slices=50,
                 random_shift=0.25,
                 random_sample=0.3,
                 stride=[1, 1, 1]):

        store_attr(but='patch_size')
        self.mod_patch_size = [int(x + x * expand_by) for x in patch_size]

    def encodes(
        self,
        x,
    ):
        img, mask, case_info = x

        centroids, bboxes = case_info['centroids'], case_info['bboxes']
        if len(centroids) == 0 or np.random.uniform() < self.random_sample:
            target_patch_size = self.set_3d_patchsize(num_slices=0, img_shape=img.shape)
            slices = create_random_bbox(target_patch_size, img.shape, self.stride)
        else:
            idx = torch.randint(low=0, high=len(centroids), size=[1])
            centroid = centroids[idx]
            bbox = bboxes[idx]
            num_slices = int(bbox[0].stop - bbox[0].start)
            target_patch_size = self.set_3d_patchsize(num_slices=num_slices, img_shape=img.shape)
            slices = get_slices_from_centroid(centroid, bbox, self.random_shift, img.shape, target_patch_size,
                                              self.stride)

        img_out, mask_out = img[tuple(slices)], mask[tuple(slices)]
        return img_out, mask_out

    def set_3d_patchsize(self, num_slices, img_shape):
        target_patch_size = self.mod_patch_size
        if len(target_patch_size) == 2:
            if num_slices == 0:
                target_patch_size = [img_shape[0]] + target_patch_size
            else:
                num_sl = np.maximum(self.minimum_num_slices, num_slices)
                target_patch_size = [num_sl] + target_patch_size
        return target_patch_size


class ExpandAndPadNpArray(ItemTransform):

    def __init__(self, patch_size, expand_by=0.3, stride=[1, 1, 1], mode='constant'):
        self.patch_size = [int((ps + ps * expand_by) * strd) for ps, strd in zip(patch_size, stride)]
        self.mode = mode

    def encodes(self, x):
        img, mask, case_info = x
        target_patch_size = self.patch_size
        if len(target_patch_size) < 3:
            target_patch_size = [img.shape[0]] + target_patch_size
        padding = ((0, 0), (0, 0), (0, 0))
        patch_size_vs_img_size = [x > y for x, y in zip(target_patch_size, img.shape)]
        if any(patch_size_vs_img_size):  # check if any dim of image is smaller than patch_size
            centroids, bboxes = case_info['centroids'], case_info['bboxes']
            padding = get_amount_to_pad_np(x[0].shape, target_patch_size)
            padded_arrays = []
            for arr in img, mask:
                arr = np.pad(arr, padding, self.mode)
                padded_arrays.append(arr)

            centroids = [np.array(padding)[:, 0] + centroid for centroid in centroids]
            case_info = {'centroids': centroids, 'bboxes': bboxes}
            img, mask = padded_arrays
        return img, mask, case_info

    def decodes(self, x, padding):
        s = [slice(p[0], s - p[1]) for p, s in zip(padding, x[0].shape)]
        x = [x[0][tuple(s)], x[1][tuple(s)]]
        return x


class ExpandAndPadTorch(ItemTransform):

    def __init__(self, patch_size, expand_by:Union[list,float]=0.3, stride=[1, 1, 1], mode='constant'):
        if isinstance(expand_by,float): expand_by=[expand_by,]*3
        self.patch_size = [int((ps + ps * expand) * strd) for ps, expand ,strd in zip(patch_size,expand_by, stride)]
        self.mode = mode

    def encodes(self, x):
        img, mask, case_info = x
        target_patch_size = self.patch_size
        if len(target_patch_size) < 3:
            target_patch_size = [img.shape[0]] + target_patch_size
        padding = (0, 0, 0, 0, 0, 0)
        patch_size_vs_img_size = [x > y for x, y in zip(target_patch_size, img.shape)]
        if any(patch_size_vs_img_size):  # check if any dim of image is smaller than patch_size
            centroids, bboxes = case_info['centroids'], case_info['bboxes']
            padding = get_amount_to_pad_torch(x[0].shape, target_patch_size)
            padded_arrays = []
            for arr in img, mask:
                arr = F.pad(arr, padding, self.mode)
                padded_arrays.append(arr)

            centroids = [np.array(padding)[::-2] + centroid for centroid in centroids]
            case_info = {'centroids': centroids, 'bboxes': bboxes}
            img, mask = padded_arrays
        return img, mask, case_info

    def decodes(self, x, padding):
        s = [slice(p[0], s - p[1]) for p, s in zip(padding, x[0].shape)]
        x = [x[0][tuple(s)], x[1][tuple(s)]]
        return x


class GetLabelCentroids(ItemTransform):

    def __init__(self, crop_center):
        crop_center = listify(crop_center)
        store_attr()

    def encodes(self, x):
        img, mask, case_info = x
        bbox_stats = case_info['bbox_stats']
        centroids = []
        for label in self.crop_center:
            bbox = [bbox for bbox in bbox_stats if bbox['tissue_type'] == label]
            if len(bbox) > 0:
                centroids.append(bbox[0]["centroids"][1:])
                bboxes = bbox[0]['bounding_boxes'][1:]
                info = {"centroids": np.vstack(centroids), "bboxes": bboxes}
            else:
                info = None
        return img, mask, info


class CenteredPatch(ItemTransform):
    '''
        Takes a 3-list of img,mask,bbox_stats
        Pads array with zeros if array is smaller than patch_size
        params: 
        patch_size: can be of three types, 3d array [slices,width,height] or  [1,width,height], which returns a singleslice patch, or [width,height] returns all_slices_in_image (variable number) x wigth x height
        crop_center: for kits is any combo in 'kidney','tumour','cyst'
        random_sample : gives pct of times a completely random location is selected instead
        '''

    def __init__(self,
                 patch_size,
                 expand_by=0.3,
                 minimum_num_slices=50,
                 random_shift=0.25,
                 random_sample=0.3,
                 stride=[1, 1, 1]):

        store_attr(but='patch_size')
        self.mod_patch_size = [int(x + x * expand_by) for x in patch_size]

    def encodes(
        self,
        x,
    ):
        img, mask, case_info = x


        centroids, bboxes = case_info['centroids'], case_info['bboxes']
        if len(centroids) == 0 or np.random.uniform() < self.random_sample:
            target_patch_size = self.set_3d_patchsize(num_slices=0, img_shape=img.shape)
            slices = create_random_bbox(target_patch_size, img.shape, self.stride)
        else:
            idx = torch.randint(low=0, high=len(centroids), size=[1])
            centroid = centroids[idx]
            bbox = bboxes[idx]
            num_slices = int(bbox[0].stop - bbox[0].start)
            target_patch_size = self.set_3d_patchsize(num_slices=num_slices, img_shape=img.shape)
            slices = get_slices_from_centroid(centroid, bbox, self.random_shift, img.shape, target_patch_size,
                                              self.stride)

        img_out, mask_out = img[tuple(slices)], mask[tuple(slices)]
        return img_out, mask_out

    def set_3d_patchsize(self, num_slices, img_shape):
        target_patch_size = self.mod_patch_size
        if len(target_patch_size) == 2:
            if num_slices == 0:
                target_patch_size = [img_shape[0]] + target_patch_size
            else:
                num_sl = np.maximum(self.minimum_num_slices, num_slices)
                target_patch_size = [num_sl] + target_patch_size
        return target_patch_size


def create_random_bbox(patch_size, img_shape, stride=1):
    if isinstance(stride, int): stride = [stride] * 3
    patch_size = np.array(patch_size)
    patch_size_strided = np.array(multiply_lists(patch_size, stride))
    flexibility = img_shape - patch_size_strided
    slc_start = np.random.randint(low=0,
                                  high=np.maximum(1,
                                                  flexibility))  # this prevent flexibility of 0 causing a value error
    slc_stop = slc_start + patch_size_strided  # -1 , otherwise it overshoots
    pad_tot =np.maximum((slc_stop-img_shape),0)[::-1]
    pad_final = [None,]*6
    for i in range(len(pad_tot)):
      pad_final[i*2]= int(np.floor(pad_tot[i]/2))
      pad_final[i*2+1]= int(np.ceil(pad_tot[i]/2))
    slices = slices_from_lists(slc_start, slc_stop, stride)

    return tuple(slices), pad_final


class ExpandPadAndCenterArray(ItemTransform):
    '''
    Wraps ExpandAndPadNpArray and CenteredPatch into one pipeline, since both share patch_size and expand_by in a typical pipeline
    '''

    def __init__(self,
                 patch_size,
                 expand_by=0.3,
                 mode='constant',
                 minimum_num_slices=50,
                 random_shift=0.25,
                 random_sample=0.3,
                 stride=[1, 1, 1]):

        self.E = ExpandAndPadNpArray(patch_size=patch_size, expand_by=expand_by, stride=stride, mode=mode)
        self.C = CenteredPatch(patch_size=patch_size,
                               expand_by=expand_by,
                               minimum_num_slices=minimum_num_slices,
                               random_shift=random_shift,
                               random_sample=random_sample,
                               stride=stride)

    def encodes(self, x):

        x = self.E.encodes(x)
        x = self.C.encodes(x)
        return x

class ExpandPadAndCenterTorch(ItemTransform):
    '''
    Wraps ExpandAndPadNpArray and CenteredPatch into one pipeline, since both share patch_size and expand_by in a typical pipeline
    '''

    def __init__(self,
                 patch_size,
                 expand_by=0.3,
                 mode='constant',
                 minimum_num_slices=50,
                 random_shift=0.25,
                 random_sample=0.3,
                 stride=[1, 1, 1]):

        self.E = ExpandAndPadTorch(patch_size=patch_size, expand_by=expand_by, stride=stride, mode=mode)
        self.C = CenteredPatch(patch_size=patch_size,
                               expand_by=expand_by,
                               minimum_num_slices=minimum_num_slices,
                               random_shift=random_shift,
                               random_sample=random_sample,
                               stride=stride)

    def encodes(self, x):

        x = self.E.encodes(x)
        x = self.C.encodes(x)
        return x




class Contextual2DArrays(ItemTransform):
    '''
    2D sampler to be used right after CenterCropOrPad
    '''

    def __init__(self, num_samples_per_case=5, ch=3, step=2):
        store_attr()

    def encodes(self, x):
        img, mask = x
        slab_size = self.step * self.ch
        highest_first_slice = img.shape[0] - slab_size
        first_slices = np.random.randint(low=0, high=highest_first_slice, size=self.num_samples_per_case)
        last_slices = first_slices + self.step * self.ch
        midslices = first_slices + (int(self.ch / 2) * self.step)
        img = [img[first_slice:last_slice:self.step] for first_slice, last_slice in zip(first_slices, last_slices)]
        img = np.stack(img)
        mask = np.expand_dims(mask[midslices], 1)
        return img, mask




def slices_from_lists(slc_start, slc_stop, stride=None):
    slices = []
    for start, stop, stride_ in zip(slc_start, slc_stop, stride):
        slices.append(slice(int(start), int(stop), stride_))
    return tuple(slices)


def get_amount_to_pad_np(img_shape, patch_size):
    pad_deficits = np.maximum(0, np.array(patch_size) - img_shape)
    padding = (math.floor(pad_deficits[0] / 2),
               math.ceil(pad_deficits[0] / 2)), (math.floor(pad_deficits[1] / 2),
                                                 math.ceil(pad_deficits[1] / 2)), (math.floor(pad_deficits[2] / 2),
                                                                                   math.ceil(pad_deficits[2] / 2))
    return padding


def get_amount_to_pad_torch(img_shape, patch_size):
    pad_deficits = np.maximum(0, np.array(patch_size) - img_shape)
    padding = (math.floor(pad_deficits[2] / 2), math.ceil(pad_deficits[2] / 2), math.floor(pad_deficits[1] / 2),
               math.ceil(pad_deficits[1] / 2), math.floor(pad_deficits[0] / 2), math.ceil(pad_deficits[0] / 2))
    return padding


def reassign_labels(src_dest_labels,x):
        def _inner(mask):
            n_classes=len(src_dest_labels)
            mask_out = torch.zeros(mask.shape,dtype=mask.dtype)
            mask_tmp = one_hot(mask,n_classes,0)
            mask_reassigned = torch.zeros(mask_tmp.shape,device=mask.device)
            for src_des in src_dest_labels:
                src,dest = src_des[0],src_des[1]
                mask_reassigned[dest]+=mask_tmp[src]

            for x in range(n_classes):
                mask_out[torch.isin(mask_reassigned[x],1.0)]=x
            return mask_out
        if len(x)==2:
            img,mask= x
            output=[img]
        elif len(x)==1:
            mask =x
            output=[]
        else: tr()
        output.append(_inner(mask))
        return output
            


class MaskLabelRemap(KeepBBoxTransform):
    '''
    switches label values from src->dest by accepting list of lists [[1,2],[2,2]] will convert all ones to twos and twos to twos
    '''

    def __init__(self,src_dest_labels: tuple):
        if isinstance(src_dest_labels,str): src_dest_labels = ast.literal_eval(src_dest_labels)
        self.func =  partial(reassign_labels, src_dest_labels)


class StrideRandom(ItemTransform):
    def __init__(self,patch_size,input_dims, stride_max=[2,2,2],pad_value=-3.0,p=0.3):
        self.patch_size = patch_size
        self.stride_max=stride_max
        self.p=p

        self.Padder = PadDeficitImgMask(patch_size=self.patch_size,input_dims=input_dims,pad_value=pad_value)
    def encodes(self,x):
        img,mask = x
        if np.random.rand()<self.p:
        
            stride = [np.random.randint(low=1,high=x+1) for x in self.stride_max]
            try:
                img,mask = [xx[::stride[0],::stride[1],::stride[2]] for xx in x]
            except:
                print("Stride Random error!")
                print(x[0].shape,x[1].shape,stride)
            img,mask = self.Padder.encodes([img,mask])
        return img,mask
    

class StrideRandom2(ItemTransform):
    def __init__(self, stride_max=[2,2,2],p=0.2):
        self.stride_max=stride_max
        self.p=p

    def encodes(self,x):
        img,mask = x
        if np.random.rand()<self.p:
        
            stride = [np.random.randint(low=1,high=x+1) for x in self.stride_max]
            try:
                img,mask = [xx[::stride[0],::stride[1],::stride[2]] for xx in x]
            except:
                print("Stride Random error!")
                print(x[0].shape,x[1].shape,stride)
        return img,mask
    


class Unsqueeze(KeepBBoxTransform):  # pass this an augmentation which will be applied in turn to image and mask

    def __init__(self):
        store_attr()

    def func(self, x):
        img, mask = x
        return img.unsqueeze(0), mask.unsqueeze(0)


@GenericPairedOrganTransform
def to_tensor_paired_organs(img_mask_pair: list):

    img = torch.tensor(img_mask_pair[0].copy(), dtype=torch.float16)
    mask = torch.tensor(img_mask_pair[1].copy(), dtype=torch.uint8)
    return img, mask


@GenericPairedOrganTransform
def unsqueeze_paired(img_mask_pair: list):
    '''
    re-implement from above Unsqueeze
    '''

    img, mask = img_mask_pair
    img, mask = img.unsqueeze(0).unsqueeze(0), mask.unsqueeze(0).unsqueeze(0)
    return img, mask


@GenericPairedOrganTransform
def segmentation_mode_paired_organs(img_mask_pair: list, label):

    img, mask = img_mask_pair
    mask[mask != label] = 0
    mask[mask == label] = 1
    return img, mask


def rotate90(x):
    x = [arr.transpose(0, 2, 1) for arr in x]  # the z-dimension (typically different from other two) is not touched
    return x


def flip_vertical(x):
    x = [np.flip(arr, axis=0) for arr in x]
    return x


class FakeTumourPaste(ItemTransform):
        def __init__(self,fake_tumour_fns, p=0.7, scale_max=2):
            self.fake_tumour_fns= fake_tumour_fns
            self.label_index=2 # tumourindex
            self.len = len(self.fake_tumour_fns)
            self.p=p
            FakeAffine  = AffineTrainingTransform3D(p=0.99,rotate_max=pi/2,scale_ranges=[0.75,scale_max],shear=False)
            self.AffineTransform = Pipeline([Unsqueeze,Unsqueeze,FakeAffine,Squeeze(dim=0),Squeeze(dim=0)])
    
        @classmethod
        def from_caseids(self, case_ids, fake_folder,*args,**kwargs):
            fake_filenames = list(fake_folder.glob("*.pt"))
            fake_filenames_final = [match_filename_with_case_id(cid,fake_filenames) for cid in case_ids]
            self=self(fake_filenames_final,*args,**kwargs)
            return self
        
        def compute_target_location(self,bbox,tmr,msk,img_shape):
                tmr = tmr[:img_shape[0]-10,:img_shape[1]-10,:img_shape[2]-10]
                msk = msk[:img_shape[0]-10,:img_shape[1]-10,:img_shape[2]-10]
                bb = bbox['bbox_stats']
                bb_k = [b for b in bb if b['tissue_type']=="kidney"][0]
                kidneys = len(bb_k['voxel_counts'])-1
                if kidneys == 2:
                    ind= np.random.randint(1,3)
                else :
                    ind = 1

                bbox = bb_k['bounding_boxes'][ind]
                tmr_center=[]
                tumour_size_half = [a/2 for a in tmr.shape]
                for i, a in enumerate(bbox):
                    lower_bound = np.ceil(tumour_size_half[i])
                    upper_bound = img_shape[i]-np.ceil(tumour_size_half[i])
                    length_kid = a.stop-a.start
                    loc_tumr = np.random.randint(a.start+0.2*length_kid,a.start+0.8*length_kid)
                    loc_tumr = np.maximum(lower_bound,loc_tumr)
                    loc_tumr = np.minimum(loc_tumr,upper_bound)
                    tmr_center.append(loc_tumr)
                tumour_last_inds= [a+b for a,b in zip(tmr_center,tumour_size_half)]
                excess = [np.minimum(0,a-b) for a,b in zip(img_shape,tumour_last_inds)]
                tmr_slcs  = []
                for inx in range(3):
                    tmr_slcs.append(slice(0,int(tmr.shape[inx]+excess[inx])))

                tmr_final = tmr[tuple(tmr_slcs)]
                mask_final = msk[tuple(tmr_slcs)]
                tmr_final_size = tmr_final.shape
                tumour_slcs = []
                for a ,b in zip(tmr_center, tmr_final_size):
                    tumour_slcs.append(slice(int(a-np.floor(b/2)),int(a+np.ceil(b/2))))
                mask_fullsize= tmr_fullsize= torch.zeros(img_shape)
                try:
                    tmr_fullsize[tumour_slcs]=tmr_final
                except:
                    tr()
                tmr_backup = tmr_fullsize.clone()
                mask_fullsize[tumour_slcs]=mask_final
                inds_tmr = torch.where(mask_fullsize==self.label_index)
                tmr_at_inds = tmr_backup[inds_tmr]
                return tmr_at_inds, inds_tmr

        def encodes(self,x):
            img,mask,bbox = x
            if np.random.rand()<self.p:
                indx= np.random.randint(0,self.len)
                tumour_fn = self.fake_tumour_fns[indx]
                im= torch.load(tumour_fn)
                tmr,msk = im['img'], im['mask']
                tmr,msk = self.AffineTransform([tmr,msk])
                tmr_values , indices= self.compute_target_location(bbox,tmr,msk,img.shape)
                img[indices]=tmr_values
                mask[indices]=self.label_index
            return img,mask,bbox

# %%

if __name__ == "__main__":
    from fran.data.dataset import ImageMaskBBoxDataset
    from fran.transforms.misc_transforms import create_augmentations
    from fran.utils.common import *
# %%
    
# %%
    P = Project(project_title="lits"); proj_defaults= P
    spacings = [1,1,1]
    src_patch_size = [220,220,110]
    patch_size = [160,160,128]
    images_folder = proj_defaults.patches_folder/("spc_100_100_250")/("dim_{0}_{0}_{2}".format(*src_patch_size))/("images")
    bboxes_fn =images_folder.parent/"bboxes_info"  
    bboxes = load_dict(bboxes_fn)
    fold = 0

    json_fname=proj_defaults.validation_folds_filename

    imgs =list((proj_defaults.raw_data_folder/("images")).glob("*"))
    masks =list((proj_defaults.raw_data_folder/("masks")).glob("*"))
    img_fn = imgs[0]
    mask_fn = masks[0]
    train_ids,val_ids,_ =  get_fold_case_ids(fold=0, json_fname=proj_defaults.validation_folds_filename)
    train_ds = ImageMaskBBoxDataset(proj_defaults,train_ids,bboxes_fn)
    valid_ds = ImageMaskBBoxDataset(proj_defaults,val_ids,bboxes_fn)


    aa = train_ds.median_shape
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
    from fran.data.dataset import ImageMaskBBoxDataset
    from fran.utils.imageviewers import *
    P = Project(project_title="lits"); proj_defaults= P
    # %%
    bboxes_pt_fn = proj_defaults.stage1_folder / ("cropped/images_pt/bboxes_info")
    bboxes_nii_fn = proj_defaults.stage1_folder / ("cropped/images_nii/bboxes_info")

# %%

    print(len(train_ds))
# %%
    
# %%
    i=15
    a,b,c = train_ds[i]

# %%
    src_dest_labels= [[0,0],[1,1],[2,2],[3,3]]
    P = PermuteImageMask(p=0.99)
    x = P([a,b,c])
    M = MaskLabelRemap([[0,0],[1,1],[2,2],[3,3]])
    C = CropImgMask(patch_size=[128,]*3,input_dims = 3)
    d = C([a,b,c])
# %%
# %%
    ImageMaskViewer([a,b])
# %%
    aa,bb = a.unsqueeze(0).unsqueeze(0), b.unsqueeze(0).unsqueeze(0)
    A  = AffineTrainingTransform3D(p=1.0)
    C  = CropImgMask(patch_size, 5)
# %%
    a2,b2 = A.encodes([aa,bb])
    a3,b3 = C.encodes([a2,b2])

# %%
    ImageMaskViewer([a2[0,0], b2[0,0]])
    ImageMaskViewer([a3[0,0], b3[0,0]])

# %%
    img,mask,bboxes = a,b,c
    expand_factor = np.minimum(np.random.uniform(),G.grow_max)
    print(expand_factor)
    bbox_all = bboxes['bbox_stats']
    tumour_bb = [a for a in bbox_all if a['tissue_type']=='tumour'][0]['bounding_boxes'][1]
    lins=[]
    for sh in img.shape:
        lins.append(np.linspace(-1,1,sh))
# %%


# %%
    lins_out = []
    for slices,linss in zip(tumour_bb,lins):
        lins_out.append (expand_lesion(slices.start,slices.stop,linss, expand_factor=expand_factor))
# %%
    plt.subplot(221)
    for ind in range(3):
       axx = plt.subplot(1,3,ind+1)
       axx.plot(lins[0],lins[0],'k.')
       axx.plot(lins[ind],lins_out[ind]) 
# %%

    ImageMaskViewer([a,x[0,0]])
# %%

    img,mask,bboxes= a,b,c
# %%
    # try:

    bbox_all = bboxes['bbox_stats']
    tumour_bb = [a for a in bbox_all if a['tissue_type']=='tumour'][0]['bounding_boxes'][1]
    # except IndexError:
    #     return img,mask,bboxes
    lins=[]
    for sh in img.shape:
        lins.append(torch.linspace(-1,1,sh))
# %%
    lins_out = []
    for slices,linss in zip(tumour_bb,lins):
        slices, linss = tumour_bb[0],lins[0]
# %%
    # lins_out.append (expand_lesion(slices.start,slices.stop,linss, expand_factor=expand_factor))
    all_locs = linss.numpy()
    lesion_start_ind,lesion_end_ind = slices.start,slices.stop
# %%

    expand_factor= np.random.uniform(1,1.5)
    expand_factor=1.5
    num_inds = len(all_locs)
    lesion_start_x = all_locs[lesion_start_ind]
    lesion_end_x = all_locs[lesion_end_ind-1]
    lesion_span = lesion_end_ind-lesion_start_ind
    lesion_xy_len= lesion_end_x-lesion_start_x
    lesion_center = (lesion_end_x+lesion_start_x)/2
    lesion_to_total_frac = lesion_span/len(all_locs)
    lesion_locs_shifted = [lesion_start_x-lesion_center, lesion_end_x-lesion_center]
    lesion_start_y = expand_factor*lesion_locs_shifted[0]+lesion_center
    lesion_end_y = expand_factor*lesion_locs_shifted[1]+lesion_center
    smooth_before_ind= np.maximum(0,int(lesion_start_ind-lesion_span))
    smooth_before_x = all_locs[smooth_before_ind]
    smooth_after_ind = np.minimum(num_inds, int(lesion_end_ind+lesion_span))
    smooth_after_x = all_locs[smooth_after_ind]
    smooth_before_len = np.minimum(lesion_span,lesion_start_ind)
    smooth_after_len = np.minimum(lesion_span,lesion_start_ind)





# %%
    # if lesion_to_total_frac>0.5:
    #     return all_locs
    # if lesion_to_total_frac> 0.4:
    #     expand_factor = np.minimum(0.2,expand_factor)
    # if lesion_to_total_frac> 0.3:
    #     expand_factor = np.minimum(0.3,expand_factor)
    # if lesion_to_total_frac> 0.2:
    #     expand_factor = np.minimum(0.4,expand_factor)
    knots = [-1.0,smooth_before_x,lesion_start_x,lesion_end_x,smooth_after_x,1.0]
    seg_y_lims = [-1,smooth_before_x,lesion_start_y,lesion_end_y,smooth_after_x,1.0]
    seg_lengths=[smooth_before_ind,lesion_start_ind-smooth_before_ind, lesion_end_ind-lesion_start_ind,smooth_after_ind-lesion_end_ind,num_inds-smooth_after_ind]
    
    y0 = torch.linspace(seg_y_lims[0],seg_y_lims[1],int(seg_lengths[0]+1))[:-1]
    y1 =  torch.linspace(seg_y_lims[1],seg_y_lims[2],int(seg_lengths[1]+1))[:-1]
    y2 =  torch.linspace(seg_y_lims[2],seg_y_lims[3],int(seg_lengths[2]+1))[:-1]
    y3 =   torch.linspace(seg_y_lims[3],seg_y_lims[4],int(seg_lengths[3]+1))[:-1]
    y4 =   torch.linspace(seg_y_lims[4],seg_y_lims[5],int(seg_lengths[4]))
    y = torch.cat([y0,y1,y2,y3,y4])
    plt.plot(np.linspace(-1,1,64),y)
# %%
    # k1 = mu_1*lesion_end_x
    # k2 = smooth_out_after_loc
    expand_by = int(expand_factor*lesion_span)
    lesion_span_new= int(lesion_span+ np.minimum(expand_by,lesion_start_ind)+np.minimum(len(all_locs)-lesion_end_x,expand_by))
    smooth_out_zone=expand_by
    smooth_out_before_ind= int(lesion_start_ind-expand_by-smooth_out_zone)
    truncate_before = int(np.minimum(lesion_start_ind-expand_by-smooth_out_zone,0))
    truncate_after = np.minimum(0,len(all_locs)-5-(lesion_end_ind + expand_by+smooth_out_zone))# shorten the after_zone if it exceeds total image dim
    smooth_out_after_ind = (int(lesion_end_ind + expand_by+smooth_out_zone+truncate_after))
    smooth_out_before_loc = all_locs[smooth_out_before_ind]
    smooth_out_after_loc = all_locs[smooth_out_after_ind-1]

    smooth_out_before_zone = torch.linspace(smooth_out_before_loc,lesion_start_x,np.maximum(0,smooth_out_zone+1+truncate_before))
    smooth_out_after_zone = torch.linspace(lesion_end_x,smooth_out_after_loc,smooth_out_zone+1+truncate_after)
    lesion_new_locs = torch.linspace(lesion_start_x,lesion_end_x,lesion_span_new)
    lesion_segment_new = torch.cat([smooth_out_before_zone[:-1],lesion_new_locs,smooth_out_after_zone[1:]])
    all_locs_modified = all_locs.clone()
    all_locs_modified[smooth_out_before_ind:smooth_out_after_ind]= lesion_segment_new

# %%
    x ,y, z= torch.meshgrid(*lins_out)
    expanded_mesh = torch.stack([z,y,x],3)
    expanded_mesh.unsqueeze_(0)
    img_= img.unsqueeze(0).unsqueeze(0)
    img_warped= F.grid_sample(img_,expanded_mesh)
    mask_ = mask.unsqueeze(0).unsqueeze(0)

    mask_warped= F.grid_sample(mask_.float(),expanded_mesh)
    img,mask = img_warped.squeeze(0).squeeze(0),mask.squeeze(0).squeeze(0)

# %%
    img,mask,bboxes = a,b,c
    expand_factor = np.minimum(np.random.uniform(),G.grow_max)
    print(expand_factor)
    bbox_all = bboxes['bbox_stats']
# %%

    lins=[]
    for sh in img.shape:
        lins.append(torch.linspace(-1,1,sh))
    x,y,z = np.meshgrid(lins[2],lins[1],lins[0])

# %%

    ax = plt.figure().add_subplot(projection='3d')

    ax.quiver(x, y, z, x, y, z, length=0.1, normalize=False)
    plt.show()
# %%
#     try:
#         tumour_bb = [a for a in bbox_all if a['tissue_type']=='tumour'][0]['bounding_boxes'][1]
#     except IndexError:
#         return img,mask,bboxes
#     lins=[]
#     for sh in img.shape:
#         lins.append(torch.linspace(-1,1,sh))
#     lins_out = []
#     for slices,linss in zip(tumour_bb,lins):
#     slices = tumour_bb[0]
#     linss = lins[0]
#
# # %%
#         lins_out.append (expand_lesion(slices.start,slices.stop,linss, expand_factor=0.8))
#     x ,y, z= torch.meshgrid(*lins_out)
#     expanded_mesh = torch.stack([z,y,x],3)
#     expanded_mesh.unsqueeze_(0)
#     img_= img.unsqueeze(0).unsqueeze(0)
#     img_warped= F.grid_sample(img_,expanded_mesh)
#
# # %%
#     lesion_start_ind = slices.start
#     lesion_end_ind = slices.stop
#     all_locs = linss
# # %%
#     lesion_start_loc = all_locs[lesion_start_ind]
#     lesion_end_loc = all_locs[lesion_end_ind-1]
#     lesion_span = lesion_end_ind-lesion_start_ind
#     lesion_to_total_frac = lesion_span/all_locs.numel()
# # %%
#     if lesion_to_total_frac>0.5:
#         return all_locs
#     if lesion_to_total_frac> 0.4:
#         expand_factor = np.minimum(0.2,expand_factor)
#     if lesion_to_total_frac> 0.3:
#         expand_factor = np.minimum(0.3,expand_factor)
#     if lesion_to_total_frac> 0.2:
#         expand_factor = np.minimum(0.4,expand_factor)
# %%
    expand_by = int(expand_factor*lesion_span)
    lesion_span_new= int(lesion_span+ np.minimum(expand_by,lesion_start_ind)+np.minimum(all_locs.numel()-lesion_end_x,expand_by))
    lesion_center = int((lesion_start_ind+lesion_end_ind)/2)
    smooth_out_zone=expand_by
    start_gap = 0 if lesion_start_ind-expand_by<7 else 5
    smooth_out_before_ind= int(np.maximum(lesion_start_ind-expand_by-smooth_out_zone,start_gap))
    truncate_before = int(np.minimum(lesion_start_ind-expand_by-smooth_out_zone-5,0))
    truncate_after = np.minimum(0,all_locs.numel()-5-(lesion_end_ind + expand_by+smooth_out_zone))# shorten the after_zone if it exceeds total image dim
    smooth_out_after_ind = (int(lesion_end_ind + expand_by+smooth_out_zone+truncate_after))
    smooth_out_before_loc = all_locs[smooth_out_before_ind]
    smooth_out_after_loc = all_locs[smooth_out_after_ind-1]

    smooth_out_before_zone = torch.linspace(smooth_out_before_loc,lesion_start_x,np.maximum(0,smooth_out_zone+1+truncate_before))
    smooth_out_after_zone = torch.linspace(lesion_end_x,smooth_out_after_loc,smooth_out_zone+1+truncate_after)
    lesion_new_locs = torch.linspace(lesion_start_x,lesion_end_x,lesion_span_new)
    lesion_segment_new = torch.cat([smooth_out_before_zone[:-1],lesion_new_locs,smooth_out_after_zone[1:]])
    all_locs_modified = all_locs.clone()
    all_locs_modified[smooth_out_before_ind:smooth_out_after_ind]= lesion_segment_new
# %%
    lesion_start_ind
    lesion_end_ind
    lesion_center = int((lesion_end_ind+lesion_start_ind)/2)
# %%
    lesion_start_x
    lesion_end_x
    smooth_out_before_loc
    smooth_out_after_loc
# %%
    plt.plot(all_locs_modified)
# %%
    ImageMaskViewer([a,x[0,0]])
# %%
    FF = FakeTumourPaste.from_caseids(train_list, proj_defaults.stage0_folder/"tumour_only",p=0.99,scale_max=1.5)
# %%
    ind = 10
    a, b, c = train_ds[ind]
    # x,y,z = FF.encodes([a,b,c])
# %%
    b[b!=2]=0
    bb=    get_bbox_from_mask(b)[0]
    ImageMaskViewer([a[bb],a[bb]])
# %%

    
# %%
        # for a ,b in zip(tmr_center, tmr_final_size):
        #     print()
        #     tumour_slcs.append(slice(int(a-np.floor(b/2)),int(a+np.ceil(b/2))))
        #
        # mask_fullsize= tmr_fullsize= torch.zeros(img_shape)
        # tmr_fullsize[tumour_slcs]=tmr_final
        # tmr_backup = tmr_fullsize.clone()
        # mask_fullsize[tumour_slcs]=mask_final
        # inds_tmr = torch.where(mask_fullsize==self.label_index)
        #
# %%
    ImageMaskViewer([x,y])
# %%
    img,mask,bbox = a,b,c
    indx= np.random.randint(0,FF.len)
    tumour_fn = FF.fake_tumour_fns[indx]
# %%
    im= torch.load(tumour_fn)
    tmr0,msk0 = im['img'], im['mask']
    tmr,msk = FF.AffineTransform([tmr0,msk0])
    print(tmr0.shape,tmr.shape)
# %%
    ImageMaskViewer([tmr0,msk0])
    ImageMaskViewer([a,b])
# %%
    bbox=c
    tumour_size = tmr.shape
    print(tumour_size)
    bb = bbox['bbox_stats']
    bb_k = [b for b in bb if b['tissue_type']=="kidney"][0]
    kidneys = len(bb_k['voxel_counts'])-1
    ind = 1
    bbox = bb_k['bounding_boxes'][ind]
    cent= bb_k['centroids'][ind]
# %%

    # folder = proj_defaults.stage1_folder/"patches_48_128_128"
    # train_ds = ImageMaskBBoxDataset(proj_defaults,train_list,[0,1,2])
    a, b, c = train_ds[203]
    ImageMaskViewer([a,b])
# %%
    a2 = a[25,60:100,40:80]
    a2.unsqueeze_(0).unsqueeze_(0)
    plt.imshow(a2[0,0])
    _,_,y_dim,x_dim= a2.shape
# %%
    def distance(p1,p2):
        d = (p1-p2)**2
        d = torch.sqrt(d.sum())
        return d
    def weight_curve(d,alpha):
        return 1-d**alpha
    def weight_fold(d,alpha):
        return alpha/(d+alpha)

# %%
######################################################################################
# %% [markdown]
## Elastic deforms
# %%
    x_dim = a2.shape[-1]
    aa = torch.linspace(-1,1,int(x_dim))
    x,y = torch.meshgrid(aa,aa)
    base_mesh = torch.stack([y,x],2)
    base_mesh.unsqueeze_(0)

# %%
######################################################################################
# %% [markdown]
## Splines
# %%
    c=4
    locs = torch.randint(0,x_dim,[c,2])
    locs
######################################################################################
# %% [markdown]
## Random elastic deformation
# %%

    def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        return gaussian_kernel
# %%
    kern = get_gaussian_kernel(channels=1,sigma=18)
    kern= kern.unsqueeze(0).unsqueeze(0)
    conv = torch.nn.Conv2d(1,1,kernel_size=3,padding='same',bias=None)
    conv.weight.data=kern


# %%
    scale=1
    x_delta = torch.rand(1,1,128,128)*2-1
    y_delta = torch.rand(1,1,128,128)*2-1
    x_delta2 = conv(x_delta*scale)
    y_delta2 = conv(y_delta*scale)
    elast = torch.stack([x_delta2.squeeze(0),y_delta2.squeeze(0)],dim=-1)
    elast.shape
# %%
    outp2 = F.grid_sample(a,base_mesh+elast).detach().cpu()
    plt.imshow(outp2[0,0])
# %%
    ff_rand = torch.rand(base_mesh.shape)*2-1
# %%
# %%


# %%
    

    alpha = 0.8
    dists  = torch.zeros(y.shape)
# %%
    p_rand = base_mesh[0,20,30,:]
    v = torch.tensor([0.12,0.73])
    v_normed = v/v.norm()
    line = p_rand-v
    for i in range(base_mesh.shape[1]):
        for j in range(base_mesh.shape[2]):
              p_ij = base_mesh[0,i,j,:]
              dists[i,j] =               ((p_ij-p_rand)-(torch.dot(p_ij-p_rand,v_normed))*v_normed).norm()

# %%
    weights = weight_curve(dists,alpha)
    weights2 = weight_fold(dists,alpha)

# %%
    ff_def = base_mesh.clone()
    for i in range(base_mesh.shape[1]):
# %%
        for j in range(base_mesh.shape[2]):
            ff_def[0,i,j,:] = base_mesh[0,i,j,:]+v*weights[i,j]
# %%
# %%
    outp = F.grid_sample(a,base_mesh)
    plt.imshow(outp[0,0])
# %%
    ff_def = base_mesh.clone()
    for i in range(base_mesh.shape[1]):
        for j in range(base_mesh.shape[2]):
            ff_def[0,i,j,:] = base_mesh[0,i,j,:]+v*weights2[i,j]
# %%
    outp2 = F.grid_sample(a,ff_def)
    plt.imshow(outp2[0,0])
# %%
    plt.figure()
# %%
    img,mask = a.clone(), b.clone()

    img.unsqueeze_(0).unsqueeze_(0)
    mask.unsqueeze_(0).unsqueeze_(0)

# %%
    plt.quiver(base_mesh[0,:,:,0],base_mesh[0,:,:,1],ff_def[0,:,:,0],ff_def[0,:,:,1])
# %%

    sz = [56,56,56]
    img = F.interpolate(img, size=sz, mode='trilinear')
    mask = F.interpolate(mask, size=sz, mode='nearest')
    ImageMaskViewer([img[0,0],mask[0,0]])
# %%
    
# %%
    ImageMaskViewer([x,y])
# %%
    CC = Contextual2DArrays(ch=1)
    # %%
    # %%
    n = 10
    stride=[1,1,1]
    G = GetLabelCentroids(crop_center=['kidney'])
    E = ExpandAndPadNpArray(patch_size)
    # %%

    # %%
    T = TrainingAugmentations(augs=[flip_random], p=[0.99])
# %%
    stride=[1,1,1]
    patch_size = [64, 128,128]
    ET = ExpandPadAndCenterTorch(patch_size=patch_size, expand_by=0.3, stride=stride)
    dest_labels = {"kidney": 1, "tumour": 1, "cyst": 1}
    S =MaskLabelRemap(src_dest_labels)
    G = GetLabelCentroids('kidney')

# %%
    E = ExpandAndPadTorch(patch_size=patch_size, expand_by=.3, stride=stride, mode='constant')
    xx,yy,zz = train_ds[0]
    x = G([xx,yy,zz])
    x = E(x)

# %%
    C = CenteredPatch(patch_size=patch_size,
                                   expand_by=0,
                                   random_shift=0.9,
                                   random_sample=0,
                                   stride=stride)

# %%
    x = C(x)

    ImageMaskViewer([x[0],x[1]])

# %%


    x = ET([xx,yy,zz])
    P = Pipeline([G,ET])
    x = P([xx,yy,zz])
# %%
    xa = x[0],x[1] 
# %%
# %%
    stride = [5, 4, 4]
    C = CenteredPatch2(patch_size=patch_size, expand_by=0.3, random_shift=0,stride_max=[2,2,2])
    M = PermuteImageMaskBBox()
# %%
    i = 26
    xx,yy,zz = train_ds[i]
# %%
# %%
    x,y,z = P([xx,yy,zz])
    xa,ya,za = M([x,y,z])
    x = C.encodes([xa,ya,za])
    ImageMaskViewer(x)

# %%

