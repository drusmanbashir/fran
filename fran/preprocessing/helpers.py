# %%
from fastcore.basics import   listify
import numpy as np
from fran.transforms.totensor import ToTensorT
from utilz.helpers import *

from utilz.helpers import *
from utilz.fileio import *

from label_analysis.utils import SITKImageMaskFixer

def to_even(input_num, lower=True):
    np.fnc = np.subtract if lower == True else np.add
    output_num = np.fnc(input_num, input_num % 2)
    return int(output_num)


def bbox_bg_only(bbox_stats):
    all_fg_bbox = [bb for bb in bbox_stats if bb["label"] == "all_fg"][0]
    bboxes = all_fg_bbox["bounding_boxes"]
    if len(bboxes) == 1:
        return True
    elif bboxes[0] != bboxes[1]:
        return False
    else:
        tr()

def import_h5py():
    import h5py
    return h5py



def get_intensity_range(global_properties: dict) -> list:
    key_idx = [key for key in global_properties.keys() if "intensity_percentile" in key][0]
    intensity_range = global_properties[key_idx]
    return key_idx, intensity_range

@str_to_path()
def get_img_mask_filepairs(parent_folder: Union[str,Path]):
    '''
    param: parent_folder. Must contain subfolders labelled masks and images
    Files in either folder belonging to a given case should be identically named.
    '''

    imgs_folder = Path(parent_folder)/'images'
    masks_folder= Path(parent_folder)/'lms'
    imgs_all=list(imgs_folder.glob('*'))
    masks_all=list(masks_folder.glob('*'))
    assert (len(imgs_all)==len(masks_all)), "{0} and {1} folders have unequal number of files!".format(imgs_folder,masks_folder)
    img_label_filepairs= []
    for img_fn in imgs_all:
            label_fn = find_matching_fn(img_fn,masks_all)
            assert label_fn.exists(), f"{label_fn} doest not exist, \ncorresponding to {img_fn}"
            img_label_filepairs.append([img_fn,label_fn])
    return img_label_filepairs


@str_to_path(0)
def verify_dataset_integrity(folder:Path, debug=False,fix=False):
    '''
    folder has subfolders images and masks
    '''
    print("Verifying dataset integrity")
    subfolder = list(folder.glob("mask*"))[0]
    args = [[fn,fix] for fn in subfolder.glob("*")]
    res = multiprocess_multiarg(verify_img_label_match,args,debug=debug,io=True)
    errors = [item for item in res if re.search("mismatch", item[0],re.IGNORECASE)]
    if len(errors)>0:
        outname = folder/("errors.txt")
        print(f"Errors found saved in {outname}")
        save_list(errors,outname)
        res.insert(0,errors)
    else:
        print("All images and masks are verified for matching sizes and spacings.")
    return res


def verify_datasets_integrity(folders:list, debug=False,fix=False)->list:
    folders = listify(folders)
    res =[]
    for folder in folders:
        res.extend(verify_dataset_integrity(folder,debug,fix))
    return res
        
    

def verify_img_label_match(label_fn:Path,fix=False):
    imgs_foldr = label_fn.parent.str_replace("lms","images")
    img_fnames = list(imgs_foldr.glob("*"))
    assert (imgs_foldr.exists()),"{0} corresponding to {1} parent folder does not exis".format(imgs_foldr,label_fn)
    img_fn = find_matching_fn (label_fn,img_fnames)
    if '.pt' in label_fn.name:
        return verify_img_label_torch(label_fn)
    else:
        S = SITKImageMaskFixer(img_fn,label_fn)
        S.process(fix=fix)
        return S.log

@str_to_path()
def verify_img_label_torch(label_fn:Path):
    if isinstance(label_fn,str): label_fn = Path(label_fn)
    img_fn = label_fn.str_replace('lms','images')
    img,mask = list(map(torch.load,[img_fn,label_fn]))
    if img.shape!=mask.shape:
        print(f"Image mask mismatch {label_fn}")
        return '\nMismatch',img_fn,label_fn,str(img.shape),str(mask.shape)

def get_label_stats(mask, label, separate_islands=True, dusting_threshold: int = 0):
    import cc3d
    if torch.is_tensor(mask):
        mask = mask.numpy()
    label_tmp = np.copy(mask.astype(np.uint8))
    label_tmp[mask != label] = 0
    if dusting_threshold >0:
        label_tmp = cc3d.dust(
            label_tmp, threshold=dusting_threshold, connectivity=26, in_place=True
        )

    if separate_islands:
        label_tmp, N = cc3d.largest_k(
            label_tmp, k=1000, return_N=True
        ) 
    stats = cc3d.statistics(label_tmp)
    return stats




def percentile_range_to_str(percentile_range):
    def _num_to_str(num: float):
        if num in [0, 100]:
            substr = str(num)
        else:
            str_form = str(num).split(".")
            prefix_zeros = 2 - len(str_form[0])
            suffix_zeros = 1 - len(str_form[1])
            substr = "0" * prefix_zeros + str_form[0] + str_form[1] + "0" * suffix_zeros
        return substr

    substrs = [_num_to_str(num) for num in percentile_range]
    return "_".join(["intensity_percentile"] + substrs)


def get_std_numerator(img_fname, dataset_mean, clip_range=None):
    img = ToTensorT(torch.float32)(img_fname)
    if clip_range is not None:
        img = torch.clip(img,min=clip_range[0],max=clip_range[1])
    var = (img - dataset_mean) ** 2
    var_sum = var.sum()
    return var_sum


def get_means_voxelcounts(img_fname, clip_range=None):
    img = ToTensorT(torch.float32)(img_fname)
    if clip_range is not None:
        img = torch.clip(img,min=clip_range[0],max=clip_range[1])
    return img.mean().item(), img.numel()



class BBoxesFromMask(object):
    """ """

    def __init__(
        self,
        filename,
        bg_label=0, # so far unused in this code
    ):
        if not isinstance(filename,Path): filename = Path(filename)
        if filename.suffix == '.pt':
            self.mask =torch.load(filename,weights_only=False)
        else:
            self.mask = sitk.ReadImage(str(filename))
        if isinstance(self.mask,torch.Tensor): self.mask = np.array(self.mask)
        if isinstance(self.mask,sitk.Image): self.mask = sitk.GetArrayFromImage(self.mask)
        case_id = info_from_filename(filename.name,full_caseid=True)['case_id']
        self.bboxes_info = {
            "case_id": case_id,
            "filename": filename,
        }
        self.bg_label=bg_label

    def __call__(self):
        bboxes_all = []
        label_all_fg = self.mask.copy()
        label_all_fg[label_all_fg > 1] = 1
        labels = np.unique(self.mask)
        labels = np.delete(labels,self.bg_label)
        for label in labels:
                stats = {"label": label}
                stats.update(
                    get_label_stats(
                        self.mask,
                        label,
                        True
                        )
        )
                bboxes_all.append(stats)

        stats = {"label": "all_fg"}
        stats.update(
            get_label_stats(
                label_all_fg,1,False)
        )
        bboxes_all.append(stats)
        self.bboxes_info["bbox_stats"] = bboxes_all
        return self.bboxes_info
def bboxes_function_version(
    filename,bg_label
):

    A = BBoxesFromMask(
        filename, bg_label=bg_label
    )
    return A()


if __name__ == '__main__':
# %%
    fn = "/s/fran_storage/datasets/raw_data/lidc/lms/lidc_0030.nii.gz"
    fn = "/s/xnat_shadow/crc/lms/crc_CRC004_20190425_CAP1p5.nii.gz"
    A = BBoxesFromMask(
        fn, bg_label=0
    )
    A()
    print(A.bboxes_info)
# %%
    import torch, os, torch.serialization
    import  zipfile
    t = torch.arange(6, dtype=torch.float32).reshape(2,3).contiguous()

# Force the zip-format that C++ expects:
    pth = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_080_ex000/images/drli_001ub.pt"
    im = torch.load(pth, weights_only=False)
    im = torch.Tensor(im)
    torch.save(im, "/tmp/pt_tensor.pt", _use_new_zipfile_serialization=True)
# %%

    path = "/tmp/pt_tensor.pt"

# 1) Save a single tensor in the new zip format
    t = torch.arange(6, dtype=torch.float32).reshape(2,3).contiguous()
    # torch.save(t, path, _use_new_zipfile_serialization=True)
    torch.jit.save(torch.jit.script(t), path)
# %%
    import torch
    from torch import nn
     
    class TensorContainer(nn.Module):
        def __init__(self, tensor_dict):
            super().__init__()
            for key,value in tensor_dict.items():
                setattr(self, key, value)
     
    x = torch.ones(4, 4)
    tensor_dict = {'x': x}
    tensors = TensorContainer(tensor_dict)
    tensors = torch.jit.script(tensors)
    tensors.save(path)
# 2) Verify the file format and the object type
    print("FILE_EXISTS:", os.path.exists(path))                   # True
    print("IS_ZIPFILE (zipfile):", zipfile.is_zipfile(path))      # True

    with open(path, "rb") as f:
        print("IS_ZIPFILE (torch):", torch.serialization._is_zipfile(f))  # True

    obj = torch.load(path, map_location="cpu")
    print("PY_OBJ_TYPE:", type(obj))                              # <class 'torch.Tensor'>
    print("PY_TENSOR_SHAPE:", obj.shape)

# %%
    import io
    x = torch.arange(10)
    f = io.BytesIO()
    torch.save(x, f, _use_new_zipfile_serialization=True)
    # send f wherever

# %%
    path = "/tmp/pt_tensor.pt"
    with open(path, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(f.getbuffer())
# %%
