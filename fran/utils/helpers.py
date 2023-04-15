if 'get_ipython' in globals():
# %%
        print("setting autoreload")
        from IPython import get_ipython
        ipython = get_ipython()
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')

# %%
import collections
import pprint
import re
from multiprocessing import Pool
from pathlib import Path
from typing import Union

import ipdb
import numpy as np
import torch
import tqdm

from fran.utils.dictopts import *
from fran.utils.fileio import load_dict, save_dict

tr = ipdb.set_trace
import gc
# from fran.utils.fileio import *
import random

def range_inclusive(start, end):
     return range(start, end+1)
def multiply_lists(a,b):
    return [aa*bb for aa, bb in zip(a,b)]
 
class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass   

class LazyDict:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def abs_list(numlist:list):
    return [abs(num) for num in numlist]


def chunks (listi, n):
    chunk_size = round(len(listi)/n)
    sld = [[x*chunk_size,(x+1)*chunk_size] for x in range(0,n-1)]
    sld.append([(n-1)*chunk_size,None])
    for s in sld:
        yield (listi[s[0]:s[1]])


def merge_dicts(d1, d2):
    outdict={}
    for k, v in d1.items():
        if not k in d2.keys():
            outdict[k]=d1[k]
        else:
            if isinstance(v, collections.abc.Mapping):
                outdict[k] = merge_dicts(d1[k],d2[k])
            else:
                outdict[k] = d1[k]+ d2[k]
    return outdict

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data



def dec_to_str(val:float,trailing_zeros=3):
    val2 = str(round(val,2))
    val2 = val2.replace(".","")
    trailing_zeros = np.maximum(trailing_zeros-len(val2),0) if trailing_zeros>0 else 0
    val2 = val2+'0'*trailing_zeros # ensure 3 digits
    return val2

def int_to_str(val:int, total_length=5):
    val = str(val)
    precending_zeros = total_length-len(val)
    return '0'*precending_zeros+val

def folder_name_from_list(prefix:str,parent_folder:Path,values_list=None):
        assert prefix in ("spc","dim"), "Choose a valid prefix between spc(for spacings) and dim(for patch size)"
        add_zeros=3 if prefix=='spc' else 0
        output= [dec_to_str(val,add_zeros) for val in values_list]
        subfolder =  "_".join([prefix]+output)
        return parent_folder/subfolder
# ========================== wrapper functions ====================
def path_to_str(fnc):
        def inner(*args,**kwargs):
            args = map(str,args)
            for k,v in kwargs.items():
                kwargs[k] = str(v) if isinstance(v,Path) else v
            output = fnc(*args,** kwargs)
            return output
        return inner



def pp(obj, *args,**kwargs):
    pp = pprint.PrettyPrinter(*args,**kwargs)
    pp.pprint(obj)

def str_to_list(fnc):
    def inner(input:str):
        tmp = input.split(",")
        final = [fnc(g) for g in tmp]
        return final
    return inner

def ask_proceed(statement=None):
    def wrapper(func):
        def inner(*args, **kwargs):
            if statement:
                print(statement)

            def _ask():
                decision = input("Proceed (Y/y) or Skip (N/n)?: ")
                return decision

            decision = _ask()
            if not decision in "YyNn":
                decision = _ask()
            if decision.lower() == "y":
                func(*args, **kwargs)

        return inner

    return wrapper

# ====================================================================================================
@str_to_list
def str_to_list_float(input):
    return float(input)

@str_to_list
def str_to_list_int(input):
    return int(input)

def purge_cuda(learn):
    gc.collect()
    torch.cuda.empty_cache()

def get_list_input(text:str="",fnc=str_to_list_float):
    str_list = input (text)
    return fnc(str_list)

def multiprocess_multiarg(func,arguments, num_processes=8,multiprocess=True,debug=False,progress_bar=True):
    results=[]
    if multiprocess==False or debug==True:
        for res in pbar(arguments,total=len(arguments)):
            if debug==True:
                tr()
            results.append(func(*res,))
    else:
        p = Pool(num_processes)
        jobs = [p.apply_async(func=func, args=(*argument, )) for argument in arguments]
        p.close()
        pbar_fnc = get_pbar() if progress_bar==True else lambda x: x
        for job in pbar_fnc(jobs):
                results.append(job.get())
    return results

  

def get_available_device(max_memory=0.8)->int:
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0 
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available=-1
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available 

def set_cuda_device(device_id=None):
        if device_id == None:        device_id= get_available_device()
        torch.cuda.set_device(device_id)



def get_last_n_dims(img,n):
    '''
    n = number of dimensions to return
    '''
    if img.ndim <=n:
        return img
    slc = (0,)*(img.ndim-n)
    slc+=(slice(None),)*n
    return img[slc]

def sigmoid(x:np.ndarray):
    return 1/(1+np.exp(-x))


def convert_sigmoid_to_label(y,threshold = 0.5):
    y[y>=threshold]=1
    y[y<threshold]=0
    return y.int()


def regex_matcher(indx=0):
    def _outer(func):

        def _inner(*args,**kwargs):
                pat, string= func(*args,**kwargs)
                pat = re.compile(pat,re.IGNORECASE)
                answer = re.search(pat,string)
                return answer[indx] if answer else None
        return _inner
    return _outer

@regex_matcher(0)
def infer_dataset_name(filename):
    pat ="^([^-_]*)"
    return pat, filename.name


@regex_matcher()
def get_case_id_from_filename(dataset_name, filename:Path):
               if dataset_name is None: dataset_name = infer_dataset_name(filename)
               pat = "{}[-_][a-z\d]*".format(dataset_name)
               return pat, filename.name

@path_to_str
@regex_matcher()
def match_filename_with_case_id(project_title, case_id,filename):
                pat = project_title+"_"+case_id+"_?\."
                return pat,filename


@regex_matcher(1)
def get_extension(fn):
    pat = r"[^\.]*\.(.*)"
    return pat, fn.name

def get_pbar():
    if 'get_ipython' in globals():
        if get_ipython().__class__.__name__ == 'TerminalInteractiveShell' :
            return tqdm.tqdm
        else:
            return  tqdm.notebook.tqdm
    return tqdm.tqdm


def project_title_from_folder(folder_name: Union[str,Path]):
    if isinstance(folder_name,Path): folder_name = folder_name.name
    pat = re.compile("(Task\d{0,5}_)?(\w*)",re.IGNORECASE)
    result= re.search(pat,folder_name)
    return result.groups()[-1]


#
# def FileSplitter_ub(json_filename,fold=0):
#     "Split `items` by providing file `fname` (contains names of valid items separated by newline)."
#     with open(json_filename,"r") as f:
#         validation_folds=json.load(f)
#     valid = validation_folds['fold_'+str(int(fold))]
#     def _func(x): return x.name in valid
#     def _inner(o): 
#         return FuncSplitter(_func)(o)
#     return _inner
#
# def EndSplitterShort(size = 48,valid_pct=0.2, valid_last=True):
#     "Create function that splits `items` between train/val with `valid_pct` at the end if `valid_last` else at the start. Useful for ordered data."
#     assert 0<valid_pct<1, "valid_pct must be in (0,1)"
#     def _inner(o):
#         o = o[:size]
#         idxs = np.arange(len(o))
#         cut = int(valid_pct * len(o))
#         return (idxs[:-cut], idxs[-cut:]) if valid_last else (idxs[cut:],idxs[:cut])
#     return _inner
#

# def create_train_valid_test_lists_from_filenames(project_title, files_list, json_filename,pct_test=0.1,pct_valid=0.2,shuffle=False):
#     pct_valid = 0.2
#     case_ids = [get_case_id_from_filename(project_title,fn) for fn in files_list]
#     if shuffle==True: 
#         print("Shuffling all files")
#         random.shuffle(case_ids)
#     else:    
#         print("Files are in sorted order")
#         case_ids.sort()
#     final_dict= {"all_cases":case_ids} 
#     n_test= int(pct_test*len(case_ids))
#     n_valid = int(pct_valid*len(case_ids))
#     len(case_ids)-n_test-n_valid
#     cases_test =case_ids[:n_test]
#     final_dict.update({"test_cases": cases_test})
#     cases_train_valid = case_ids[n_test:]
#     folds = int(1/pct_valid)
#     n_valid_per_fold =  math.ceil(len(cases_train_valid)*pct_valid)
#     print("Given proportion {0} of validation files yield {1} folds".format(pct_valid,folds))
#     slices = [slice(fold*n_valid_per_fold,(fold+1)*n_valid_per_fold) for fold in range(folds)]
#     val_cases_per_fold = [cases_train_valid[slice] for slice in slices]
#     for n in range(folds):
#         train_cases_fold = list(set(cases_train_valid)-set(val_cases_per_fold[n]))
#         fold = {'fold_{}'.format(n):{'train':train_cases_fold, 'valid':val_cases_per_fold[n]}}
#         final_dict.update(fold)
#     print("Saving folds to {}  ..".format(json_filename))
#     save_dict(final_dict,json_filename)

def get_train_valid_test_lists_from_json(project_title, fold, json_fname, image_folder,ext=".pt"):
        all_folds= load_dict(json_fname)
        train_case_ids, validation_case_ids, test_case_ids,= all_folds['fold_'+str(int(fold))]['train'], all_folds['fold_'+str(int(fold))]['valid'],all_folds['test_cases']
        all_files =list(image_folder.glob("*{}".format(ext)))
        train_files , valid_files ,test_files=[],[], []
        for cases_list,output_list in zip([train_case_ids,validation_case_ids,test_case_ids],[train_files,valid_files,test_files]):
            for case_id in cases_list:
                    matched_files = [fn for fn in all_files if match_filename_with_case_id(project_title,case_id,fn)]
                    if len(matched_files)>1:
                        tr()
                    output_list.append(matched_files)
        return train_files,valid_files,test_files

def get_fold_case_ids(fold:int,json_fname):
        all_folds= load_dict(json_fname)
        train_case_ids, validation_case_ids, test_case_ids,= all_folds['fold_'+str(int(fold))]['train'], all_folds['fold_'+str(int(fold))]['valid'],all_folds['test_cases']  # fold ->int -> str because sometimes fold = 0 is read by pandas as bool False!
        return train_case_ids,validation_case_ids,test_case_ids


def get_fileslist_from_path(path:Path, ext:str = ".pt"):
    return list(path.glob("*"+ext))

def make_channels (base_ch=16,depth=5):
    base_multiplier = int(np.log2(base_ch))
    return [1,]+[2**n for n in range(base_multiplier,base_multiplier+depth)]

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False 


def write_file_or_not(output_filename, overwrite=True):
    if Path(output_filename).exists() and overwrite==False:
                print("File exists. Skipping {}".format(output_filename))
                return False
    else:
        return True

def write_files_or_not(output_filenames, overwrite=True):
    return list(map(write_file_or_not,output_filenames,[overwrite,]*len(output_filenames)))

pbar=get_pbar()
# %%
if __name__=="__main__":
    dd = load_dict("/home/ub/datasets/preprocessed/lits/patches/spc_100_100_200/dim_256_256_128/bboxes_info.pkl")
    dd[0]
    from fran.utils.common import *
    P = Project(project_title="lits"); proj_defaults= P.proj_summary
    image_folder = proj_defaults.stage0_folder/("images")
    files_list = get_fileslist_from_path(proj_defaults.stage0_folder/"images")
    json_fname=proj_defaults.validation_folds_filename

    create_train_valid_test_lists_from_filenames(proj_defaults.project_title,files_list,json_fname)
    train_files_w_64_192, valid_files_w_64_192= get_train_valid_test_lists_from_json(0,json_fname, proj_defaults.stage1_folder/"64_192_192/images")
# %%
    st1= '/home/ub/datasets/raw_database/raw_data/kits19/images/KiTS19_00194_0000.nii.gz'
    st2= '/home/ub/datasets/raw_database/raw_data/kits19/images/KiTS19_00194.nii.gz'
    pat = re.compile("kits19_(\d*)",re.IGNORECASE)
    aa = re.search(pat,st1)
    aa.groups()



