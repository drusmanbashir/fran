from datetime import datetime
import re
from pathlib import Path
from fastcore.basics import listify
import ipdb
import numpy as np

tr = ipdb.set_trace

def regex_matcher(indx=0):
    def _outer(func):

        def _inner(*args,**kwargs):
                pat, string= func(*args,**kwargs)
                pat = re.compile(pat,re.IGNORECASE)
                answer = re.search(pat,string)
                return answer[indx] if answer else None
        return _inner
    return _outer

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



def append_time(input_str, now=True):
    now = datetime.now()
    dt_string = now.strftime("_%d%m%y_%H%M")
    return input_str + dt_string


def infer_dataset_name(filename):
    pat = "^([^-_]*)"
    return pat, filename.name


def strip_extension(fname: str):
    exts = ".npy .nii.gz .nii .nrrd .pt".split(" ")
    for e in exts:
        pat = r"{}$".format(e)
        fname_stripped = re.sub(pat, "", fname)
        if fname_stripped != fname:
            return fname_stripped


def strip_slicer_strings(fname: str):
    """
    This is on extension-less fnames. Use strip_extension before this.
    """
    # pt = re.compile("(-?label(_\d)?)|_.*(_\d$)",re.IGNORECASE)
    pt = re.compile("(_\d)?$", re.IGNORECASE)
    pt2 = re.compile("(_\d)?-segment.*$",re.IGNORECASE)
    fname = fname.replace("-label", "")
    fname = fname.replace("-test", "")
    fname_cl1 = fname.replace("-tissue", "")
    fname_cl2 = re.sub(pt, "", fname_cl1)
    fname_cl3 = re.sub(pt2, "", fname_cl2)

    return fname_cl3


def str_to_path(arg_inds=None):
    arg_inds=listify(arg_inds)
    def wrapper(func):
        def inner (*args,**kwargs):
            if arg_inds is None:
                args = [Path(arg) for arg in args]
                kwargs = {key:Path(val) for key,val in kwargs.items()}
            else:
                args = list(args)
                all_inds = range(len(args))
                args = [Path(arg) if ind in arg_inds else arg for ind, arg in zip(all_inds,args) ]
            return func(*args,**kwargs)
        return inner
    return wrapper

def path_to_str(fnc):
        def inner(*args,**kwargs):
            args = map(str,args)
            for k,v in kwargs.items():
                kwargs[k] = str(v) if isinstance(v,Path) else v
            output = fnc(*args,** kwargs)
            return output
        return inner



@str_to_path(0)
@regex_matcher(1)
def get_extension(fn):
    pat = r"[^\.]*\.(.*)"
    return pat, fn.name


def cleanup_fname(fname: str):
    '''
    If this is a slicer labelmap/segmentation, make sure you strip_slicer_strings first
    '''
    
    fname = strip_extension(fname)

    pt_token = "(_[a-z0-9]*)"
    tokens = re.findall(pt_token, fname)
    if (
        len(tokens) > 1
    ):  # this will by pass short filenames with single digit pt id confusing with slicer suffix _\d
        fname = strip_slicer_strings(fname)
    return fname


def drop_digit_suffix(fname: str):
    """
    Postprocessing will create multiple matches of the same case_id. This allows us to identify which case patches belong to
    """

    pat = "(_\d{1,3})?$"
    fname_cl = re.sub(pat, "", fname)
    return fname_cl


def info_from_filename(fname: str):
    """
    returns [proj_title,case_id,desc, ?all-else]
    """
    tags = ["proj_title","case_id", "date", "desc"]
    name = cleanup_fname(fname)

    parts = name.split("_")
    output_dic={}
    for key,val in zip(tags,parts):
        output_dic[key]=val
    return output_dic

def match_filenames(fname1:str,fname2:str):
    info1=info_from_filename(fname1)
    info2=info_from_filename(fname2)
    matched=all([val1==val2 for val1,val2 in zip(info1.values(),info2.values())])
    return matched

# %%
if __name__ == "__main__":
    name = "lits_11_20111509.nii"
    name2 = "lits_11.nii"
    name3 = "lits_11_20111509_jacsde3d_thick.nii"
    pt = "(-?label(_\d)?)|(_\d$)"
    name = "drli_005.nrrd"
    nm = strip_extension(name)
    nm = strip_slicer_strings(nm)

    pt = "(-?label(_\d)?)|(_\d$)"
    re.sub(pt, nm)
    rpt_pt = "([a-z0-9]*_)"
    st = "litq_11_20190927_2"
    re.findall(rpt_pt, st)
    re.sub(pt, "", st)
    # %%
    fname = "litq_40_20171117_1-label"
    pt_token = "(_[a-z0-9]*)"
    re.findall(pt_token, fname)
    pt = re.compile("(([a-z0-9]*_)*)((-label)?(_\d)?)$", re.IGNORECASE)
    jj = re.sub(pt, "", fname)
    print(jj)
    pt = re.compile("(-?label(_\d)?)|_.*(_\d$)", re.IGNORECASE)
