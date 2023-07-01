from datetime import datetime
import re
from pathlib import Path
import ipdb
tr = ipdb.set_trace


def append_time(input_str, now=True):
    now = datetime.now()
    dt_string = now.strftime("_%d%m%y_%H%M")
    return input_str+dt_string

def infer_dataset_name(filename):
    pat ="^([^-_]*)"
    return pat, filename.name


def strip_extension(fname:str):
    exts = (".npy .nii.gz .nii .nrrd .pt".split(" "))
    for e in exts:
        pat = r'{}$'.format(e)
        fname_stripped = re.sub(pat,"",fname)
        if fname_stripped!=fname:
            return fname_stripped

def strip_slicer_strings(fname:str):
    '''
    This is on extension-less fnames. Use strip_extension before this.
    '''
    # pt = re.compile("(-?label(_\d)?)|_.*(_\d$)",re.IGNORECASE)
    pt = re.compile("(_\d)?$",re.IGNORECASE)
    fname_cl1 = fname.replace("-label","")
    fname_cl2=re.sub(pt,"",fname_cl1)
    return  fname_cl2

def cleanup_fname(fname:str):
    fname = strip_extension(fname)

    pt_token = "(_[a-z0-9]*)"
    tokens = re.findall(pt_token,fname) 
    if len(tokens)>1: # this will by pass short filenames with single digit pt id confusing with slicer suffix _\d
        fname = strip_slicer_strings(fname)
    return fname
        
def drop_digit_suffix(fname:str):
    '''
    Postprocessing will create multiple matches of the same case_id. This allows us to identify which case patches belong to
    '''
    
    pat = "(_\d{1,3})?$"
    fname_cl=re.sub(pat,"",fname)
    return fname_cl

def info_from_filename(fname:str, tag='pt_id'):
               '''
               returns [proj_title,pt_id, ?all-else]
               '''
               valid_tags = ['pt_id','date','desc','proj_title']
               assert tag in valid_tags, "Please select tag as one of {}".format(valid_tags) 
               name = fname.name
               name = cleanup_fname(name)
    
               parts = name.split("_")
               proj_title= parts[0]
               pt_id = "_".join(parts[:2])
               output = [parts[0],parts[1]]
               if len(parts)>2:
                    output.append("_".join(parts[2:]))
               return output

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
    re.sub(pt,nm)
    rpt_pt = "([a-z0-9]*_)"
    st = "litq_11_20190927_2"
    re.findall(rpt_pt,st)
    re.sub(pt,"",st)
# %%
    fname = 'litq_40_20171117_1-label'
    pt_token = "(_[a-z0-9]*)"
    re.findall(pt_token,fname) 
    pt = re.compile("(([a-z0-9]*_)*)((-label)?(_\d)?)$",re.IGNORECASE)
    jj = re.sub(pt,"",fname)
    print(jj)
    pt = re.compile("(-?label(_\d)?)|_.*(_\d$)",re.IGNORECASE)

