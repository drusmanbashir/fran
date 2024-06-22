import collections

from bs4 import BeautifulSoup as BS
import tqdm,yaml
import pickle, os,json
from pathlib import Path
from typing import Union
import numpy as np
import ipdb
import torch
import SimpleITK as sitk
from fastcore.basics import patch_to
import pandas as pd

from fran.utils.string import get_extension, str_to_path

tr = ipdb.set_trace

# %%

@str_to_path()
def is_sitk_file(fn: Path):
    if fn.is_dir(): return False
    fn_name = fn.name
    sitk_exts = ".nii", ".nrrd"
    for ext in sitk_exts:
        if ext in fn_name:
            return True
    return False



@patch_to(Path)
def str_replace(self,str1,str2):
    interm = str(self)
    interm= interm.replace(str1,str2)
    return self.__class__(interm)

def convert_uint16_to_uint8(fname):
    x = np.load(fname)
    if x.dtype!=np.uint8:
        np.save(fname,x.astype(np.uint8))
    return fname,x.dtype

def save_file_wrapper(fnc):
    def _inner(object,filename, verbose=False,overwrite=True,makedir=True):
        if not isinstance(filename, Path): filename = Path(filename)
        if overwrite==True or not filename.exists():
            if not filename.parent.exists() and makedir == True:
                os.mkdir(filename.parent)
            if verbose==True:
                print("Saving to file: {}".format(filename))
        else:
            print("File {0} exists. Set overwrite to True if necessary".format(filename))
        return fnc(object, filename)
    return _inner


@save_file_wrapper
def save_np(object,filename):
    np.save(filename,object)

#
# def load(fnc):
#     def _inner(filename):
#         with open(filename, 'rb') as file:
#             object = fnc.load(file)
#             return object
#     return _inner
#
#
def load_file(*ar,**kwargs):
    def inner(func ):
        def wrapper (filename ):
            with open(filename,*ar,**kwargs) as f:
                contents = func(f)
                return contents
        return wrapper
    return inner
        
@load_file("r")
def load_json(filename): return json.load(filename)

@load_file("rb")
def load_pickle(filename): return pickle.load(filename)

@load_file('r')
def load_yaml(filename):
            output_dic_ = yaml.safe_load(filename)
            return output_dic_

def load_text(filename):
    with open(filename) as f:
        contents = f.readlines()
        contents = [a.strip() for a in contents]
        return contents


@load_file('r')
def load_xml(filename):
        return BS(filename,'xml')


def save_xml(itm, filename):
    with open(filename,'w') as f:
        f.write(itm)



def save_json(dictionary, filename):
    with open(filename,'w') as f:
        json.dump(dictionary,f)
def save_sitk(img: Union[torch.Tensor, np.ndarray, sitk.Image], output_filename, verbose=True):


    if isinstance(img,torch.Tensor): img = img.cpu().detach().numpy()
    if not isinstance(img,sitk.Image): img = sitk.GetImageFromArray(img)

    parent_folder =  Path(output_filename).parent
    if isinstance(output_filename,Path): output_filename= str(output_filename)
    if not parent_folder.exists(): maybe_makedirs(parent_folder)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_filename)
    writer.Execute(img)
    if verbose==True: print("Saved to file: {}".format(output_filename))


def np_to_ni(input_filename,output_filename, overwrite=False):
        if not output_filename.exists() or overwrite==True:
            img = np.load(input_filename)
            print("Saving file {}".format(output_filename))
            save_sitk(img,output_filename)
        else: print("File {} exists. Skipping".format(output_filename))


def is_filename(x:str):
    if not "/" in x:
        if "." in x:
            ext = x.split(".")[1]
            if len(ext)>3: return False
            else: return True
        else: return False

    else:
        return False


def maybe_makedirs(x:Union[list,tuple, str,Path]):
    def _inner(x):
        try:
            if not isinstance(x,Path):
                x = Path(x)
            if not x.exists() and not is_filename(str(x)): 
                print("Making folder {}".format(str(x)))
                os.makedirs(x)
        except ValueError as err:
            print(err.args)
            tr()
    if isinstance(x,list):
        [_inner(xx) for xx in x]
    else:
        _inner(x)

 
def dump(fnc, *args1):
    def _inner(object,filename,*args2,**kwargs):
        with open(filename,*args1,*args2,**kwargs) as fout:
            fnc.dump(object, fout)
    return _inner
      
save_pickle = dump(pickle,"wb")
save_json = dump(json,"w")
# load_pickle = load(pickle)

# %%
@str_to_path(0)
def load_dict(filename):
    def _inner(filename,ext):
                if ext == 'json': 
                    return load_json(filename)
                else:
                    return load_pickle(filename)

    try:
                ext =(filename.name.split(".")[1])
                return _inner(filename,ext)
    except:
                for ext in ["json", "pkl"]:
                    filename_ = filename.parent/(".".join([filename.name,ext]))
                    if filename_.exists(): return _inner(filename_,ext)
                raise FileNotFoundError

def save_dict(object,filename,sort=False):
    filename = str(filename).split(".")[0]
    try: 
        if sort==True: 
            object = collections.OrderedDict(sorted(object.items()))
        save_json(object,filename+".json")
    except: 
        os.remove(filename+".json")
        save_pickle(object,filename+".pkl")


def sitk_filename_to_numpy(fname):
            arr = sitk.ReadImage(str(fname))
            arr= sitk.GetArrayFromImage(arr)
            return arr



@str_to_path(0)
def load_image(fn):
        val_extensions={
            'np': np.load,
            'pt': torch.load,
            'nii.gz':sitk.ReadImage,
            'nii':sitk.ReadImage
        }
        for key,fnc in val_extensions.items():
            if (ext:=get_extension(fn))==key:
                return fnc(fn)
        print("Extension invalid: ".format(ext))


def rename_and_move_images_and_masks(project_title,img_files,mask_files=None,counter_start=0,dataset_subid:str="",ext=None,overwrite=False,log=True):
    def _to_sitk(im,subfolder,output_filename):
            output_filename_full = subfolder/output_filename   
            if not output_filename_full.exists() or overwrite==True:
                img_sitk= sitk.ReadImage(str(im))
                save_sitk(img_sitk,output_filename_full)
            else:
                print("File {} exists. Skipping".format(str(output_filename_full)))
    def _get_files_folders(i,img_files,mask_files):
        if mask_files:
            mask_file = mask_files[i] 
            img_file  = [im for im in img_files if im.parent ==mask_file.parent][0]
        else:
            mask_file = None
            img_file = img_files[i]
        subfolders = main_folder/("images"), main_folder/("lms")
        maybe_makedirs(subfolders)
        return img_file,mask_file, subfolders

    main_folder = img_files[0].parent
    if not ext: ext = "."+img_files[0].name.split(".")[1]
    pseudo_ids=[]
    new_ids = []
    total = len(mask_files)  if mask_files else len(img_files)
    for i in tqdm.tqdm(range(total)):
        counter = counter_start+i
        img_file,mask_file,subfolders = _get_files_folders(i,img_files,mask_files)
        pseudo_id = img_file.parent.parent.name
        new_id =project_title+"_"+ dataset_subid + str(counter).zfill(4)
        pseudo_ids.append(pseudo_id)
        new_ids.append(new_id)
        output_filename =new_id+ext
        if mask_file:
            for im,subfolder in zip([img_file,mask_file],subfolders):
                _to_sitk(im,subfolder,output_filename)
        else:
            _to_sitk(img_file,subfolders[0])

    if log==True:
        data  = {"pseudo_ids":pseudo_ids, "new_ids":new_ids}
        df = pd.DataFrame(data)
        csv_filename = main_folder/("case_ids.csv")
        print("Storing names to {}".format(csv_filename))
        df.to_csv(csv_filename,index = False)

def save_list(listi,filename:Path): 
        def write(item):
                        f.write(str(item)+"\n")

        def _inner_recursion(sub_list):
                    f.write("\n")
                    for item in sub_list:
                        if isinstance(item,Union[list,tuple]):
                            _inner_recursion(item)
                        else:
                            write(item)

        with open(filename,'w') as f:
                    _inner_recursion(listi)


# %%
if __name__ == "__main__":
     fn = "/s/nnUNet_preprocessed/nnunet/Task500_kits21/nnUNetPlansv2.1_plans_3D"
     a =      load_dict(fn)

# %%
