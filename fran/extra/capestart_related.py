
# %%
import os
from monai.transforms.io.dictionary import SaveImaged
from pathlib import Path
import shutil
from fastcore.basics import chunked
import ipdb
import SimpleITK as sitk
from monai.data.dataset import Dataset
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from torch.utils.data.dataloader import DataLoader
from xnat.helpers import collate_nii_foldertree

from fran.transforms.imageio import LoadSITKd
from fran.transforms.intensitytransforms import IntensityNorm
from fran.transforms.totensor import ToTensorT
from utilz.helpers import chunks, find_matching_fn, multiprocess_multiarg
from label_analysis.utils import  align_sitk_imgs, thicken_nii

from utilz.string import info_from_filename
tr = ipdb.set_trace
import shutil
import pandas as pd
import torch
from torch import nn
def find_files_from_list(partial_fns,img_fldr):
    fnames = []
    for f in partial_fns:
        for fn in Path(img_fldr).glob("*"):
            if str(f) in fn.name:
                fnames.append(fn)
    return fnames


def normalise_sitk_img(img_sitk):

    img = ToTensorT()(img_sitk).float()
    img = IntensityNorm()(img)

    img_sitk2 = sitk.GetImageFromArray(img)
    img_sitk2 = align_sitk_imgs(img_sitk2,img_sitk)
    return img_sitk2


def normalise_sitk_file(img_fn):

    img_sitk = sitk.ReadImage(img_fn)
    img_sitk2 = normalise_sitk_img(img_sitk)
    return img_sitk2, img_fn

def copy_to_folder(fn,outfolder,overwrite=False,ext='nii.gz'):
    name = fn.name
    pref= name.split(".")[1]
    neoname = pref+"."+ext
    outname = Path(outfolder)/neoname
    if not outname.exists() or overwrite==True:
        # img = sitk.ReadImage(fn)
        # sitk.WriteImage(img,outname)
        shutil.copy(fn,outname)
## collate nifti
# %%
if __name__ == "__main__":

    fldr = Path("/s/xnat_shadow/nodes/images")
    fldr_out = Path("/s/xnat_shadow/nodes/capestart/")
    img_fns = list(fldr.glob("*"))
# %%
    dicis= []
    for ind , img_fn in enumerate(img_fns):
        info_from_filename
        fn_name = "case_"+str(ind)+".nii.gz"
        fn_full = fldr_out/fn_name
        dici = {'fn_org':img_fn, 'fn_out':fn_full}
        dicis.append(dici)

    df = pd.DataFrame(dicis)

# %%

    data = [{'image': img_fn } for img_fn in df['fn_org'].to_list()]
    tfms =Compose( [LoadSITKd(keys=['image']), NormalizeIntensityd(keys=['image'])])
    ds = Dataset(data, transform =tfms)
    dl = DataLoader(ds,batch_size=10,collate_fn = dict_list_collated(['image']))
    S = SaveImaged(['image'],output_dir ="/s/xnat_shadow/nodes/capestart/",separate_folder=False,output_postfix="")
    for i , batch in enumerate(dl):
        images = batch['image']
        for img in images:
            dici = {'image':img}
            S(dici)
# %%
    # df.to_csv(fldr_out.parent/"capestart_pseudoids.csv",index=False)

    for i, row in df.iterrows():
        try:
            capest_imgs = list(fldr_out.glob("*"))
            fn_in = find_matching_fn(row['fn_org'],capest_imgs)
            fn_out = row.fn_out
            os.rename(fn_in,fn_out)
        except:
            print(row['fn_out'])

    # res = multiprocess_multiarg(normalise_sitk_file,args)

# %%


    fn_name = "case_"+str(ind)+".nii.gz"
    img_fn  = img_fns[ind]

    multiprocess_multiarg()




    



