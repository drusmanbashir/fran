# %%
import ipdb
from utilz.fileio import maybe_makedirs
tr = ipdb.set_trace
import shutil


import re
import SimpleITK as sitk
import torch
from pathlib import Path
import pandas as pd
from utilz.string import dec_to_str, info_from_filename

# %%
#SECTION:-------------------- SORTING IMAGES_PENDING FOLDER--------------------------------------------------------------------------------------

df = pd.DataFrame(columns=["fn","thin","thick","too_thin"])
pat_1p5 = r"1p5|3p0"
pat_thick = r"5p0"
pat_too_thin= r"0p7|1p0"
fldr_1p5 = Path("/s/xnat_shadow/nodes/images_pending/thin_slice")
fldr_too_thin  = Path("/s/xnat_shadow/nodes/images_pending/1mm_or_less")
fldr_5p0= Path("/s/xnat_shadow/nodes/images_pending/5mm")
fldr = Path("/s/xnat_shadow/nodes/images_pending")
fn = fls[0]
thin = re.search(pat_1p5,fn.name)
thick= re.search(pat_thick,fn.name)
too_thin = re.search(pat_too_thin,fn.name)
assert not all([thin,thick,too_thin]), "Too many matches"
# %%
fls = [fn for fn in fldr.glob("*") if not fn.is_dir()]
for fn in fls:
    img = sitk.ReadImage(fn)
    thickness = img.GetSpacing()[-1]
    as_fl = dec_to_str(thickness)
    full = as_fl[0]+"p"+as_fl[1:]

    fn_out_name =  fn.name.split(".")[0]+"_"+full+".nii.gz"
    print ("{0} ---> {1}\n{2} ".format(thickness,full,fn_out_name))
    tr()
    fn_out = fn.parent/fn_out_name

# %%


images_done = Path("/s/xnat_shadow/nodes/images")
img_fns = list(images_done.glob("*"))
ids_done = [info_from_filename(fn.name)["case_id"] for fn in img_fns]    


# %%
# %%
#SECTION:-------------------- Moving done files to processed fldr--------------------------------------------------------------------------------------
fldr = Path("/s/xnat_shadow/nodes/images_pending/1mm_or_less/")
fldr_thin = Path("/s/xnat_shadow/nodes/images_pending/thin_slice/images")
preds_subfoldr = fldr_thin/("LITS-1230")
preds_fldr = Path("/s/fran_storage/predictions/nodes/LITS-1230")
preds = list(preds_fldr.glob("*"))
fls = list(fldr.glob("*"))
processed_fldr = fldr/("case_ids_processed_already")
maybe_makedirs(processed_fldr)


for fl in fls:
    cid = info_from_filename(fl.name)["case_id"]
    if cid in ids_done:
        print(fl)
        tr()

        fn_out = processed_fldr / fl.name
        shutil.move(fl,fn_out)


# %%
# %%
#SECTION:-------------------- COPYING PREDICTIONS TO PREDICTIONS SUBFOLDER INSIDE IMAGES FOR CAPESTART--------------------------------------------------------------------------------------
fls = list(fldr_thin.glob("*"))
for fl in fls:
    nm = fl.name
    pred_nm = preds_fldr/nm
    shutil.copyfile(pred_nm,preds_subfoldr/nm)
# %%

