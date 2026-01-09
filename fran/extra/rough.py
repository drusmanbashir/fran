# %%
import ipdb
import itk
from label_analysis.merge import LabelMapGeometry

tr = ipdb.set_trace
import importlib.resources
import re
import shutil
import sys
from pathlib import Path

import fran
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
from label_analysis.helpers import *
from torch.nn.modules import CrossEntropyLoss
from utilz.helpers import *
from utilz.imageviewers import ImageMaskViewer
from utilz.string import dec_to_str

set_autoreload()
base = os.path.dirname(fran.__file__)
rel = os.path.join(base,"cpp","build","debug")
sys.path.append(rel)

bad_names = "nodes_89_20190421_Abdomen3p0I30f3.pt,nodes_90_20201201_CAP1p5SoftTissue.pt,nodes_82_20210427_CAP1p5SoftTissue.pt,nodes_83_20210427_CAP1p5SoftTissue.pt,nodes_46_20220609_CAP1p5SoftTissue.pt,nodes_47_20220601_CAP1p5SoftTissue.pt,nodes_84_20211129_CAP1p5SoftTissue.pt,nodes_81_20210507_CAP1p5SoftTissue.pt,nodes_25_20201216_CAP1p5SoftTissue.pt,nodes_43_20220805_CAP1p5SoftTissue.pt,nodes_78_20210617_CAP1p5.pt"

# %%
#SECTION:-------------------- badl labels--------------------------------------------------------------------------------------
# %%
img_fn = "/s/xnat_shadow/nodes/lms/nodes_20_20190926_CAP1p5.nii.gz"
img_fn = "/tmp/plain_tensor.pt"
img_fn = "/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_ric03e8a587_ex050/lms/nodes_73_410705_CAP1p5Br383.pt"
img = torch.load(img_fn, weights_only=False)
type(img)
img.keys()
img["data"].shape
img['meta']
# %%
# LG = LabelMapGeometry(img_fn)
# LG.nbrhoods
fn2 = "/s/xnat_shadow/crc/lms/crc_CRC014_20190923_CAP1p5.nii.gz"
# %%
fn3 ="/s/xnat_shadow/crc/lms/crc_CRC004_20190425_CAP1p5.nii.gz"
img = sitk.ReadImage(fn3)
img_arr = sitk.GetArrayFromImage(img)

sp = img.GetSpacing()
org = img.GetOrigin()
# %%
# bb['bbox_stats'][-1]
bb1 = fh.numpy_to_bboxes(img_arr, sp, org, fn3)

df = pd.DataFrame(bb1['rows'])
print(df)
# %%
bb2 = fh.process_file_py(fn3)
df2 = pd.DataFrame(bb2['rows'])
print(df2)

# %%
LG2 = LabelMapGeometry(fn3)
df3 = LG2.nbrhoods

# %%
#SECTION:-------------------- checking pybind--------------------------------------------------------------------------------------
# %%



# %%
#SECTION:-------------------- tensor -> cpp--------------------------------------------------------------------------------------
class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])
# %%
fn =   "/r/datasets/preprocessed/lidc2/lbd/spc_080_080_150/lms/lidc2_0021.pt";
x = torch.randn(1, 3, 128, 128)      # Example tensor
torch.save(x,fn)
fn = "/tmp/cpp_tnsr.pt"
im = torch.load(fn,weights_only=False)
print(type(im))
imm = torch.Tensor(im)
torch.save(imm,"/home/ub/code/fran/fran/cpp/files/sample_tensor.pt")
# %%
mt = {
    "tnsr":im
}

C = torch.jit.script(Container(mt))
C.save("/home/ub/code/fran/fran/cpp/files/sample.pt")
# %%
#SECTION:-------------------- View torch images--------------------------------------------------------------------------------------
import fran.templates as tl

with importlib.resources.files(tl).joinpath("tune.yaml").open("r") as f:
    cfg = yaml.safe_load(f)
    base  = cfg.get("base")

# %%
img_fn = "/r/datasets/preprocessed/lidc/lbd/spc_080_080_150_ric8c38fe68_ex000/images/lidc_0011.pt"
lm_fn = "/r/datasets/preprocessed/lidc/lbd/spc_080_080_150_ric8c38fe68_ex000/lms/lidc_0011.pt"

fn = "/tmp/cpp_tnsr.pt"
aa = torch.Tensor(img)
torch.save(aa,fn)
img = torch.load(img_fn,weights_only=False)
lm = torch.load(lm_fn,weights_only=False)
ImageMaskViewer([img,lm])
# %%
# %%
#SECTION:-------------------- Read and view SITK images--------------------------------------------------------------------------------------
parent_fldr = Path("/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_ric03e8a587_ex050")

prd_patch1 = Path("/s/fran_storage/predictions/nodes/LITS-1288/nodes_78_410617_CAP1p5.nii.gz")
prd_fn_base = Path("/s/fran_storage/predictions/nodes/LITS-1230/nodes_78_410617_CAP1p5.nii.gz")
prd_fn_final = Path("/s/fran_storage/predictions/nodes/LITS-1290_LITS-1230_LITS-1288/nodes_78_410617_CAP1p5.nii.gz")
img_fn = Path("/s/xnat_shadow/nodes/images_pending/thin_slice/images/nodes_78_410617_CAP1p5.nii.gz")
img = sitk.ReadImage(img_fn)
print(img.GetSpacing())

# %%
lm_base = sitk.ReadImage(prd_fn_base)
lm_base.GetSpacing()
lm_patch1 = sitk.ReadImage(str(prd_patch1))
lm_patch1.GetSpacing()
# %%
prd_fn_patch = Path("/s/fran_storage/predictions/nodes/LITS-1290_LITS-1230_LITS-1288/nodes_78_410617_CAP1p5_LITS-1290.nii.gz")
lm_patch = sitk.ReadImage(prd_fn_patch)
lm_final = sitk.ReadImage(prd_fn_final)
lm_patch.GetSpacing()
lm_base.GetOrigin()
lm_patch.GetOrigin()
print(lm_final.GetSpacing())
# %%
img_fns = list(img_fldr.glob("*"))
lab_fns = list(lms_fldr.glob("*"))
# %%
n=2
img = torch.load(img_fns[n], weights_only=False)
lm = torch.load(lab_fns[n], weights_only=False)
ImageMaskViewer([img,lm])
# %%
lab_fn = "/r/datasets/preprocessed/totalseg/fixed_spacing/spc_100_100_100_rscr1/lms/totalseg_s0591.pt"
lab_fn = "/r/datasets/preprocessed/totalseg/fixed_spacing/spc_100_100_100_rscr1/images/totalseg_s0591.pt"

for img_fn, lab_fn in zip(img_fns,lab_fns):
    img = torch.load(img_fn, weights_only=False)
    lab = torch.load(lab_fn, weights_only=False)
    print(img.shape)
    assert img.shape == lab.shape, "shape mismatch"
    print(lab.max())
    if lab.max() >19:
        tr()
    median = img.median()
    mean = img.mean()
    print(mean,median)
# %%
# %%
#SECTION:-------------------- FILE type ?int--------------------------------------------------------------------------------------

lm_fn = Path("/s/fran_storage/predictions/nodes/LITS-1290/nodes_109_Ta50327_CAP1p5Soft.nii.gz")
fn = sitk.ReadImage(lm_fn)
fn.GetPixelID()
# %%
#SECTION:-------------------- H5 file check--------------------------------------------------------------------------------------
import h5py

h5fn = "/s/datasets_bkp/litsmall/fg_voxels.h5"
h5fn = "/s/datasets_bkp/lits_segs_improved/fg_voxels.h5"
f = h5py.File(h5fn)
pp(f.keys())
        # lab[lab>0]=1

        # torch.save(lab,lab_fn)
# %%
imgfn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/totalseg/spc_100_100_100_rscr1/images/totalseg_s0587.pt"
labfn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/totalseg/spc_100_100_100_rscr1/lms/totalseg_s0587.pt"
outputfn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/totalseg/spc_100_100_100_rscr1/lms/totalseg_s0587.pt"
# imgfn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/totalseg/spc_100_100_100_rscr1/images/totalseg_s0587.pt"
# labfn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/totalseg/spc_100_100_100_rscr1/lms/totalseg_s0587.pt"
img = torch.load(imgfn, weights_only=False,map_location='cpu')
lab = torch.load(labfn, weights_only=False,map_location="cpu")
lab.max()
print(lab.shape)
print(img.shape)
output = torch.load(outputfn, weights_only=False)
print(lab.max())

img=img.permute(2,1,0)
lab=lab.permute(2,1,0)
ImageMaskViewer([img,lab])
from pathlib import Path

# %%
import pandas as pd


def replace_key_in_first_column(
    root, 
    old_value="src_dest_labels", 
    new_value="remapping_train", 
    recursive=True
):
    root = Path(root)
    pattern = "**/*.xlsx" if recursive else "*.xlsx"
    files = sorted(root.glob(pattern))

    for f in files:
        print(f"Processing {f}")
        xls = pd.ExcelFile(f)
        updated = False
        out = {}

        for sheet in xls.sheet_names:
            # Read as strings to avoid losing exact text; don't auto-convert blanks to NaN
            df = pd.read_excel(f, sheet_name=sheet, dtype=str, keep_default_na=False)
            if df.shape[1] == 0:
                out[sheet] = df
                continue

            first_col = df.columns[0]
            # Normalize whitespace for comparison
            mask = df[first_col].astype(str).str.strip() == old_value
            if mask.any():
                df.loc[mask, first_col] = new_value
                print(f"  Updated '{sheet}': {mask.sum()} row(s)")
                updated = True

            out[sheet] = df

        # Only rewrite file if something changed
        if updated:
            with pd.ExcelWriter(f, engine="openpyxl", mode="w") as writer:
                for sheet, df in out.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)
        else:
            print("  No changes needed.")

# Example:
# replace_key_in_first_column("/path/to/folder")
# %%

# %%
#SECTION:-------------------- REMAPPING_TRAIN --------------------------------------------------------------------------------------

from pathlib import Path

import pandas as pd


def build_plans_sheet_for_file(xlsx_path: Path, plan_prefix="plan", out_sheet="plans"):
    xls = pd.ExcelFile(xlsx_path)
    plan_sheets = [s for s in xls.sheet_names if s.lower().startswith(plan_prefix.lower())]
    if not plan_sheets:
        print(f"No plan* sheets in {xlsx_path.name}")
        return

    # Collect all unique keys from col 1 across all plan sheets
    all_keys = []
    per_sheet_maps = {}

    for sheet in plan_sheets:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, dtype=str, keep_default_na=False)
        if df.shape[1] < 2:
            # Need at least two columns to map key->value
            per_sheet_maps[sheet] = {}
            continue

        # Clean leading/trailing whitespace
        df[0] = df[0].astype(str).str.strip()
        df[1] = df[1].astype(str).str.strip()

        # Drop empty keys in first column
        df = df[df[0] != ""].copy()

        # If duplicate keys exist in a sheet, keep the last occurrence
        kv = dict(zip(df[0].tolist(), df[1].tolist()))
        per_sheet_maps[sheet] = kv
        all_keys.extend(kv.keys())

    columns = sorted(set(all_keys))  # deterministic order

    # Build the output table: rows = sheet names, columns = all unique keys
    out_rows = []
    for sheet in plan_sheets:
        kv = per_sheet_maps.get(sheet, {})
        row = {"_sheet": sheet}
        for k in columns:
            row[k] = kv.get(k, "")
        out_rows.append(row)

    plans_df = pd.DataFrame(out_rows, columns=["_sheet"] + columns)

    # Write back: keep existing sheets, replace/create 'plans'
    # Re-read to preserve all non-plan sheets as-is
    sheets_dict = {s: pd.read_excel(xlsx_path, sheet_name=s) for s in xls.sheet_names if s != out_sheet}
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="w") as writer:
        # write original sheets (unchanged)
        for s, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=s, index=False)
        # write the new 'plans' sheet
        plans_df.to_excel(writer, sheet_name=out_sheet, index=False)

    print(f"Wrote '{out_sheet}' in {xlsx_path.name} with {len(plan_sheets)} rows and {len(columns)} columns.")

def build_plans_for_folder(root, recursive=True):
    root = Path(root)
    pattern = "**/*.xlsx" if recursive else "*.xlsx"
    for f in sorted(root.glob(pattern)):
        try:
            build_plans_sheet_for_file(f)
        except Exception as e:
            print(f"Failed on {f}: {e}")

# Example:
# build_plans_for_folder("/path/to/folder")
build_plans_for_folder(folder)
# %%
#SECTION:-------------------- torch lm lab values--------------------------------------------------------------------------------------
fldr = Path("/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_plan2")
fldr_lms= fldr/"lms"

lm_fns = list(fldr_lms.glob("*"))
img_fn = lm_fn.str_replace("lms","images")
img = torch.load(img_fn,weights_only=False)

lm_fn = lm_fns[0]
lm= torch.load(lm_fn,weights_only=False)
ImageMaskViewer([img,lm])
for lm_fn in lm_fns:
    lm = torch.load(lm_fn,weights_only=False)
    print(lm.max())
    if not lm.max() == 1:
        tr()
        # lm[lm>0]=1
        # torch.save(lm,lm_fn)
        
# ImageMaskViewer([img,lab])
# %%
#SECTION:-------------------- remapping sitk lms--------------------------------------------------------------------------------------
fldr = Path("/s/xnat_shadow/nodesthick/lms")
fls = list(fldr.glob("*"))
for fl in pbar(fls):
    print(fl)
    lm = sitk.ReadImage(fl)
    labs = get_labels(lm)
    if not labs ==[1]:
        lm = to_binary(lm)
        # id = lm.GetPixelID()
        sitk.WriteImage(lm,fl)
        # lm = relabel()
# %%
#SECTION:-------------------- ITK--------------------------------------------------------------------------------------
imgfn = "/s/fran_storage/predictions/totalseg/LITS-1238/nodes_40_20221205_CAP1p5SoftTissue.nii.gz"
img = sitk.ReadImage(imgfn)
img.GetDirection()
img.GetSpacing()
img_pt= sitk.GetArrayFromImage(img)
stats = sitk.StatisticsImageFilter()
stats.Execute(img)
mx = stats.GetMaximum()       # float or int depending on pixel type
mn = stats.GetMinimum()  
# %%
Imag_pteMaskViewer([img_pt,img_pt])
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
fls = list(fldr.glob("*"))
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
    shutil.move(fn,fn_out)
# %%
segs_folder = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_plan4/lms")
imgs_folder =  Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_plan4/images")
# %%
seg_fns= list(segs_folder.glob("*"))
img_fns = list(imgs_folder.glob("*"))
# %%
means = []
for imgfn in img_fns:
    img = sitk.ReadImage(imgfn)
    arr = sitk.GetArrayFromImage(img)
    means.append(arr.mean())
# %%

lmfn = seg_fns[0]
lmfn = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_plan4/lms/drli_075.pt")
imgfn = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_plan4/images/drli_075.pt")
img = torch.load(imgfn, weights_only=False)
lm = torch.load(lmfn, weights_only=False)
lm =lm.cpu()
img = img.cpu()
print(lm.shape)
print(img.shape)

ImageMaskViewer([img, lm])
# %%
n= 5

ImageMaskViewer([img[n][0], lm[n][0]])

# %%
imgfn = "/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_2/images/nodes_13_20230322_NCAPC.pt"
lmfn = "/r/datasets/preprocessed/tmp/lms/nodes_13_20230322_NCAPC.pt"
# %%
fldr = Path("/s/fran_storage/predictions/nodes/LITS-1159")
lmfns = list(fldr.glob("*"))
lmfn = lmfns[0]
lm = sitk.ReadImage(lmfn)
lma = sitk.GetArrayFromImage(lm)
lma.max()
ImageMaskViewer([img,lm])
# %%
print(f"Image shape: {img.shape}")
print(f"Output shape: {output.shape}")
print(f"Label shape: {lab.shape}")

# %%
# Visualize one sample (first sample, index 0)
sample_idx = 0

# Get one sample from each tensor
img_sample = img[sample_idx, 0]  # Shape: (224, 224) - single channel
output_sample = output[sample_idx]  # Shape: (5, 224, 224) - all 5 channels
lab_sample = lab[sample_idx]  # Shape: (224, 224)

# Create a grid: 1 image + 5 output channels + 1 label = 7 subplots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f"Sample {sample_idx}: Image, Output Channels, and Label", fontsize=16)

# Flatten axes for easier indexing
axes = axes.flatten()

# Plot original image
axes[0].imshow(img_sample.cpu().numpy(), cmap="gray")
axes[0].set_title("Input Image")
axes[0].axis("off")

# Plot all 5 output channels
for i in range(5):
    axes[i + 1].imshow(output_sample[i].cpu().numpy(), cmap="viridis")
    axes[i + 1].set_title(f"Output Channel {i}")
    axes[i + 1].axis("off")

# Plot label
axes[6].imshow(lab_sample.cpu().numpy(), cmap="jet")
axes[6].set_title("Label")
axes[6].axis("off")

# Hide the last unused subplot
axes[7].axis("off")

plt.tight_layout()
plt.show()

# %%

ce_loss = CrossEntropyLoss()
output_sample = output_sample.unsqueeze(0)
lab2 = lab[0].unsqueeze(0)
loss_ce = ce_loss(output_sample, lab2[:].long())
import re
# %%
import shutil
from pathlib import Path

import cudf
import cugraph
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from gudhi.cubical_complex import CubicalComplex
from label_analysis.helpers import get_labels, to_binary
from matplotlib import pyplot as plt
from send2trash import send2trash
from torch import nn
from utilz.fileio import load_dict, maybe_makedirs
from utilz.helpers import find_matching_fn, info_from_filename, pbar, relabel
from utilz.imageviewers import ImageMaskViewer

# %%
fn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/totalseg/spc_150_150_150/bboxes_info.pkl"
dici = load_dict(fn)
# Create a simple synthetic grayscale image with a horizontal gradient.
H, W = 64, 64
# Create an image with values ranging from 0 to 1 across each row.
image = np.tile(np.linspace(0, 1, W), (H, 1))
# Convert the numpy array to a PyTorch tensor and add batch and channel dimensions: (1, 1, H, W)
image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

##############################
# 1D Convolution on 2D Image
##############################
# We want to treat each row of the image as an independent 1D signal.
# For Conv1d, the expected input shape is (batch_size, channels, sequence_length).
# Here, we reshape the image so that each row is a separate sample:
# Resulting shape: (H, 1, W)
img_for_conv1d = image_tensor.squeeze(0).squeeze(0).unsqueeze(1)

# Define a 1D convolution layer with kernel size 3, padding to keep the same width.
conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
# Set the kernel to [-1, 0, 1] for horizontal edge detection.
with torch.no_grad():
    conv1d.weight[:] = torch.tensor([[[-1, 0, 1]]], dtype=torch.float32)

# Apply the 1D convolution to each row.
# The output shape will be (H, 1, W).
result1d = conv1d(img_for_conv1d)

##############################
# 2D Convolution on 2D Image
##############################
# Define a 2D convolution layer with kernel size 3 and padding.
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
# Create a 2D kernel that applies the same horizontal filter only in the middle row.
kernel_2d = torch.zeros((1, 1, 3, 3))
kernel_2d[0, 0, 1, :] = torch.tensor([-1, 0, 1], dtype=torch.float32)
with torch.no_grad():
    conv2d.weight[:] = kernel_2d

# Apply the 2D convolution to the original image tensor.
# The output shape will be (1, 1, H, W).
result2d = conv2d(image_tensor)

##############################
# Display the Results
##############################
plt.figure(figsize=(12, 4))
# Original image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray", aspect="auto")
plt.title("Original Image")
plt.axis("off")

# Result of 1D convolution
plt.subplot(1, 3, 2)
# Squeeze the channel dimension: shape becomes (H, W)
plt.imshow(result1d.squeeze(1).detach().numpy(), cmap="gray", aspect="auto")
plt.title("1D Convolution (per row)")
plt.axis("off")

# Result of 2D convolution
plt.subplot(1, 3, 3)
plt.imshow(result2d.squeeze().detach().numpy(), cmap="gray", aspect="auto")
plt.title("2D Convolution")
plt.axis("off")

plt.tight_layout()
plt.show()


# %%
#
# Step 8: Plot the persistence diagram
def plot_persistence_diagram(diagram):
    births, deaths = [], []
    for point in diagram:
        if point[0] == 0:  # 0-dim components (connected components)
            births.append(point[1][0])
            deaths.append(point[1][1])

    plt.figure(figsize=(8, 6))
    plt.scatter(
        births,
        deaths,
        label="0-dim Features (Connected Components)",
        color="blue",
        s=10,
    )
    plt.plot(
        [0, max(deaths)], [0, max(deaths)], linestyle="--", color="red"
    )  # Diagonal
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title("Persistence Diagram")
    plt.legend()
    plt.show()


# Step 7: Plot the persistence barcode
def plot_persistence_barcode(intervals, title="Persistence Barcode"):
    plt.figure(figsize=(10, 5))
    for i, (birth, death) in enumerate(intervals):
        if death == float("inf"):
            death = max(
                top_dimensional_cells
            )  # Set death to a max finite value for visualization
        plt.plot([birth, death], [i, i], "b", lw=2)

    plt.xlabel("Filtration Value")
    plt.ylabel("Barcode Index")
    plt.title(title)
    plt.show()


# %%
if __name__ == "__main__":
    fn = "/s/xnat_shadow/crc/sampling/tensors/fixed_spacing/images/crc_CRC012_20180422_ABDOMEN.pt"
    lm = torch.load(fn)
    ImageMaskViewer([lm, lm])
    fldr = Path("/s/xnat_shadow/crc/tensors/fixed_spacing/lms/")
    fls = list(fldr.glob("*"))
    # %%
    bad = []
    for fn in fls:
        lm = torch.load(fn)
        if "filename_or_obj" in lm.meta.keys():
            print("Pass")
        else:
            bad.append(fn)
    # %%
    fn = "/s/xnat_shadow/crc/tensors/fixed_spacing/lms/crc_CRC016_20190121_CAP11.pt"

    lm = torch.load(fn)
    lm.meta
    # %%
    fldr = Path("/s/xnat_shadow/crc/lms/")
    img_fldr = Path("/s/xnat_shadow/crc/images/")
    lm_fns = list(fldr.glob("*"))

    out_fldr_img = Path("/s/crc_upload/images")
    out_fldr_lm = Path("/s/crc_upload/lms")
    maybe_makedirs([out_fldr_lm, out_fldr_img])
    # %%
    nodes_fldr = Path("/s/xnat_shadow/nodes/images_pending_neck")
    nodes_done_fldr = Path("/s/xnat_shadow/nodes/images")
    nodes_done = list(nodes_done_fldr.glob("*"))
    nodes = list(nodes_fldr.glob("*"))

    # %%
    img_fn = Path("/s/fran_storage/misc/img.pt")
    lm_fn = Path("/s/fran_storage/misc/lm.pt")
    lm = torch.load(lm_fn)

    # %%
    pred_fn = Path("/s/fran_storage/misc/pred.pt")
    im = torch.load(img_fn)
    pred = torch.load(pred_fn)
    pred = nn.Softmax()(pred)
    thres = 0.01
    # %%
    preds_bin = (pred > thres).float()
    preds_np = preds_bin.detach().numpy()
    ImageMaskViewer([im.detach(), preds_bin])
    ImageMaskViewer([im.detach(), pred])

    # Step 3: Flatten the numpy array for GUDHI (top-dimensional cells are voxel values)
    # GUDHI requires a flattened list of the voxel values for constructing the cubical complex.
    # %%
    top_dimensional_cells = preds_np.flatten()

    # Step 4: Define the dimensions of the 3D volume
    dimensions = preds_np.shape

    # Step 5: Create a Cubical Complex
    cubical_complex = CubicalComplex(
        dimensions=dimensions, top_dimensional_cells=top_dimensional_cells
    )

    # Step 6: Compute the persistence
    cubical_complex.persistence()

    # Step 7: Get the persistence diagram

    persistence_intervals = cubical_complex.persistence_intervals_in_dimension(0)
    betti_0 = len(persistence_intervals)
    # %%
    plot_persistence_barcode(persistence_intervals)
    # Plotting the persistence diagram

    # ImageMaskViewer([lm.detach(),pred_bin.detach()], 'mm')
    # %%
    # %%
    import matplotlib.pyplot as plt

    # Data for the persistence diagram
    birth_times = [0, 0, 0]  # All components are born at the start of filtration
    death_times = [10, 10, 10]  # Persist until the end (using a high value)

    # Create the persistence diagram
    plt.figure(figsize=(8, 6))
    plt.scatter(birth_times, death_times, color="b", label="Connected Components")
    plt.plot([0, 10], [0, 10], "r--", label="Diagonal (y = x)")  # Diagonal line

    # Plot settings
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title("Persistence Diagram for 0-Dimensional Connected Components")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    # Show plot
    plt.show()

    # 1. Test cuDF Installation
    try:
        print("Testing cuDF...")
        # Create a simple cuDF DataFrame
        df = cudf.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        print("cuDF DataFrame:\n", df)
        print("cuDF head operation successful:", df.head())
        print("cuDF installed successfully.\n")
    except Exception as e:
        print("cuDF test failed:", e)

    # 2. Test cuML Installation
    try:
        print("Testing cuML...")
        import numpy as np
        from cuml.linear_model import LinearRegression

        # Create some simple data
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X, y)
        print("cuML Linear Regression coefficient:", model.coef_)
        print("cuML installed successfully.\n")
    except Exception as e:
        print("cuML test failed:", e)

    # 3. Test cuGraph Installation
    try:
        print("Testing cuGraph...")
        import cudf
        import cugraph

        # Create a simple graph edge list using cuDF
        sources = [0, 1, 2, 3]
        destinations = [1, 2, 3, 4]
        gdf = cudf.DataFrame({"src": sources, "dst": destinations})

        # Create a graph using cuGraph
        G = cugraph.Graph()
        G.from_cudf_edgelist(gdf, source="src", destination="dst")

        # Run connected components
        df = cugraph.connected_components(G)
        print("cuGraph Connected Components:\n", df)
        print("cuGraph installed successfully.\n")
    except Exception as e:
        print("cuGraph test failed:", e)

    print("RAPIDS installation test completed.")
    # %%
    for i in range(len(nodes_done)):
        # print("Filename ", node_done)
        node_done = nodes_done[i]
        ina = info_from_filename(node_done.name)
        cid1, desc1 = ina["case_id"], ina["desc"]
        for j, test_pend in enumerate(nodes):
            test_pend = nodes[j]
            into = info_from_filename(test_pend.name)
            cid2, desc2 = into["case_id"], into["desc"]
            if cid1 == cid2:
                print("Already processed", test_pend.name)
                send2trash(test_pend)
        # %%
        # %%
        new_filename = re.sub(r"_\d{8}_", "_", im_fn.name)
        out_lm_fname = out_fldr_lm / new_filename
        out_img_fname = out_fldr_img / new_filename
        shutil.copy(im_fn, out_img_fname)
        if not ".nii.gz" in lm_fn.name:
            lm = sitk.ReadImage(str(lm_fn))
            sitk.WriteImage(lm, out_lm_fname)
        else:
            shutil.copy(lm_fn, out_lm_fname)
        # %%
        lm = sitk.ReadImage(str(lm_fn))
        labels = get_labels(lm)
        if not labels == [1]:

# %%
#SECTION:-------------------- Gradients--------------------------------------------------------------------------------------
        from torch import nn
        x = torch.tensor([0.0,2], requires_grad=True)
        y = (x>0.5).float()



        class HardT(nn.Module):
            def forward(self,x):
                return (x>0.5).float()
# %%
        class STET(torch.autograd.Function):
            @staticmethod
            def forward(ctx,x):
                return (x>0.5).float()  
            @staticmethod
            def backward(ctx,grad_output):
                return grad_output
# %%
        x= torch.tensor([0.4], requires_grad=True)
        ste_t = STET.apply
        y_hard =HardT()(x)
        loss_h = (y_hard - 1)**2
        loss_h.backward()
        x.grad

# %%
        y_ste = ste_t(x)
        loss_s = (y_ste - 1)**2
        loss_s.backward()
        x.grad
# %%
