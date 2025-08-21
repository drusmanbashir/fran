# %%
import ipdb
tr = ipdb.set_trace
from label_analysis.helpers import *

import shutil
import re
from utilz.imageviewers import ImageMaskViewer
from utilz.helpers import *
import SimpleITK as sitk
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.modules import CrossEntropyLoss
import pandas as pd
from utilz.string import dec_to_str

imgfn = "/r/datasets/preprocessed/totalseg/lbd/spc_100_100_100_plan4/images/totalseg_s0024.pt"
labfn = "/r/datasets/preprocessed/totalseg/lbd/spc_100_100_100_plan5/lms/totalseg_s0367.pt"
outputfn = "/s/tmp/CSA-Net/CSANet/outputs.pt"
img = torch.load(imgfn, weights_only=False)
lab = torch.load(labfn, weights_only=False)
output = torch.load(outputfn, weights_only=False)
print(lab.max())

img=img.permute(2,1,0)
lab=lab.permute(2,1,0)
# %%
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
# %%
import shutil
from matplotlib import pyplot as plt
from torch import nn

from gudhi.cubical_complex import CubicalComplex
import cudf
import cugraph
from send2trash import send2trash

from utilz.helpers import find_matching_fn, info_from_filename, pbar, relabel
import torch
import SimpleITK as sitk
import re
from pathlib import Path

from label_analysis.helpers import get_labels, to_binary
from utilz.fileio import load_dict, maybe_makedirs
from utilz.imageviewers import ImageMaskViewer


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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
        from cuml.linear_model import LinearRegression
        import numpy as np

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
            lm = to_binary(lm)
