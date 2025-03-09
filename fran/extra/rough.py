# %%
import shutil
import socket
from matplotlib import pyplot as plt
from torch import nn
from gudhi import CubicalComplex
import cudf
import cuml
import cugraph
from send2trash import send2trash

from utilz.helpers import info_from_filename
import torch
import time
import SimpleITK as sitk
import re
from pathlib import Path

from label_analysis.helpers import get_labels, view_sitk
from fran.managers.data import find_matching_fn, pbar
from utilz.fileio import maybe_makedirs
from utilz.imageviewers import ImageMaskViewer

# Step 8: Plot the persistence diagram
def plot_persistence_diagram(diagram):
        births, deaths = [], []
        for point in diagram:
            if point[0] == 0:  # 0-dim components (connected components)
                births.append(point[1][0])
                deaths.append(point[1][1])

        plt.figure(figsize=(8, 6))
        plt.scatter(births, deaths, label="0-dim Features (Connected Components)", color="blue", s=10)
        plt.plot([0, max(deaths)], [0, max(deaths)], linestyle='--', color='red')  # Diagonal
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.title("Persistence Diagram")
        plt.legend()
        plt.show()
# Step 7: Plot the persistence barcode
def plot_persistence_barcode(intervals, title="Persistence Barcode"):
    plt.figure(figsize=(10, 5))
    for i, (birth, death) in enumerate(intervals):
        if death == float('inf'):
            death = max(top_dimensional_cells)  # Set death to a max finite value for visualization
        plt.plot([birth, death], [i, i], 'b', lw=2)

    plt.xlabel("Filtration Value")
    plt.ylabel("Barcode Index")
    plt.title(title)
    plt.show()

# %%
if __name__ == "__main__":
    fn = "/s/xnat_shadow/crc/sampling/tensors/fixed_spacing/images/crc_CRC012_20180422_ABDOMEN.pt"
    lm = torch.load(fn)
    ImageMaskViewer([lm,lm])
    fldr = Path("/s/xnat_shadow/crc/tensors/fixed_spacing/lms/")
    fls = list(fldr.glob("*"))
# %%
    bad= []
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
    maybe_makedirs([out_fldr_lm,out_fldr_img])
# %%
    nodes_fldr = Path("/s/xnat_shadow/nodes/images_pending_neck")
    nodes_done_fldr = Path("/s/xnat_shadow/nodes/images")
    nodes_done = list(nodes_done_fldr.glob("*"))
    nodes = list(nodes_fldr.glob("*"))

# %%
    img_fn = Path("/s/fran_storage/misc/img.pt")
    lm_fn =  Path("/s/fran_storage/misc/lm.pt")
    lm = torch.load(lm_fn)

# %%
    pred_fn = Path("/s/fran_storage/misc/pred.pt")
    im = torch.load(img_fn)
    pred = torch.load(pred_fn)
    pred = nn.Softmax()(pred)
    thres = 0.01
# %%
    preds_bin = (pred>thres).float()
    preds_np = preds_bin.detach().numpy()
    ImageMaskViewer([im.detach(),preds_bin])
    ImageMaskViewer([im.detach(),pred])

# Step 3: Flatten the numpy array for GUDHI (top-dimensional cells are voxel values)
# GUDHI requires a flattened list of the voxel values for constructing the cubical complex.
# %%
    top_dimensional_cells = preds_np.flatten()

# Step 4: Define the dimensions of the 3D volume
    dimensions = preds_np.shape

# Step 5: Create a Cubical Complex
    cubical_complex = CubicalComplex(dimensions=dimensions, top_dimensional_cells=top_dimensional_cells)

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
    plt.scatter(birth_times, death_times, color='b', label='Connected Components')
    plt.plot([0, 10], [0, 10], 'r--', label='Diagonal (y = x)')  # Diagonal line

# Plot settings
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.title('Persistence Diagram for 0-Dimensional Connected Components')
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
        df = cudf.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
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
        gdf = cudf.DataFrame({'src': sources, 'dst': destinations})

        # Create a graph using cuGraph
        G = cugraph.Graph()
        G.from_cudf_edgelist(gdf, source='src', destination='dst')

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
            cid1,desc1 = ina['case_id'], ina['desc']
            for j, test_pend in enumerate(nodes):
                test_pend = nodes[j]
                into = info_from_filename(test_pend.name)
                cid2,desc2 = into['case_id'], into['desc']
                if cid1==cid2 :
                    print("Already processed", test_pend.name)
                    send2trash(test_pend)
# %%
# %%
            new_filename = re.sub(r'_\d{8}_', '_', im_fn.name)
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
                lm= to_binary(lm)

# %%
