# %%
import torch
from pathlib import Path
import os
import random
import ipdb
import matplotlib.pyplot as plt
from fastcore.basics import warnings
from monai.data.dataset import PersistentDataset
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from utilz.fileio import load_dict, load_yaml
from utilz.helpers import pp
from utilz.imageviewers import ImageMaskViewer

from fran.managers.project import Project
from fran.two.5d.datamanagers import DataManagerDual2
from fran.utils.common import *
from fran.configs.parser import ConfigMaker


tr = ipdb.set_trace
# %%

#SECTION: -------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR>
if __name__ == '__main__':
    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")
    torch.set_float32_matmul_precision("medium")
    project_title = "litsmc"
    proj_litsmc = Project(project_title=project_title)

    C= ConfigMaker(
        proj_litsmc
    )
    C.setup(1)
    conf_litsmc= C.configs
# %%

    project_title = "totalseg"
    proj_tot = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    )
    configuration_filename = None

    C2= ConfigMaker(
        proj_tot
    )
    C2.setup(1)
    conf_tot= C2.configs

    global_props = load_dict(proj_tot.global_properties_filename)

# %%
# SECTION:-------------------- LBD-------------------------------------------------------------------------------------- <CR> <CR> <CR>
    conf_litsmc['plan_train']['patch_size']=[256,256]
    batch_size = 8
    ds_type = "lmdb"
    # D = DataManagerDual2(
    #     project_title=proj_litsmc.project_title,
    #     config=conf_litsmc,
    #     batch_size=batch_size,
    #     ds_type=ds_type,
    # )
    #


    data_fldr= Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_ex070/slices")
    D = DataManagerDual2(
        project_title=proj_litsmc.project_title,
        config=conf_litsmc,
        batch_size=batch_size,
        ds_type=ds_type,
        data_folder=data_fldr
    )

    tm = D.train_manager
    tm.data_folder
    # D.train_manager.plan['patch_size']=[128,128]
    # D.valid_manager.plan['patch_size']

# %%
    D.prepare_data()
    D.setup()
    tm = D.train_manager

# %%
# %%
#SECTION:-------------------- TROUBLESHOOTING DL-------------------------------------------------------------------------------------- %%

# %%
    tmv = D.valid_manager
    dici = tmv.ds[2]
    for dici in tmv.ds:

        n= 0
        print(dici[n]['lm'].shape)
        print(dici[n]['image'].shape)
    dlv = tmv.dl
    # dl = DataLoader(tmv.ds, batch_size=2, num_workers=2)
# %%
    dlt = tm.dl
    for item in dlt:
        print("patch size:", item["image"].shape)
        print("patch size:", item["lm"].shape)
# %%
    dl = tm.dl
# %%
    for item in dl:
        print("patch size:", item["image"].shape)
        print("patch size:", item["lm"].shape)

        tr()
    # tm.transforms_dict
# %%
    ds = tm.ds
    dat = ds[1]
    tm.tfms_list
# SECTION:-------------------- TROUBLESHOOT COLLATE_FN-------------------------------------------------------------------------------------- <CR>

    ds = tm.ds
    dici = ds[0]
    dici2 = ds[1]

    batch = [dici, dici2]

    bb = tm.collate_fn(batch)
    bb["image"].shape
# %%
# SECTION:-------------------- TROUBLESHOOTING TRAIN DS-------------------------------------------------------------------------------------- <CR>
    # keys_tr = "L,Ld,E,Rtr,F1,F2,Affine,Re,N,IntensityTfms"

    keys_tr = ("L,Ld,E,Rtr,Re,Ex,N,IntensityTfms",)
    keys_val = ("L,Ld,E,Rva,Re,Ex, N",)
    dici = ds.data[18]
    dici = tm.transforms_dict["L"](dici)
    dici = tm.transforms_dict["Ld"](dici)
    dici = tm.transforms_dict["E"](dici)
    dici = tm.transforms_dict["Rtr"](dici)
    dici = tm.transforms_dict["Rva"](dici)
    dici = tm.transforms_dict["N"](dici)

# %%

    # keys_tr = "L,Ld,E,Rtr,F1,F2,Affine,Re,N,IntensityTfms"
    dici2 = ds.data[19]
    dici2 = tm.transforms_dict["L"](dici2)
    dici2 = tm.transforms_dict["Ld"](dici2)
    dici2 = tm.transforms_dict["E"](dici2)
    dici2 = tm.transforms_dict["Rtr"](dici2)
    dici2 = tm.transforms_dict["Rva"](dici2)
    dici2 = tm.transforms_dict["N"](dici2)
# %%
# SECTION:-------------------- TROUBLESHOOTING VALID DS-------------------------------------------------------------------------------------- <CR>

    ds = tmv.ds
    dici = ds.data[10]
    dici = tmv.transforms_dict["L"](dici)
    dici = tmv.transforms_dict["Ld"](dici)
    dici = tmv.transforms_dict["E"](dici)
    pp(dici["lm"].shape)
    # dici = tmv.transforms_dict['Rtr'](dici)
    dici = tmv.transforms_dict["Rva"](dici)
    dici = tmv.transforms_dict["N"](dici)

# %%
    pp(dici["image"].shape)
    pp(dici["image"].max())
    pp(dici["image"].min())
    pp(dici[0]["image"].shape)
    dici = tm.transforms_dict["Ex"](dici)
    pp(dici["lm"].shape)

# %%
    dici = Rtr(dici)
# %%
    dici["lm"].shape
# %%

    dici[0]["image"].shape
    dici[0]["lm"].shape
# %%
    dici["lm"].shape
    dici["lm_centre"].shape
# %%
    dici = dat
    # Extract tensors for visualization
    lm_tensor = dici[0]["image"]  # Shape: 3xHxW
    lm_centre_tensor = lm_tensor[1, :]

    print(f"lm tensor shape: {lm_tensor.shape}")
    print(f"lm_centre tensor shape: {lm_centre_tensor.shape}")

    # Create matplotlib visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Display the 3 channels of lm tensor
    for i in range(3):
        if i < 2:
            row, col = 0, i
        else:
            row, col = 1, 0

        axes[row, col].imshow(lm_tensor[i].cpu().numpy(), cmap="gray")
        axes[row, col].set_title(f"lm Channel {i}")
        axes[row, col].axis("off")

    # Display the single channel of lm_centre tensor
    axes[1, 1].imshow(lm_centre_tensor[0].cpu().numpy(), cmap="gray")
    axes[1, 1].set_title("lm_centre Channel 0")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

# %%
    D.train_ds[0]
    dlv = D.train_dataloader()
# %%

    iteri = iter(dlv)
    while iteri:
        batch = next(iteri)
        print(batch["image"].shape)

# %%

    n = 0
    im = batch["image"][n][0]
    ImageMaskViewer([im, batch["lm"][n][0]])
# %%
    ds1 = PersistentDataset(
        data=D.valid_manager.data,
        transform=D.valid_manager.transforms,
        cache_dir=D.valid_manager.cache_folder,
    )
    dici = ds1[0]
# %%

    #
# %%
# %%
# SECTION:-------------------- TROUBLESHOOTING-------------------------------------------------------------------------------------- <CR>

# SECTION:-------------------- ROUGH-------------------------------------------------------------------------------------- <CR> <CR> <CR>
    label_path = "/s/fran_storage/datasets/preprocessed/fixed_spacing/litsmc/spc_080_080_150/lms/drli_004.pt"
    label = torch.load(label_path, weights_only=False)
    img_path = "/s/fran_storage/datasets/preprocessed/fixed_spacing/litsmc/spc_080_080_150/images/drli_004.pt"
    img = torch.load(img_path, weights_only=False)
# %%

    H, W, D = label.shape
    dici = {"image": img, "lm": label}

    print(dici["lm"].shape)
# %%
    # Extract slices
    En = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
    E = ExtractContiguousSlicesd()
    dici = En(dici)
    print(dici["lm"].shape)

    dici = E(dici)
    print(dici["lm"].shape)
# %%
    z = random.randint(1, D - 2)
    label_z_minus = label[:, :, z - 1]  # shape: [1, H, W]
    label_z_minus.shape
    label_z = label[:, :, z]  # shape: [1, H, W]
    label_z.shape
    label_z_plus = label[:, :, z + 1]  #
    label_stack = torch.stack([label_z_minus, label_z_plus], dim=0).unsqueeze(
        0
    )  # [1, 2, H, W]
    label_stack = label_stack.float()
    label_stack.shape

    # interpolate along depth from 2 → 3
# %%
    lm_avg = (label_z_minus + label_z_plus) / 2
    lm_avg.shape

# %%
    # Display
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(label_z_minus, cmap="gray")
    axs[0].set_title(f"z-1 ({z-1})")

    axs[1].imshow(lm_avg, cmap="gray")
    axs[1].set_title(f"z ({z})")

    axs[2].imshow(label_z_plus, cmap="gray")
    axs[2].set_title(f"z+1 ({z+1})")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# %%
