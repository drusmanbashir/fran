# %%
import os
import pickle

from fran.inference.helpers import (
    filter_existing_files,
    get_patch_spacing,
    infer_project,
    load_images_nifti,
    load_params,
)
from utilz.cprint import cprint
from utilz.helpers import chunks

os.environ["TORCHDYNAMO_DISABLE"] = "1"  # set as early as possible in the process

import ipdb
import torch
import torch._dynamo as dynamo
from fran.managers import Project
from tqdm.auto import tqdm as pbar
from utilz.stringz import headline

tr = ipdb.set_trace

from pathlib import Path

import numpy as np
import torch
from utilz.listify import listify
from fran.data.dataset import NormaliseClipd
from fran.managers.unet import UNetManager
from fran.trainers import checkpoint_from_model_id, write_normalized_ckpt
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import (
    KeepLargestConnectedComponentWithMetad,
    SaveMultiChanneld,
    SqueezeListofListsd,
    ToCPUd,
)
from fran.transforms.spatialtransforms import ResizeToMetaSpatialShaped
from lightning.fabric import Fabric
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.post.dictionary import AsDiscreteD
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import (
    CastToTyped,
    EnsureChannelFirstd,
    SqueezeDimd,
)
from utilz.dictopts import DictToAttr, fix_ast


if __name__ == "__main__":
# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>

    from fran.inference.common_vars import load_best_runs_yaml
    best_runs = load_best_runs_yaml(conf_fldr)


# %%
# SECTION:-------------------- LITSMC-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>

    run_litsmc = ["LITS-1007"]
    run_litsmc = ["LITS-999"]
    safe_mode = False
    bs = 5
    save_channels = False
    overwrite = True
    devices = [1]
    L = BaseInferer(
        run_litsmc[0],
        save_channels=save_channels,
        safe_mode=safe_mode,
        devices=devices,
        k_largest=1,
    )

# %%
    preds = L.run(
        imgs_crc,
        chunksize=1,
        overwrite=overwrite,
    )
# %%
    data = P.ds.data[0]
# %%
# SECTION:-------------------- TOTALSEG-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>

    save_channels = False
    overwrite = False
    overwrite = True

    devices = [0]

    patch_overlap = 0.5
# %%
    run = best_runs["totalseg"]["run_ids"][0]
    safe_mode = True

    debug_ = True
    debug_ = False
# %%

    T = BaseInferer(
        run,
        patch_overlap=patch_overlap,
        save_channels=save_channels,
        safe_mode=safe_mode,
        devices=devices,
        debug=debug_,
    )

# %%

    # preds = T.run(imgs_crc, chunksize=2, overwrite=overwrite)
    # imgs = kits_imgs
    # imgs = imgs_curvas

    imgs = nodes_imgs_training
    preds = T.run(imgs, chunksize=2, overwrite=overwrite)
# %%
    preds = T.run(img_nodes, chunksize=2, overwrite=overwrite)
    preds = T.run(img_nodes2, chunksize=2, overwrite=overwrite)
# %%
    print(getattr(T.N, "_torchdynamo_inline", None))
    case_id = "crc_CRC089"
    imgs_crc = [fn for fn in imgs_crc if case_id in fn.name]
    import torch._dynamo

    print(torch._dynamo.is_compiled_fn(T.N.forward))

    # datamodule_ %%
# %%
# %%
# SECTION:--------------------  NODES-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
    run = run_nodes2[0]
    run = run_nodes3[0]
    run_nodes = best_runs["nodes"]["run_ids"]
    run = run_nodes[0]


    debug_ = False

    save_channels = False
    safe_mode = True
    bs = 1
    overwrite = True
    devices = [0]
    save_channels = False

    T = BaseInferer(
        run,
        save_channels=save_channels,
        safe_mode=safe_mode,
        devices=devices,
        debug=debug_,
    )

# %%
    imgs = nodes_imgs_training
    preds = T.run(imgs, chunksize=2, overwrite=overwrite)
    preds[0]["pred"].meta
# %%
# SECTION:-------------------- TROUBLESHOOTING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>
    overwrite = True
    T.setup()
    imgs = img_nodes

# %%

    patch_overlap = 0.0
    mode = "constant"
    sw_device = "cuda"
    device = "cpu"
    T.inferer = SlidingWindowInferer(
        roi_size=T.params["configs"]["plan_train"]["patch_size"],
        sw_batch_size=bs,
        overlap=patch_overlap,
        mode=mode,
        progress=True,
        sw_device=sw_device,
        device=device,
    )
# %%
    if overwrite == False and (isinstance(imgs[0], str) or isinstance(imgs[0], Path)):
        imgs = T.filter_existing_preds(imgs)
    imgs = chunks(imgs, n_sized_chunks=4)
    # for imgs_sublist in imgs:
    #     output = T.process_imgs_sublist(imgs_sublist)
    imgs_sublist = imgs[0]
# %%

    # output = T.process_imgs_sublist(imgs_sublist)
    # return output

    data = T.load_images(imgs_sublist)
    T.prepare_data(data, collate_fn=None)
    # preds = T.predict()
# %%
    data = T.ds[0]
# %%
    T.create_postprocess_transforms(T.ds.transform)
    T.model.eval()
    print(getattr(T.N, "_torchdynamo_inline", None))
    iteri = iter(T.pred_dl)
    batch = next(iteri)
    batch = T.predict_inner(batch)
# %%

    # def predict_inner(T, batch):
    img = batch["image"]
    if T.devices != "cpu":
        img = img.cuda(non_blocking=True)
    if T.safe_mode:
        img = img.to("cpu")

    logits = T.inferer(inputs=img, network=T.model)  # [B,117,DS,H,W]
# %%
    logits = logits[0]  # model has deep supervision only 0 channel is needed
    # Collapse channels early; keep on same device
# %%
    if T.safe_mode == True or T.save_channels == False:
        labels = torch.argmax(logits, dim=1, keepdim=True)
        labels = labels.to(torch.uint8)
        batch["pred"] = labels
        del logits
    else:
        batch["pred"] = logits
    batch["pred"].meta = batch["image"].meta.copy()
# %%
    batch = T.postprocess_transforms(batch)
# %%
    #     if T.save == True:
    #         T.save_pred(batch)
    #     if T.safe_mode == True:
    #         T.reset()
    #         return None
    #     return batch
# %%

    """Process a subset of images using the data module"""
    T.pred_dl = T.data_module.setup(imgs_sublist)
    preds = T.predict()
    # output = T.postprocess(preds)
# %%
    T.create_postprocess_transforms()
    out_final = []
    # for batch in preds:
    batch = preds[0]
    tmp = T.postprocess_transforms(batch)
    out_final.append(tmp)
# %%
    if T.save:
        T.save_pred(output)
    if T.safe_mode:
        T.reset()
# %%

    T = En.Ps[0]
    Sq = SqueezeDimd(keys=["pred", "image"], dim=0)

    # below is expensive on large number of channels and on discrete data I am unsure if it uses nearest neighbours
    # I = Invertd(
    #     keys=["pred"], transform=T.ds.transform, orig_keys=["image"]
    # )  # watchout: use detach beforeharnd. make sure spacing are correct in preds
    A = Activationsd(keys="pred", softmax=True)
    DS = AsDiscreted(keys=["pred"], argmax=True)  # ,threshold=0.5)
    U = ToCPUd(keys=["image", "pred"])
    Sa = SaveMultiChanneld(
        keys=["pred"],
        output_dir=T.output_folder,
        output_postfix="",
        separate_folder=False,
    )
    I = ResizeToMetaSpatialShaped(keys=["pred"], mode="nearest")

# %%
    out_final = []
    if T.save_channels == True:
        tfms = [Sq, Sa, A, DS, I]
    else:
        tfms = [Sq, A, DS, I]
    if T.k_largest:
        K = KeepLargestConnectedComponentWithMetad(
            keys=["pred"], independent=False, num_components=T.k_largest
        )  # label=1 is the organ
        tfms.insert(-1, K)
    if T.safe_mode == True:
        tfms.insert(0, U)
    else:
        tfms.append(U)
# %%
    pred = batch["pred"]
    pred.meta["spatial_shape"]
# %%
    batch = Sq(batch)
    batch = A(batch)
    batch = DS(batch)
    batch = U(batch)
    batch = I(batch)

    x = torch.rand(1, 1, 128, 128, 96).to("cuda")
    output = T.model(x)
    [a.shape for a in output]

# %%
    if En.devices == "cpu":
        fabric_devices = "auto"
        accelerator = device = "cpu"
    else:
        fabric_devices = En.devices
        device_id = En.devices[0]
        device = torch.device(f"cuda:{device_id}")
        accelerator = "gpu"
    model = UNetManager.load_from_checkpoint(
        En.ckpt,
        plan=En.plan,
        project_title=En.project.project_title,
        dataset_params=En.dataset_params,
        strict=False,
        map_location=device,
        weights_only=False,
    )
    model.eval()
    fabric = Fabric(
        precision="bf16-mixed", devices=fabric_devices, accelerator=accelerator
    )
    En.model = fabric.setup(model)

# %%
    tfms_keys = T.preprocess_tfms_keys
    transform = T.tfms_from_dict(tfms_keys, T.preprocess_transforms_dict)
# %%
    batch["pred"].shape
# %%
