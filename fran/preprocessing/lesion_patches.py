# %%

import numpy as np
from monai.transforms import MapTransform


class BBoxCropd(MapTransform):
    def __init__(self, keys, margin=(8, 8, 8)):  # (mz,my,mx)
        super().__init__(keys)
        self.margin = np.array(margin, dtype=int)

    def __call__(self, data):
        d = dict(data)
        z0, z1 = int(d["z0"]), int(d["z1"])
        y0, y1 = int(d["y0"]), int(d["y1"])
        x0, x1 = int(d["x0"]), int(d["x1"])

        start = np.array([z0, y0, x0]) - self.margin
        end = np.array([z1, y1, x1]) + self.margin

        # after EnsureChannelFirstd, spatial dims are last 3
        shp = np.array(d["image"].shape[-3:])  # (D,H,W)
        start = np.maximum(0, start)
        end = np.minimum(shp, end)

        # crop image only (classification)
        img = d["image"]
        d["image"] = img[..., start[0] : end[0], start[1] : end[1], start[2] : end[2]]
        return d


# %%
if __name__ == "__main__":
    import os
    from pathlib import Path

    import pandas as pd
    import torch
    from fran.transforms.imageio import TorchReader
    from label_analysis.merge import get_labels
    from monai.data import DataLoader, Dataset
    from monai.networks.nets.resnet import resnet18
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        ResizeWithPadOrCropd,
    )

    bboxes_fn = Path(
        "/r/datasets/preprocessed/lidc/lbd/spc_075_075_075_rlb109adb5e_rlb109adb5e_ex000/bboxes_info.csv"
    )
    df = pd.read_csv(bboxes_fn)

    # ---- example dict from your bboxes_info.csv row 1 (second line shown earlier) ----
    sample = {
        "image": "/r/datasets/preprocessed/lidc/lbd/spc_075_075_075_rlb109adb5e_rlb109adb5e_ex000/images/lidc_0060.pt",
        "label": int(float(1.0)),  # from CSV "label"
        "z0": 0,
        "z1": 30,
        "y0": 13,
        "y1": 51,
        "x0": 170,
        "x1": 203,
    }

    # target model input size for ResNet3D
    patch_size = (96, 96, 96)

    tx = Compose(
        [
            LoadImaged(keys=["image"], reader=TorchReader),  # fill image path above
            EnsureChannelFirstd(keys=["image"]),  # -> C,D,H,W
            BBoxCropd(keys=["image"], margin=(12, 12, 12)),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=patch_size),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )

    # %%
    ds = Dataset(data=[sample], transform=tx)
    batch = next(iter(DataLoader(ds, batch_size=1)))

    x = batch["image"]  # shape: [B,C,D,H,W]
    y = batch["label"].long().view(-1)  # shape: [B], int class ids

    print("x shape:", x.shape)  # e.g. torch.Size([1,1,96,96,96])
    print("y:", y)  # e.g. tensor([1])

    img = x[0, 0]
    ImageMaskViewer([img.detach().cpu(), img])  # noqa: F821
    # %%
    # ResNet3D classification accepts this
    model = resnet18(
        spatial_dims=3,
        n_input_channels=x.shape[1],
        num_classes=2,  # set from inferred dataset classes
    )
    logits = model(x)
    print("logits shape:", logits.shape)  # [B, num_classes]

    # %%
    lmfln = "/s/fran_storage/predictions/lidc/LIDC-0021/"
    lms = os.listdir(lmfln)
    labs_all = []
    for lmfn in lms:
        lms = lmfln + "/" + lmfn
        lm = sitk.ReadImage(lms)  # noqa: F821
        llas = get_labels(lm)
        labs_all.append([llas, lms])
# %%
