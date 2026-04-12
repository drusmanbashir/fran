# %%
import ray
import torch
from fran.localiser.preprocessing.data3d.nii2pt import (
    PreprocessorNII2PT3D,
    _PreprocessorNII2PTWorkerBase3D,
)
from fran.localiser.transforms import MultiRemapsTSL, NormaliseZeroToOne
from fran.transforms.imageio import LoadTorchd
from fran.transforms.intensitytransforms import MakeBinary
from fran.transforms.misc_transforms import BoundingBoxesYOLOd
from monai.apps.detection.transforms.dictionary import ConvertBoxToStandardModed
from monai.transforms.croppad.dictionary import BoundingRectd
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped


class DynamicResized(Resized):
    def __init__(self, keys, mode, min_size):
        assert len(min_size) == 3, (
            "min_size should be a tuple of (min_height, min_width, min_depth)"
        )
        self.min_size = min_size
        super().__init__(keys=keys, spatial_size=None, mode=mode)

    def _get_spatial_size(self, img):
        if img.ndim == 4:
            shpe = img.shape[1:]  # (C, H, W, D)
        else:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}. Expected 4")
        return shpe

    def __call__(self, data, lazy=None):
        img = data[self.keys[0]]
        spatial_size = self._get_spatial_size(img)
        new_size = [min(s, m) for s, m in zip(spatial_size, self.min_size)]
        self.resizer.spatial_size = new_size
        return super().__call__(data, lazy=lazy)


class _NII2PTTSLWorkerBase3D(_PreprocessorNII2PTWorkerBase3D):
    def __init__(self, output_folder, max_output_size, device="cpu", debug=False):
        assert len(max_output_size) == 3, (
            "max_output_size should be a tuple of (max_height, max_width, max_depth)"
        )
        self.max_output_size = max_output_size
        super().__init__(output_folder, device, debug)

    def worker_tfms_keys(self):
        return "L,E,O,DynRes,Remap"

    def create_transforms(self, device="cpu"):
        self.label_key = "label"
        self.lm_key = "lm"
        self.image_key = "image"

        super().create_transforms(device=device)

        self.box_key = "lm_bbox"
        self.DynRes = DynamicResized(
            keys=[self.image_key, self.lm_key],
            mode=["trilinear", "nearest"],
            min_size=self.max_output_size,
        )

        self.Ld = LoadTorchd([self.image_key, self.label_key])
        self.E = EnsureChannelFirstd(keys=[self.image_key])
        self.ToBinary = MakeBinary([self.label_key])
        self.Et = EnsureTyped(keys=[self.image_key], dtype=torch.float32)
        self.Et2 = EnsureTyped(keys=[self.label_key], dtype=torch.long)
        self.E2 = EnsureTyped(keys=[self.image_key], dtype=torch.float16)
        self.ExtractBbox = BoundingRectd(keys=[self.label_key])
        self.CB = ConvertBoxToStandardModed(mode="xxyy", box_keys=[self.box_key])
        self.YoloBboxes = BoundingBoxesYOLOd(
            [self.box_key],
            2,
            key_template_tensor=self.label_key,
            output_keys=["bbox_yolo"],
        )
        self.N = NormaliseZeroToOne(keys=[self.image_key])
        self.Remap = MultiRemapsTSL(lm_key=self.lm_key)
        self.transforms_dict["Remap"] = self.Remap
        self.transforms_dict["DynRes"] = self.DynRes


@ray.remote(num_cpus=1)
class TSLWorker3D(_NII2PTTSLWorkerBase3D):
    pass


class TSLWorkerLocal3D(_NII2PTTSLWorkerBase3D):
    pass


class PreprocessorNII2PTTSL3D(PreprocessorNII2PT3D):
    def __init__(self, data_folder, output_folder, max_output_size):
        self.max_output_size = max_output_size
        super().__init__(data_folder, output_folder)
        self.actor_cls = TSLWorker3D
        self.local_worker_cls = TSLWorkerLocal3D

    def build_worker_kwargs(self, device, debug):
        return {
            "max_output_size": self.max_output_size,
            "output_folder": self.output_folder,
            "device": device,
            "debug": debug,
        }


# %%
if __name__ == "__main__":
    from pathlib import Path

    import lightning as L
    import pandas as pd

# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
    import torch
    from fran.data.dataregistry import DS
    from utilz.fileio import maybe_makedirs
    from utilz.imageviewers import ImageMaskViewer

    data_folder = "/s/fran_storage/datasets/raw_data/lidc"
    data_folder = DS["totalseg"].folder
    img_fns = sorted((Path(data_folder) / "images").glob("*.nii.gz"))
    img_fn = img_fns[0]
    lm_fn = Path(data_folder) / "lms" / img_fn.name
    output_folder = "/s//tmp/nii2pt_tsl3d_debug"
# %%

    img_dir = Path(output_folder) / "images"
    lm_dir = Path(output_folder) / "lms"
    maybe_makedirs([img_dir, lm_dir])
    P = PreprocessorNII2PTTSL3D(
        data_folder=data_folder,
        output_folder=output_folder,
        max_output_size=(256, 256, 256),
    )
    P.setup()
    P.process()
# %%
    mini_df = pd.DataFrame(
        [
            {
                "case_id": "lidc_0001",
                "image": img_fn,
                "lm": lm_fn,
            },
            {
                "case_id": "lidc_0002",
                "image": img_fns[1],
                "lm": Path(data_folder) / "lms" / img_fns[1].name,
            },
        ]
    )

# %%
    def print_dici(dici, label):
        print("\n" + "=" * 80)
        print(label)
        for key, value in dici.items():
            if hasattr(value, "shape"):
                print(
                    key,
                    type(value).__name__,
                    "shape=",
                    tuple(value.shape),
                    "dtype=",
                    value.dtype,
                )
                if hasattr(value, "meta") and "filename_or_obj" in value.meta:
                    print("  filename_or_obj:", value.meta["filename_or_obj"])
            else:
                print(key, value)

# %%
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 100)

    print(mini_df)
    ind = 0
    row = mini_df.iloc[ind]
    w._process_row(row)

    dici = {"image": row["image"], "lm": row["lm"]}
    L = w.transforms_dict["L"]
    E = w.transforms_dict["E"]
    O = w.transforms_dict["O"]

    R = w.transforms_dict["Resize"]

# %%

    dici = dici2
    img = dici["image"]
    if img.ndim == 4:
        sz = img.shape[1:]
    else:
        sz = img.shape
# %%
    output_sz = []
    for s in sz:
        if s < 256:
            output_sz.append(s)
        else:
            output_sz.append(256)
    Remap = w.transforms_dict["Remap"]
    dici = w.transforms(dici)

# %%
    row0 = mini_df.iloc[0]
    row1 = mini_df.iloc[1]
    dici1 = {"image": row1["image"], "lm": row1["lm"]}
    print_dici(dici1, "Before transforms")
    dici2 = L(dici1)
    dici2 = E(dici2)
    dici2 = O(dici2)
    print_dici(dici2, "After LEO")

# %%
    dici3 = rz(dici2)
    dici3["lm"].shape
    dici4 = Remap(dici3)

    print_dici(dici3, "After Remap")
    img = dici4["image"]
    lm = dici4["lm"]
    ImageMaskViewer([img, lm], "im")
# %%
    img.shape[-1]
    ss = img.shape[1]
    nls = ss // 3
    sls = (slice(0, nls), slice(nls, 2 * nls), slice(2 * nls, ss))

    v1 = [-450.0, 1050.0]
    v2 = [-1350.0, 150.0]
# %%
    img1 = torch.clamp(img, v1[0], v1[1])
    img11 = img1[:, sls[0], ...].mean(dim=1)
    img12 = img1[:, sls[1], ...].mean(dim=1)
    img13 = img1[:, sls[2], ...].mean(dim=1)

# %%

# %%
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img11[0].cpu())
    ax[1].imshow(img12[0].cpu())
    ax[2].imshow(img13[0].cpu())

# %%
    print("print_dici(dici0, 'after L')")
    print("dici0 = E(dici0)")
    print("dici0 = O(dici0)")
    print("dici0 = Remap(dici0)")
# %%

    ss = lm.sha

    src_3d = DS["totalseg"].folder
    out_fldr = Path("/s/xnat_shadow/totalseg3d")
    out_pt = out_fldr / "pt"
    P = PreprocessorNII2PTTSL3D(src_3d, out_pt)
    P.setup(device="cpu", num_processes=8, debug=False)
    P.process()
# %%
    imgfn = "/s/xnat_shadow/totalseg3d/pt/images/totalseg_s0717.pt"
    lmfn = "/s/xnat_shadow/totalseg3d/pt/lms/totalseg_s0717.pt"

    img = torch.load(imgfn, weights_only=False)
    lm = torch.load(lmfn, weights_only=False)

    ImageMaskViewer([img[0], lm[0]], "im")
# %%

