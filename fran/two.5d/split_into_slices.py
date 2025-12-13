# %%
import os
import re
import shutil
import torch
from pathlib import Path

from utilz.fileio import maybe_makedirs
from tqdm.auto import tqdm

def cases_to_separate_folders(src_dir):
    fnames = os.listdir(src_dir)
    pattern = re.compile(r"^(.*)_slice\d{3}.pt$")  # Captures prefix before "_slice###"
    for fname in pbar(fnames):
        if not os.path.isfile(os.path.join(src_dir, fname)):
            continue
        match = pattern.match(fname)
        if match:
            prefix = match.group(1)
            target_dir = os.path.join(src_dir, prefix)
            os.makedirs(target_dir, exist_ok=True)

            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(target_dir, fname)
            shutil.move(src_path, dst_path)
# %%


parent_fldr = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_ex070")
image_dir = parent_fldr / "images"
label_dir = parent_fldr / "lms"
image_out = parent_fldr/("slices/images")
label_out = parent_fldr/("slices/lms")

maybe_makedirs([image_out,label_out])
# %%
img_fns = list(image_dir.glob("*.pt"))
img_path =img_fns[0]

# %%
# Process each image fsorted(image_dir.glob("*.pt"))ile
imgs = sorted(image_dir.glob("*.pt"))
for img_path in tqdm(imgs):
    name = img_path.stem  # e.g., drli_001
    lbl_path = label_dir / f"{name}.pt"
    assert lbl_path.exists(), f"Missing label: {lbl_path}"
    image_out_subfolder = image_out/name

    img_vol = torch.load(img_path,weights_only=False)  # shape: [D, H, W]
    lbl_vol = torch.load(lbl_path,weights_only=False)  # shape: [D, H, W]

    assert img_vol.shape == lbl_vol.shape, f"Shape mismatch for {name}"

    for z in range(img_vol.shape[-1]):
        img_slice = img_vol[:,:,z].contiguous()
        lbl_slice = lbl_vol[:,:,z].contiguous()

        slice_name = f"{name}_slice{z:03d}.pt"
        torch.save(img_slice, image_out / slice_name)
        torch.save(lbl_slice, label_out / slice_name)


# %%
#SECTION:-------------------- MOVE SLICES INTO FOLDERS--------------------------------------------------------------------------------------
src_dir_imgs = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_plan9/slices/images"  # Change to your actual folder path
src_dir = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_plan9/slices/lms"  # Change to your actual folder path

image_out
cases_to_separate_folders(image_out)
cases_to_separate_folders(label_out)
