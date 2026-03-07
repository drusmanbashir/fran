
import os
from pathlib import Path

import torch
from utilz.helpers import multiprocess_multiarg


def fixer(img_fn: Path):
      img = torch.load(img_fn, weights_only=False)
      meta = img.meta
      meta["spatial_shape"] = [int(x) for x in meta["spatial_shape"]]
      img.meta = meta
      torch.save(img, img_fn)


# %%
if __name__ == "__main__":
      data_folder = Path(
          "/r/datasets/preprocessed/lidc/patches/spc_080_080_150_rspbb76320a_128128096"
      )
      img_fldr = data_folder / "images"
      imgs = sorted(img_fldr.glob("*.pt"))

      img = imgs[0]
      im = torch.load(img, weights_only=False)
      aa = im.meta['spatial_shape']

      [print(type(a)) for a in aa]
# %%
      # multiprocess_multiarg expects iterable of arg-tuples/lists
      args = [(img,) for img in imgs]

      nproc = min(24, len(args), os.cpu_count() or 1)
      multiprocess_multiarg(
          func=fixer,
          arguments=args,
          num_processes=nproc,
          io=True,  # better for file I/O-heavy workloads
      )# %%
#SECTION:-------------------- crop--------------------------------------------------------------------------------------

# %%
    lm_fn = img_fn.parent.parent/("lms")/img_fn.name
    im = torch.load(img_fn,weights_only=False)
    lm = torch.load(lm_fn,weights_only=False)

# %%
    lm = lm.permute(2,0,1)
    im = im.permute(2,0,1)

    ImageMaskViewer([im,lm])

# %%
    # ImageMaskViewer([im,lm])
    lm1 = lm.unsqueeze(0)
    im1 = im.unsqueeze(0)
    dici = {'image': im1, 'lm': lm1, 'image_meta' :im1.meta, 'lm_meta' :lm1.meta}
    lm1[lm1>0]=0
# %%

    patch_size = (128,96,96)
    select_fn = is_positive

    C = CropForegroundd(keys = ["image", "lm"], source_key="lm",select_fn=select_fn, margin=[30,0,12], allow_smaller=True)
    S = SpatialCropForegroundd(keys = ["image", "lm"], source_key="lm",spatial_size = patch_size, meta_keys =["image_meta", "lm_meta"], select_fn =is_positive, allow_smaller=False)
    C2 = CropForegroundMinShaped(keys = ["image", "lm"], source_key="lm", min_shape = patch_size, select_fn =is_positive)

    
# %%
    dic2 = C2(dici)
    dic2['image'].shape
    dic2['image'].shape
    ImageMaskViewer([dic2['image'][0], dic2['lm'][0]])
# %%

    dic3 = C(dici)
    dic3['image'].shape
    dic3['lm'].shape

    import h5py
    h5 = "/media/UB/datasets/lidc/fg_voxels.h5"
    f = h5py.File(h5,"r")
    dat = f['lidc_0376']
    dat.keys()


# %%
#SECTION:-------------------- DICE--------------------------------------------------------------------------------------
    pred = torch.load("predub.pt", weights_only=False)
    target = torch.load("targetub.pt", weights_only=False)
    pred.shape
    target.shape

    dsc = DiceLoss()
    print(dsc(pred, target))
