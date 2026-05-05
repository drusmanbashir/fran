from fran.run.misc import view_image
from label_analysis.helpers import get_labels
from tqdm.auto import tqdm
import matplotlib
matplotlib.get_backend()
matplotlib.use("Qt5Agg")
from pathlib import Path

import torch


def fix_spatial_shape(img_fn: Path):
      img = torch.load(img_fn, weights_only=False)
      meta = img.meta
      meta["spatial_shape"] = [int(x) for x in meta["spatial_shape"]]
      img.meta = meta
      torch.save(img, img_fn)


# %%
if __name__ == "__main__":
    from fran.transforms.misc_transforms import ApplyBBoxd
    import os
    from utilz.helpers import multiprocess_multiarg
    from utilz.imageviewers import ImageMaskViewer
    import SimpleITK as sitk

# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
      data_folder = Path(
          "/r/datasets/preprocessed/kits2/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/"
      )
      fix  = Path("/r/datasets/preprocessed/kits2/fixed_spacing/spc_080_080_150/images")
      im_fn =  Path("/r/datasets/preprocessed/kits23/rbd/spc_080_080_150_54787144/images/kits23_00487.pt")
      lm_fn =  Path("/r/datasets/preprocessed/kits23/rbd/spc_080_080_150_54787144/lms/kits23_00487.pt")


      d = torch.load("/tmp/localiser_totalseg_abdpelvis_crops/totalseg_s0407.pt", weights_only=False)
      img = d["image"]          # torch.Tensor, cropped RAS 1mm array
      bbox = d["bbox"]
      print(img.shape, bbox)
      img = img.permute(2,0,1)
# %%
        img_fldr = Path("/r/datasets/preprocessed/kits23/rbd/spc_080_080_150_75d46294/images")
        lm_fldr = Path("/r/datasets/preprocessed/kits23/rbd/spc_080_080_150_75d46294/lms")

        img_fn = img_fldr/"kits23_00576.pt"
        lm_fn = lm_fldr/"kits23_00576.pt"

        ImageMaskViewer([img_fn,lm_fn])
# %%
        

# %%
# %%
#SECTION:-------------------- ImageViewer--------------------------------------------------------------------------------------
      img_fn = Path("/s/fran_storage/datasets/preprocessed/kits23/fixed_spacing/spc_080_080_150/images/kits23_00525.pt")
      lm_fn = Path("/s/fran_storage/datasets/preprocessed/kits23/fixed_spacing/spc_080_080_150/lms/kits23_00525.pt")
      im = torch.load(img_fn, weights_only=False)
      lm = torch.load(lm_fn, weights_only=False)

      im = im.permute(2,0,1)
      lm = lm.permute(2,0,1)
      ImageMaskViewer([im,lm])
# %%
      imgs = list(fix.glob("*.pt"))
      for img_fn in imgs:
          im = torch.load(img_fn, weights_only=False)
          print(im.shape)
          print(im.meta)
# %%

      pred_fldr = Path("/s/fran_storage/predictions/kits2/KITS-n7")
      img_fldr = data_folder / "images"
      lms_fldr = data_folder / "lms"
      imgs = sorted(img_fldr.glob("*.pt"))
      lms=sorted(lms_fldr.glob("*.pt"))
      preds = sorted(pred_fldr.glob("*.pt"))


      lmfn = '/media/UB/datasets/kits23/lms/kits23_00042.nii.gz'
      lm=sitk.ReadImage(lmfn)
      lm.GetSize()
      im =  '/media/UB/datasets/kits23/images/kits23_00042.nii.gz'
      im = sitk.ReadImage(im)
      im.GetSize()
      ImageMaskViewer([im,lm])
# %%

      img_fn = imgs[0]
      im = torch.load(img_fn, weights_only=False)
      aa = im.meta['spatial_shape']

      [print(type(a)) for a in aa]
# %%
      # multiprocess_multiarg expects iterable of arg-tuples/lists
      args = [(img,) for img in imgs]

      nproc = min(24, len(args), os.cpu_count() or 1)
      multiprocess_multiarg(
          func=fix_spatial_shape,
          arguments=args,
          num_processes=nproc,
          io=True,  # better for file I/O-heavy workloads
# %%
# %%
#SECTION:--------------------  LABELMAPS--------------------------------------------------------------------------------------
        fldr = Path("/s/xnat_shadow/litq/lms") 
        fns = list(fldr.glob("*"))
        len(fns)
        for fn in tqdm(fns):
            im = sitk.ReadImage(fn)
            print(get_labels(im))
# %%
#SECTION:-------------------- PT model dice comparson--------------------------------------------------------------------------------------

      img_fn = imgs[0]
      img_fn = [fn for fn in imgs if "0001" in fn.name][0]
      im = torch.load(img_fn, weights_only=False)
      lm_fn = img_fn.parent.parent/("lms")/img_fn.name
      lm = torch.load(lm_fn,weights_only=False)
      pred_fn = pred_fldr/lm_fn.name

      pred_fn = "/s/fran_storage/predictions/kits2/KITS-n7/kits23_00001.pt"
      pred = torch.load(pred_fn, weights_only=False)
      print(pred.shape)
      pred=  pred.squeeze(0)
      pred = pred.cpu().detach()
      ImageMaskViewer([im,pred], 'im')
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

