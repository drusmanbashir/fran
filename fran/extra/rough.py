# %%
from __future__ import annotations

import torch
from utilz.imageviewers import ImageMaskViewer

from pathlib import Path
# %%

if __name__ == "__main__":
    img_fn = =Path("/r/datasets/preprocessed/pancreas/lbd/spc_080_080_150_rica8920439_ex050/images/pancreasmsd07_061.pt")
    img_fn =Path("/r/datasets/preprocessed/pancreas/lbd/spc_080_080_150_rica8920439_ex050/images/CURVASPDAC_00303.pt")
    lm_fn = img_fn.parent.parent/("lms")/img_fn.name
    image = torch.load(img_fn,weights_only=False)
    lm = torch.load(lm_fn,weights_only=False)

    image=image.permute(2,0,1)
    lm = lm.permute(2,0,1)

    ImageMaskViewer([image,lm])
# %%
# %%
#SECTION:-------------------- H5py analysis--------------------------------------------------------------------------------------
    import h5py
    h5 = "/media/UB/datasets/lidc/fg_voxels.h5"
    f = h5py.File(h5,"r")
    dat = f['lidc_0376']
    dat.keys()



