# %%
from __future__ import annotations

import torch
from utilz.imageviewers import ImageMaskViewer

# %%

if __name__ == "__main__":
    img_fn ="/r/datasets/preprocessed/lidc/lbd/spc_080_080_150_ric8c38fe68_ex000/images/lidc_0001.pt"
    lm_fn = "/r/datasets/preprocessed/lidc/lbd/spc_080_080_150_ric8c38fe68_ex000/lms/lidc_0001.pt"
    image = torch.load(img_fn,weights_only=False)
    lm = torch.load(lm_fn,weights_only=False)


    ImageMaskViewer([image,lm])
# %%
