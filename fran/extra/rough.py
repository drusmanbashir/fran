# %%
from __future__ import annotations

import torch
from utilz.imageviewers import ImageMaskViewer

# %%

if __name__ == "__main__":
    image =  torch.load("img.pt",weights_only=False)
    pred =  torch.load("pred.pt",weights_only=False)

    n=0
    im = image[n,0].detach().cpu()
    ImageMaskViewer([image[n,0].detach().cpu(),pred[n,0].detach().cpu()])
# %%
