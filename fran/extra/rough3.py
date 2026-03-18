# %%


if __name__ == "__main__":
    import torch
    from utilz.imageviewers import ImageMaskViewer

    # %%
    imgfn = "/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_plan2/images/nodes_56_41T410_CAP1p5SoftTissue.pt"
    # lmfn = "/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_plan2/lms/nodes_24_410813_ChestAbdoC1p5SoftTissue.pt"

    img = torch.load(imgfn, weights_only=False)
    img = img.permute(2, 0, 1)

    ImageMaskViewer([img, img], apply_transpose=True)
# %%
