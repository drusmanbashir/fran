import torch
# %%
def collate_fn_paired_organs(batches):
    imgs, masks = [], []
    for img_mask_pairs in batches:
        for pair in img_mask_pairs:
            imgs.append(pair[0])
            masks.append(pair[1])
    imgs = torch.cat(imgs)
    masks = torch.cat(masks)
    return imgs, masks
