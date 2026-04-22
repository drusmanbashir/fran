import numpy as np
import torch


def spacing_from_affine(affine) -> np.ndarray:
    affine_np = torch.as_tensor(affine).detach().cpu().numpy()
    spacing = np.sqrt((affine_np[:3, :3] ** 2).sum(axis=0))
    spacing = np.where(spacing > 0, spacing, 1.0)
    return spacing

