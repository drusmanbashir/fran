
# %%
import torch

N=10,
M=3
var = torch.tensor(0.09)
noise = torch.normal(mean=0, std=torch.sqrt(var))
# %%
