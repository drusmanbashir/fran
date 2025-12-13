from monai.networks.nets import UNet
import torch

# %%


if __name__ == '__main__':
    x = torch.randn(1,1,256,256)
# %%
    net = UNet(spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
# %%
    y = net(x)
    
    y.shape
