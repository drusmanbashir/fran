# %%
import torch
from monai.networks.nets import DynUNet
import ipdb
tr = ipdb.set_trace


def get_kernel_strides(patch_size,spacings):
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, patch_size)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(patch_size, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {patch_size[idx]} in the spatial dimension {idx}."
                )
        patch_size = [i / j for i, j in zip(patch_size, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

class DynUNet_UB(DynUNet):
    # def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size: Sequence[Union[Sequence[int], int]], strides: Sequence[Union[Sequence[int], int]], upsample_kernel_size: Sequence[Union[Sequence[int], int]], filters: Optional[Sequence[int]] = None, dropout: Optional[Union[Tuple, str, float]] = None, norm_name: Union[Tuple, str] = ..., act_name: Union[Tuple, str] = ..., deep_supervision: bool = False, deep_supr_num: int = 1, res_block: bool = False, trans_bias: bool = False):
    #     super().__init__(spatial_dims, in_channels, out_channels, kernel_size, strides, upsample_kernel_size, filters, dropout, norm_name, act_name, deep_supervision, deep_supr_num, res_block, trans_bias)


    def forward(self, x):
        out = self.skip_layers(x)
        out = self.output_block(out)
        if self.deep_supervision:
            out_all = [out]
            for feature_map in self.heads:
                out_all.append(feature_map)
            return out_all
        return out
    #
    # def forward(self,x):
    #     x = super().forward(x)
    #     x = x.unbind(1)
    #     return x
    #
# %%
if __name__ == "__main__":
    from torchinfo import summary
    patch_size = [192,192,96]
    spacings=[1,1,1]
    k,s=  get_kernel_strides(patch_size,spacings)
# %%
    kernels, strides = get_kernel_strides(
        patch_size,[1,1,2]
    )
# %%
    net= DynUNet(
        3,
        1,
        3,
        kernel_size=k,
        strides=s,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=3,
    )
    net2= DynUNet_UB(
        3,
        1,
        3,
        kernel_size=k,
        strides=s,
        upsample_kernel_size=s[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=3,
    )
# %%
    x = torch.randn(2, 1, *patch_size, device="cpu")
    y= net(x)
    y2= net2(x)
    print(y.shape)
    [print(a.shape) for a in y2]

    yy = torch.unbind(y,1)
# %%
    summ = summary(net, input_size=tuple([1,1]+patch_size),col_names=["input_size","output_size","kernel_size"],depth=4, verbose=0,device='cuda')
    summ2 = summary(net2, input_size=tuple([1,1]+patch_size),col_names=["input_size","output_size","kernel_size"],depth=4, verbose=0,device='cuda')
    print(summ2)
# %%
