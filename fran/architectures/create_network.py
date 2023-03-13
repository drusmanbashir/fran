
# %%
from fran.architectures.unet3d.model import ResidualUNet3D, UNet3D
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin
from nnunet.network_architecture.initialization import InitWeights_He
from monai.networks.nets import SwinUNETR
from fran.architectures.dynunet import get_kernel_strides, DynUNet
from torch import nn
import ipdb
tr = ipdb.set_trace


from fran.utils.config_parsers import make_patch_size


def create_model_from_conf(model_params, dataset_params,metadata=None,deep_supervision=True):
    # if 'out_channels' not in model_params:
    #         model_params["out_channels"] =  out_channels_from_dict_or_cell(model_params['src_dest_labels'])

    if 'patch_size' not in dataset_params.keys():
        dataset_params['patch_size'] = make_patch_size(dataset_params['patch_dim0'],dataset_params['patch_dim1'])
    arch = model_params["arch"]
    if arch == "UNet3D":
        model = create_model_from_conf_unet(model_params, dataset_params)
    elif arch == "nnUNet":
        model = create_model_from_conf_nnUNet(model_params, dataset_params,deep_supervision)
    elif arch == "SwinUNETR":
        model = create_model_from_conf_swinunetr(model_params, dataset_params,deep_supervision)
    elif arch == "DynUNet":
        model = create_model_from_conf_dynunet(model_params, dataset_params)
    else:
        raise NotImplementedError
    return model


def create_model_from_conf_dynunet(model_params, dataset_params):

    kernels, strides = get_kernel_strides(
        dataset_params["patch_size"], dataset_params["spacings"]
    )
    model = DynUNet(
        3,
        model_params["in_channels"],
        model_params["out_channels"],
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=bool(model_params["deep_supervision"]),
        deep_supr_num=3,
    )
    return model

def pool_op_kernels_nnunet(patch_size):
    _ , pool_op_kernel_sizes = get_kernel_strides(patch_size,[1,1,1])
    pool_op_kernel_sizes = pool_op_kernel_sizes[1:]
    # pool_op_kernel_sizes.reverse() # try witho reverse and without both
    return pool_op_kernel_sizes
def create_model_from_conf_nnUNet(model_params, dataset_params,deep_supervision):
    # pool_op_kernel_sizes = pool_op_kernels_nnunet(dataset_params['patch_size'])
    pool_op_kernel_sizes=None
    in_channels, out_channels = (
        model_params["in_channels"],
        model_params["out_channels"],
    )

    model = Generic_UNet(
        in_channels,
        base_num_features=32,
        num_classes=out_channels,
        num_pool=5,
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=nn.Conv3d,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 0.01, "inplace": True},
        deep_supervision=deep_supervision,
        dropout_in_localization=False,
        final_nonlin=lambda x: x,
        weightInitializer=InitWeights_He(1e-2),
        pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]] if pool_op_kernel_sizes is None else pool_op_kernel_sizes,
        conv_kernel_sizes=[
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        upscale_logits=False,
        convolutional_pooling=True,
        convolutional_upsampling=True,
        max_num_features=None,
        basic_block=ConvDropoutNormNonlin,
        seg_output_use_bias=False,
    )
    return model


def create_model_from_conf_swinunetr(model_params, dataset_params,deep_supervision=None):
    model = SwinUNETR(
        dataset_params["patch_size"],
        model_params["in_channels"],
        model_params["out_channels"],
    )
    return model


def create_model_from_conf_unet(model_params, dataset_params):
    model = UNet3D(
        in_channels=model_params["in_channels"],
        out_channels=model_params["out_channels"],
        final_sigmoid=False,
        f_maps=model_params["base_ch_opts"],
        layer_order=model_params["layer_order"],
        num_levels=model_params["depth_opts"],
        n_bottlenecks=model_params["n_bottlenecks"],
        heavy=model_params["heavy"],
        deep_supervision=model_params["deep_supervision"],
        self_attention=model_params["self_attention"],
    )
    return model

# %%
if __name__ == "__main__":
    import torch
    from torchinfo import summary
    patch_size = [192,192,96]
    x = torch.rand(1,1,192,192,96)
    model_params = {'in_channels':1, 'out_channels':3}
    dataset_params = {'patch_size':patch_size}
    deep_supervision=True
    pool_op_kernel_sizes = [[2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]] 
    pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
    # net = create_model_from_conf_nnUNet(model_params,dataset_params,deep_supervision)
    net2 = create_model_from_conf_nnUNet(model_params,dataset_params,deep_supervision)
    # out = net(x)
    # summ = summary(net, input_size=tuple([1,1]+patch_size),col_names=["input_size","output_size","kernel_size"],depth=4, verbose=0,device='cuda')
    summ2 = summary(net2, input_size=tuple([1,1]+patch_size),col_names=["input_size","output_size","kernel_size"],depth=4, verbose=0,device='cuda')
# %%
    print(summ2)
