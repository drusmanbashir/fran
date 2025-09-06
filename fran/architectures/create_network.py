# %%
import torch
from fran.architectures.nnunet import Generic_UNet_PL

# from fran.architectures.unet3d.model import UNet3D
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin
from nnunet.network_architecture.initialization import InitWeights_He
from monai.networks.nets import SwinUNETR
from fran.architectures.dynunet import DynUNet_UB, get_kernel_strides, DynUNet
from torch import nn
import torch.nn.functional as F
import ipdb
from fran.architectures.unet3d.model import UNet3D

tr = ipdb.set_trace


from fran.architectures.unetcraig import nnUNetCraig
from fran.utils.config_parsers import ConfigMaker, make_patch_size


def get_batch_size(
    model: nn.Module,
    device: torch.device,
    input_shape,
    output_shape,
    dataset_size: int,
    max_batch_size: int = None,
    num_iterations: int = 5,
) -> int:
    model.to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 2
    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = batch_size // 2
            break
        try:
            for _ in range(num_iterations):
                # dummy inputs and targets
                inputs = torch.rand(*(batch_size, *input_shape), device=device)
                targets = torch.rand(*(batch_size, *output_shape), device=device)
                outputs = model(inputs)
                loss = F.mse_loss(targets, outputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            batch_size *= 2
        except RuntimeError:
            batch_size //= 2
            break
    del model, optimizer
    torch.cuda.empty_cache()
    return batch_size


def create_model_from_conf_nnUNetCraig(model_params, deep_supervision):
    pool_op_kernel_sizes = None
    in_channels, out_channels = (
        model_params["in_channels"],
        model_params["out_channels"],
    )

    model = nnUNetCraig(
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
        pool_op_kernel_sizes=(
            [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
            if pool_op_kernel_sizes is None
            else pool_op_kernel_sizes
        ),
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
        record_embedding=True,
    )
    return model


def create_model_from_conf(model_params, plan, deep_supervision=True):
    # if 'out_channels' not in model_params:
    #         model_params["out_channels"] =  out_channels_from_dict_or_cell(model_params['remapping_train'])

    if "patch_size" not in plan.keys():
        plan["patch_size"] = make_patch_size(plan["patch_dim0"], plan["patch_dim1"])
    arch = model_params["arch"]
    if arch == "UNet3D":
        model = create_model_from_conf_unet(model_params, plan)
    elif arch == "nnUNet":
        model = create_model_from_conf_nnUNet(model_params, plan, deep_supervision)
    elif arch == "SwinUNETR":
        model = create_model_from_conf_swinunetr(model_params, plan, deep_supervision)
    elif arch == "DynUNet":
        model = create_model_from_conf_dynunet(model_params, plan)

    elif arch == "DynUNet_UB":
        model = create_model_from_conf_dynunet_ub(model_params, plan)
    else:
        raise NotImplementedError

    if model_params["compiled"] == True:
        model = torch.compile(model, dynamic=True)
    return model


def create_model_from_conf_dynunet(model_params, plan):

    kernels, strides = get_kernel_strides(plan["patch_size"], plan["spacing"])
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


def create_model_from_conf_dynunet_ub(model_params, plan):

    kernels, strides = get_kernel_strides(plan["patch_size"], plan["spacing"])
    model = DynUNet_UB(
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
    _, pool_op_kernel_sizes = get_kernel_strides(patch_size, [1, 1, 1])
    pool_op_kernel_sizes = pool_op_kernel_sizes[1:]
    # pool_op_kernel_sizes.reverse() # try witho reverse and without both
    return pool_op_kernel_sizes


def create_model_from_conf_nnUNet(model_params, plan, deep_supervision):
    # pool_op_kernel_sizes = pool_op_kernels_nnunet(plan['patch_size'])
    pool_op_kernel_sizes = None
    in_channels, out_channels = (
        model_params["in_channels"],
        model_params["out_channels"],
    )

    model = Generic_UNet_PL(
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
        pool_op_kernel_sizes=(
            [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
            if pool_op_kernel_sizes is None
            else pool_op_kernel_sizes
        ),
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


def create_model_from_conf_swinunetr(model_params, plan, deep_supervision=None):
    model = SwinUNETR(
        in_channels=model_params["in_channels"],
        out_channels=model_params["out_channels"],
        # patch_size = plan["patch_size"],
        
    )
    return model


def create_model_from_conf_unet(model_params, plan):
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
    from fran.utils.common import *
    from fran.managers.project import Project

    P = Project(project_title="nodes")
    C = ConfigMaker(P, raytune=False, configuration_filename=None)
    config = C.config
# %%
    patch_size = [192, 192, 96]
    x = torch.rand(1, 1, 192, 192, 96)
    model_params = {"in_channels": 1, "out_channels": 3}
    dataset_params = {"patch_size": patch_size}
    deep_supervision = True
    plan = config["plan_train"]
    net = create_model_from_conf_swinunetr(model_params, plan, deep_supervision)
    img = torch.rand(1, 1, 128, 128, 96)
    pred = net(img)
# %%
    pool_op_kernel_sizes = [[2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
    pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
    # net = create_model_from_conf_nnUNet(model_params,dataset_params,deep_supervision)
    model_params = config["model_params"]
    dataset_params=config["dataset_params"]
    deep_supervision = True
    net2 = create_model_from_conf_nnUNet(model_params, dataset_params, deep_supervision)
    x = x.to("cuda")
    net2.to("cuda")
    out = net2(x)
# %%
    x = torch.rand(1, 1, 128, 128, 96)
    x = x.to("cuda")
    print(x.shape)

    skips = []
    seg_outputs = []
    for d in range(len(net2.conv_blocks_context) - 1):
        x = net2.conv_blocks_context[d](x)
        skips.append(x)
        print(x.shape)
        if not net2.convolutional_pooling:
            x = net2.td[d](x)
            print(x.shape)
        print("--" * 20)
# %%

    x = net2.conv_blocks_context[-1](x)
    print(x.shape)

    net2.conv_blocks_context.parameters()
# %%
    for u in range(len(net2.tu)):
        x = net2.tu[u](x)
        print(x.shape)
        y = skips[-(u + 1)]
        print(y.shape)
        x = torch.cat((x, y), dim=1)
        print(x.shape)
        x = net2.conv_blocks_localization[u](x)
        print(x.shape)

        print("--" * 20)
        z = net2.final_nonlin(net2.seg_outputs[u](x))
        print(z.shape)
        print("==" * 20)
        seg_outputs.append(z)

# %%
    outs = tuple(
        [seg_outputs[-1]]
        + [
            i(j)
            for i, j in zip(list(net2.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])
        ]
    )
# %%
    for out in outs:
        print(out.shape)
        print("**" * 20)
# %%
# %%
# SECTION:-------------------- Model parts-------------------------------------------------------------------------------------- <CR>
    # NOTE: TD

    cc = net2.td.children()
    list(cc)

    # summ = summary(net, input_size=tuple([1,1]+patch_size),col_names=["input_size","output_size","kernel_size"],depth=4, verbose=0,device='cuda')
# %%
    summ2 = summary(
        net2,
        input_size=tuple([1, 1] + patch_size),
        col_names=["input_size", "output_size", "kernel_size"],
        depth=4,
        verbose=0,
        device="cuda",
    )
# %%
    print(summ2)
