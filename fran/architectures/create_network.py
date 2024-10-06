# %%
import lightning.pytorch as pl
from lightning.pytorch.core import LightningModule
import torch

# from fran.architectures.unet3d.model import UNet3D
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin
from nnunet.network_architecture.initialization import InitWeights_He
from monai.networks.nets import SwinUNETR
from fran.architectures.dynunet import DynUNet_UB, get_kernel_strides, DynUNet
from torch import nn
import torch.nn.functional as F
import ipdb
from fran.architectures.unet3d.model import UNet3D

tr = ipdb.set_trace


from fran.extra.deepcore.deepcore.nets.nets_utils.recorder import EmbeddingRecorder
from fran.utils.config_parsers import make_patch_size


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


class Generic_UNet_PL(Generic_UNet, LightningModule):
    def forward(self, x):
        skips = []
        x_l = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)

            z_l = self.seg_outputs[u](x)
            x_l.append(self.final_nonlin(z_l))
        if self._deep_supervision and self.do_ds:
            return tuple(
                [x_l[-1]]
                + [
                    i(j)
                    for i, j in zip(list(self.upscale_logits_ops)[::-1], x_l[:-1][::-1])
                ]
            )
        else:
            return x_l[-1]


class nnUNetCraig(Generic_UNet_PL):
    def __init__(
        self,
        input_channels,
        base_num_features,
        num_classes,
        num_pool,
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=nn.Conv2d,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        deep_supervision=True,
        dropout_in_localization=False,
        final_nonlin=...,
        weightInitializer=...,
        pool_op_kernel_sizes=None,
        conv_kernel_sizes=None,
        upscale_logits=False,
        convolutional_pooling=False,
        convolutional_upsampling=False,
        max_num_features=None,
        basic_block=...,
        seg_output_use_bias=False,
        capture_grads=True,
        record_embedding=True,
    ):
        super().__init__(
            input_channels,
            base_num_features,
            num_classes,
            num_pool,
            num_conv_per_stage,
            feat_map_mul_on_downscale,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            deep_supervision,
            dropout_in_localization,
            final_nonlin,
            weightInitializer,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            upscale_logits,
            convolutional_pooling,
            convolutional_upsampling,
            max_num_features,
            basic_block,
            seg_output_use_bias,
        )
        # self.grad_L_x = None
        self.capture_grads = capture_grads
        self.final_final_nonlin = final_nonlin
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        if self.capture_grads == True:
            final_conv = self.seg_outputs[-1]  # this is convblock
            final_conv.register_full_backward_hook(self.capture_grad_L_x)
            # self.final_final_nonlin.register_full_backward_hook(self.capture_grad_sigma)

    def get_last_layer(self):
        return self.tu[0]

    def capture_grad_L_x(self, module, grad_input, grad_output):
        self.grad_L_x = grad_output[0]

    def compute_grad_sigma_z(self, sigma, z_l):
        x_l = sigma(z_l)
        x_l.retain_grad()
        dummy_l = x_l.sum()
        dummy_l.backward(retain_graph=True)
        self.grad_sigma_z = x_l.grad


    def contract(self, level:int,x:torch.Tensor,skip:torch.Tensor,capture_grad=False):
            x = self.tu[level](x)
            # skip = skips[-(u + 1)]
            x = torch.cat((x, skip), dim=1)
            x = self.conv_blocks_localization[level](x)

            # x = self.tu[level](x)
            # x = torch.cat((x, skip), dim=1)
            # x = self.conv_blocks_localization[level](x)

            return x

    def forward(self, x):
        skips = []
        x_ls = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)
        x = self.embedding_recorder(x)

        for uplevel in range(n_levels:=(len(self.tu))):
            if self.capture_grads==True and uplevel==(n_levels-1):
                capture_grad=True
            else:
                capture_grad=False
            skip=skips[-(uplevel+1)]
            x= self.contract(uplevel,x,skip,capture_grad)
            z_l = self.seg_outputs[uplevel](x)
            x_l = self.final_nonlin(z_l)
            x_ls.append(x_l)
        if self._deep_supervision and self.do_ds:
            return tuple(
                [x_ls[-1]]
                + [
                    i(j)
                    for i, j in zip(list(self.upscale_logits_ops)[::-1], x_ls[:-1][::-1])
                ]
            )
        else:
            return x_ls[-1]


def create_model_from_conf(
    model_params, dataset_params, metadata=None, deep_supervision=True
):
    # if 'out_channels' not in model_params:
    #         model_params["out_channels"] =  out_channels_from_dict_or_cell(model_params['src_dest_labels'])

    if "patch_size" not in dataset_params.keys():
        dataset_params["patch_size"] = make_patch_size(
            dataset_params["patch_dim0"], dataset_params["patch_dim1"]
        )
    arch = model_params["arch"]
    if arch == "UNet3D":
        model = create_model_from_conf_unet(model_params, dataset_params)
    elif arch == "nnUNet":
        model = create_model_from_conf_nnUNet(
            model_params,  deep_supervision
        )

    elif arch== "nnUNetCraig":
        model = create_model_from_conf_nnUNetCraig(
            model_params,  deep_supervision)
    elif arch == "SwinUNETR":
        model = create_model_from_conf_swinunetr(
            model_params, dataset_params, deep_supervision
        )
    elif arch == "DynUNet":
        model = create_model_from_conf_dynunet(model_params, dataset_params)

    elif arch == "DynUNet_UB":
        model = create_model_from_conf_dynunet_ub(model_params, dataset_params)
    else:
        raise NotImplementedError

    if model_params["compiled"] == True:
        model = torch.compile(model)
    return model


def create_model_from_conf_dynunet(model_params, dataset_params):

    kernels, strides = get_kernel_strides(
        dataset_params["patch_size"], dataset_params["spacing"]
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


def create_model_from_conf_dynunet_ub(model_params, dataset_params):

    kernels, strides = get_kernel_strides(
        dataset_params["patch_size"], dataset_params["spacing"]
    )
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


def create_model_from_conf_nnUNet(model_params,  deep_supervision):
    # pool_op_kernel_sizes = pool_op_kernels_nnunet(dataset_params['patch_size'])
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

def create_model_from_conf_nnUNetCraig(model_params,  deep_supervision):
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
        record_embedding=True
    )
    return model


def create_model_from_conf_swinunetr(
    model_params, dataset_params, deep_supervision=None
):
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


class nnUNet(pl.LightningModule):
    def __init__(
        self, in_channels, out_channels, deep_supervision, pool_op_kernel_sizes=None
    ):
        super().__init__()
        self.model = Generic_UNet(
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


# %%
if __name__ == "__main__":
    import torch
    from torchinfo import summary

    patch_size = [192, 192, 96]
    x = torch.rand(1, 1, 192, 192, 96)
    model_params = {"in_channels": 1, "out_channels": 3}
    dataset_params = {"patch_size": patch_size}
    deep_supervision = True
    pool_op_kernel_sizes = [[2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
    pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
    # net = create_model_from_conf_nnUNet(model_params,dataset_params,deep_supervision)
    net2 = create_model_from_conf_nnUNetCraig(model_params,  deep_supervision)
    net2.to("cuda")
# %%
    x = torch.rand(1, 1, 128, 128, 96)
    x = x.to("cuda")
    print(x.shape)
    y= net2(x)
# %%
#SECTION:-------------------- DOwnsampling--------------------------------------------------------------------------------------

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
    for u in range(len(net2.tu)-1):
        x = net2.tu[u](x)
        if u == 1:
            tr()
        print(x.shape)
        skip = skips[-(u + 1)]
        print(skip.shape)
        x = torch.cat((x, skip), dim=1)
        print(x.shape)
        x = net2.conv_blocks_localization[u](x)
        print(x.shape)

        print("--" * 20)
        z = net2.final_nonlin(net2.seg_outputs[u](x))
        print(z.shape)
        print("==" * 20)
        seg_outputs.append(z)

# %%
    u=-1
    x = net2.tu[u](x)
    print(x.shape)
    skip = skips[-(u + 1)]
    print(skip.shape)
    x = torch.cat((x, skip), dim=1)
    print(x.shape)
    x = net2.conv_blocks_localization[u](x)
    print(x.shape)

    print("--" * 20)
    x_out = net2.final_nonlin(net2.seg_outputs[u](x))
    grad_sigma_z = torch.autograd.grad(x_out,x,grad_outputs = torch.ones_like(x_out))
    print(x_out.shape)
    print("==" * 20)
    seg_outputs.append(x_out)


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
# SECTION:-------------------- Model parts-------------------------------------------------------------------------------------- <CR> <CR>
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
