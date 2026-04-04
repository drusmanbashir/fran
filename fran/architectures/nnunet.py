from typing import Type, Union

import lightning.pytorch as pl
from dynamic_network_architectures.architectures.unet import (
    PlainConvUNet,
    ResidualEncoderUNet,
)
from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)
from dynamic_network_architectures.building_blocks.residual import (
    BasicBlockD,
    BottleneckD,
)
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunetv2.experiment_planning.experiment_planners.network_topology import (
    get_pool_and_conv_props,
)
from torch import nn

_PLAIN_BLOCKS_PER_STAGE_ENCODER = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
_PLAIN_BLOCKS_PER_STAGE_DECODER = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)

_RESENC_L_BLOCKS_PER_STAGE_ENCODER = (1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
_RESENC_L_BLOCKS_PER_STAGE_DECODER = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)


class PlainConvUNetPL(PlainConvUNet, pl.LightningModule):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage,
        conv_op,
        kernel_sizes,
        strides,
        n_conv_per_stage,
        num_classes: int,
        n_conv_per_stage_decoder,
        conv_bias: bool = False,
        norm_op=None,
        norm_op_kwargs: dict | None = None,
        dropout_op=None,
        dropout_op_kwargs: dict | None = None,
        nonlin=None,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
    ):
        super(pl.LightningModule, self).__init__()
        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            nonlin_first=nonlin_first,
        )

    @property
    def seg_outputs(self):
        return self.decoder.seg_layers


class Generic_UNet_PL(Generic_UNet, pl.LightningModule):
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
    ):
        super(pl.LightningModule, self).__init__()
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


class ResidualEncoderUNetPL(ResidualEncoderUNet, pl.LightningModule):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage,
        conv_op,
        kernel_sizes,
        strides,
        n_blocks_per_stage,
        num_classes: int,
        n_conv_per_stage_decoder,
        conv_bias: bool = False,
        norm_op=None,
        norm_op_kwargs: dict | None = None,
        dropout_op=None,
        dropout_op_kwargs: dict | None = None,
        nonlin=None,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels=None,
        stem_channels=None,
    ):
        super(pl.LightningModule, self).__init__()
        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            block=block,
            bottleneck_channels=bottleneck_channels,
            stem_channels=stem_channels,
        )

    @property
    def seg_outputs(self):
        return self.decoder.seg_layers


def _compute_topology(plan: dict):
    spacing = tuple(float(x) for x in plan["spacing"])
    patch_size = tuple(int(x) for x in plan["patch_size"])
    dim = len(spacing)
    if dim not in (2, 3):
        raise ValueError(f"Unsupported dimensionality: {dim}")

    conv_op = convert_dim_to_conv_op(dim)
    norm_op = get_matching_instancenorm(conv_op)
    max_num_features = 512 if dim == 2 else 320

    _, strides, kernel_sizes, patch_size, _ = get_pool_and_conv_props(
        spacing,
        patch_size,
        4,
        999999,
    )
    num_stages = len(strides)
    features_per_stage = tuple(
        min(max_num_features, 32 * 2**i) for i in range(num_stages)
    )
    return (
        conv_op,
        norm_op,
        patch_size,
        num_stages,
        features_per_stage,
        strides,
        kernel_sizes,
    )


def create_plainconvunet_pl(
    model_params: dict, plan: dict, deep_supervision: bool = True
):
    (
        conv_op,
        norm_op,
        patch_size,
        num_stages,
        features_per_stage,
        strides,
        kernel_sizes,
    ) = _compute_topology(plan)

    model = PlainConvUNetPL(
        input_channels=model_params["in_channels"],
        n_stages=num_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=_PLAIN_BLOCKS_PER_STAGE_ENCODER[:num_stages],
        num_classes=model_params["out_channels"],
        n_conv_per_stage_decoder=_PLAIN_BLOCKS_PER_STAGE_DECODER[: num_stages - 1],
        conv_bias=True,
        norm_op=norm_op,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
        nonlin_first=False,
    )
    model.ds_strides = strides
    model.arch_kwargs = {
        "n_stages": num_stages,
        "features_per_stage": features_per_stage,
        "kernel_sizes": kernel_sizes,
        "strides": strides,
    }
    plan["patch_size"] = list(patch_size)
    return model


def create_resencunet_l_pl(
    model_params: dict, plan: dict, deep_supervision: bool = True
):
    (
        conv_op,
        norm_op,
        patch_size,
        num_stages,
        features_per_stage,
        strides,
        kernel_sizes,
    ) = _compute_topology(plan)

    model = ResidualEncoderUNetPL(
        input_channels=model_params["in_channels"],
        n_stages=num_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_blocks_per_stage=_RESENC_L_BLOCKS_PER_STAGE_ENCODER[:num_stages],
        num_classes=model_params["out_channels"],
        n_conv_per_stage_decoder=_RESENC_L_BLOCKS_PER_STAGE_DECODER[: num_stages - 1],
        conv_bias=True,
        norm_op=norm_op,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
    model.ds_strides = strides
    model.arch_kwargs = {
        "n_stages": num_stages,
        "features_per_stage": features_per_stage,
        "kernel_sizes": kernel_sizes,
        "strides": strides,
    }
    plan["patch_size"] = list(patch_size)
    return model


if __name__ == "__main__":
    import torch
    from dynamic_network_architectures.architectures.abstract_arch import (
        test_submodules_loadable,
    )

    network = ResidualEncoderUNetPL(
        input_channels=32,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[[3, 3, 3] for _ in range(6)],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=2,
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )
    network.initialize(network)
    test_submodules_loadable(network)

# %%
