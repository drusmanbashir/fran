# %%


# %%

if __name__ == "__main__":
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    from torch import nn

    # %%
    # SECTION:-------------------- setup--------------------------------------------------------------------------------------
    # we know both of these networks run with batch size 2 and 12 on ~8-10GB, respectively
    net = ResidualEncoderUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=(1, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        num_classes=3,
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
    )
    # %%
    print(
        net.compute_conv_feature_map_size((128, 128, 128))
    )  # -> 558319104. The value you see above was finetuned
    # from this one to match the regular nnunetplans more closely

    # %%
    net = ResidualEncoderUNet(
        input_channels=1,
        n_stages=7,
        features_per_stage=(32, 64, 128, 256, 512, 512, 512),
        conv_op=nn.Conv2d,
        kernel_sizes=3,
        strides=(1, 2, 2, 2, 2, 2, 2),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6, 6),
        num_classes=3,
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1, 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
    )
    print(net.compute_conv_feature_map_size((512, 512)))  # -> 129793792
# %%
# %%
