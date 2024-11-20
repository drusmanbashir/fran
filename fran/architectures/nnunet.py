from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet, InitWeights_He
import lightning.pytorch as pl

from torch import nn
from lightning.pytorch.core import LightningModule
class Generic_UNet_PL(Generic_UNet,LightningModule):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None, dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False, final_nonlin=..., weightInitializer=..., pool_op_kernel_sizes=None, conv_kernel_sizes=None, upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False, max_num_features=None, basic_block=..., seg_output_use_bias=False):
        super(LightningModule,self).__init__()
        super().__init__(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage, feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization, final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes, upscale_logits, convolutional_pooling, convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)



class nnUNet(pl.LightningModule):
    def __init__(self,in_channels,out_channels,deep_supervision,pool_op_kernel_sizes=None):
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
