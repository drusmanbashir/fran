# %%
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import ResEncUNetPlanner, nnUNetPlannerResEncL,  nnUNetPlannerResEncXL, nnUNetPlannerResEncM
import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
from torch import nn

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props

# %%

if __name__ == '__main__':
# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
        # we know both of these networks run with batch size 2 and 12 on ~8-10GB, respectively
    net = ResidualEncoderUNet(input_channels=1, n_stages=6, features_per_stage=(32, 64, 128, 256, 320, 320),
                              conv_op=nn.Conv3d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2),
                              n_blocks_per_stage=(1, 3, 4, 6, 6, 6), num_classes=3,
                              n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
                              conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={}, dropout_op=None,
                              nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=True)
# %%
    print(net.compute_conv_feature_map_size((128, 128, 128)))  # -> 558319104. The value you see above was finetuned
    # from this one to match the regular nnunetplans more closely

# %%
    net = ResidualEncoderUNet(input_channels=1, n_stages=7, features_per_stage=(32, 64, 128, 256, 512, 512, 512),
                              conv_op=nn.Conv2d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2, 2),
                              n_blocks_per_stage=(1, 3, 4, 6, 6, 6, 6), num_classes=3,
                              n_conv_per_stage_decoder=(1, 1, 1, 1, 1, 1),
                              conv_bias=True, norm_op=nn.InstanceNorm2d, norm_op_kwargs={}, dropout_op=None,
                              nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=True)
    print(net.compute_conv_feature_map_size((512, 512)))  # -> 129793792
# %%
# %%
