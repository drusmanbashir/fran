import torch

from torch import nn

from fran.architectures.nnunet import Generic_UNet_PL
from fran.extra.deepcore.deepcore.nets.nets_utils.recorder import EmbeddingRecorder

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
