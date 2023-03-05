# %%
import os,sys
from fastai.data.block import ConvLayer

from fastcore.meta import delegates
from fastcore.script import store_attr

sys.path += ['/home/ub/Dropbox/code/fran']

from functools import reduce
import ipdb
import torch
import torch.nn.functional as F
tr = ipdb.set_trace
import torch.nn as nn
import ipdb
from torch.nn.modules.pooling import MaxPool3d
tr = ipdb.set_trace

from fran.architectures.unet3d.buildingblocks import DoubleConv, ExtResNetBlock, SingleConv, create_attention_gates, create_deep_segmenters, create_encoders, \
    create_decoders, create_bottlenecks
from fran.architectures.unet3d.utils import number_of_features_per_level, get_class
def fuse_deep_segmentations(seg_a,seg_b):
            seg_a = F.interpolate(seg_a,size=seg_b.shape[2:],mode='trilinear')
            return seg_a + seg_b
 

class Abstract3DUNet(nn.Module):
    """
    NO ACTIVATION IS APPLIED BY MY ADAPTATION
    Base class for standard and residual UNet.

    Args: in_channels (int): number of input channels out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2, pool_stride=1,
                 conv_padding=1, n_bottlenecks = 1, heavy=False,deep_supervision=True,self_attention=True, **kwargs):
        super(Abstract3DUNet, self).__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size,pool_stride, heavy=heavy)

        self.bottlenecks = create_bottlenecks(f_maps[-1],basic_module=basic_module,layer_order=layer_order, n_bottlenecks=n_bottlenecks)
        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        upsample=True,**kwargs)
        convs=[]
        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        if self_attention==True:
            self.attention_gates = create_attention_gates(f_maps)
        else:
            self.attention_gates = [None,]*(len(f_maps)-1)
        if deep_supervision==True:
            self.deep_segs=create_deep_segmenters(f_maps,out_channels)
        else:
            self.deep_segs =[None,]*(len(f_maps)-1)
        # if is_segmentation:
        #     # semantic segmentation problem
        #     if final_sigmoid:
        #         self.final_activation = nn.Sigmoid()
        #     else:
        #         self.final_activation = nn.Softmax(dim=1)
        # else:
        #     # regression problem
        #     self.final_activation = None
        #
    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        x = self.bottlenecks(x)

        # decoder part
        seg_outputs=[]
        for decoder,attention_gate, encoder_features, deep_seg in zip(self.decoders, self.attention_gates, encoders_features,self.deep_segs):
            if attention_gate is not None:
                encoder_features, x = attention_gate(encoder_features,x)
            x = decoder(encoder_features, x)
            if deep_seg is not None:
                seg = deep_seg(x)
                # seg = self.final_activation(seg) 
                seg_outputs.append(seg) 
        x = self.final_conv(x)

        seg_outputs.append(x)
        x = reduce(fuse_deep_segmentations,seg_outputs)
        # if self.final_activation is not None:
        #     x = self.final_activation(x)
        # return seg_outputs[:-1]+[x]
        return [x]


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and trilinear upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)



class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             **kwargs)



class Generator(nn.Module):
    @delegates(UNet3D)
    def __init__(self, in_channels=2,out_channels=1,final_sigmoid=False,num_levels=6,deep_supervision=False,self_attention=False,pool_stride=2, **kwargs) -> None:
        super().__init__()
        self.model=UNet3D(in_channels=in_channels,out_channels=out_channels,final_sigmoid=final_sigmoid,num_levels=num_levels,deep_supervision=deep_supervision,
                          self_attention=self_attention,pool_stride=pool_stride,**kwargs)
    def forward(self,x):
        mask=x[:,:-1,:].clone()
        x = self.model(x)[0]
        x = torch.tanh(x)
        return torch.cat([mask,x],1)

class Discriminator_ub(nn.Module):
        def __init__(self,in_channels, num_levels,f_maps,patch_size,scalar_output=False):
            super().__init__()
            store_attr('in_channels')
            self.dummy= torch.rand(1,self.in_channels,*patch_size)
            strides = [2]*(num_levels)  +[1]
            orders =['sl']+['sil']*(num_levels)
            if isinstance(f_maps, int):
                f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
            f_maps = [in_channels]+f_maps+[1]
            assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
            assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
            # layers= [nn.Conv3d(f_maps[i],f_maps[i+1],3,strides[i])for i in range(num_levels)]
            layers = [SingleConv(f_maps[i],f_maps[i+1],kernel_size=3,stride=strides[i],order=orders[i]) for i in range(len(f_maps)-1)]
            final_linear=self.compute_linear_layer(in_channels,patch_size,layers) 
            activation = nn.Sigmoid()
            if scalar_output==True:
                self.model = nn.Sequential(*layers,nn.Flatten(),final_linear,activation)
            else:
                self.model = nn.Sequential(*layers,activation)
            self.output_shape=patch_size


        def compute_linear_layer(self,in_channels, patch_size,layers):
            with torch.no_grad():
                layers =nn.Sequential(*layers)
                tra = layers(self.dummy)
            final_in_ch = reduce(lambda a,b:a*b,tra.shape[1:])
            in_ch_linear= tra.view(-1,1,final_in_ch).shape[-1]
            return nn.Linear(in_ch_linear,1)
        def forward(self,x):
            # x = torch.cat(x,1)       
            return self.model(x)

        @property
        def output_shape(self):
            return self._output_shape

        @output_shape.setter
        def output_shape(self,patch_size):
            with torch.no_grad():
                tmp_out = self.model(self.dummy)
                self._output_shape=tmp_out.shape[1:]

def get_model(model_config):
    model_class = get_class(model_config['name'], modules=['architectures.unet3d.model'])
    return model_class(**model_config)
# %%
if __name__ == "__main__":
        import torch
        # model=UNet3D(in_channels=1,out_channels=2,final_sigmoid=False,f_maps = 32,num_levels=4,layer_order='clb')
        f_maps = 32
        levels= 5
        model=UNet3D(in_channels=1,out_channels=2,final_sigmoid=False,f_maps = f_maps,num_levels=levels,layer_order='clb',heavy=True,deep_supervision=True,n_bottlenecks=10, pool_stride=1)
        xx = torch.rand(1,1,48,128,128)
    
        
        # pred = model(xx)
    

# %%
        c= ConvLayer(1,32,ndim=3)
        
# %%
# %%
        x = xx.clone()
        encoders_features = []
        for encoder in model.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        [a.shape for a in encoders_features]
        # remove the last encoder's output from the list

# %%
# %%
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        [a.shape for a in encoders_features]
# %%
        encoder_features = encoders_features[0]
        d= model.decoders[0]
# %%
        print(x.shape)
        print(encoder_features.shape)
        x_d0 = d(encoder_features,x)

# %%
        xxx = d.upsampling(encoder_features=encoder_features, x=x)
        xxxx = d.joining(encoder_features, xxx)
        x5 = d.basic_module(xxxx)
        [a.shape for a in [x,xxx,xxxx,x5]]

# %%
         
        g = x.clone()
        W_g = nn.Conv3d(in_channels=g.shape[1],out_channels=F_int,kernel_size=1)
        W_x = nn.Conv3d(in_channels=encoder_features.shape[1],out_channels=F_int,kernel_size=1,stride=2)
        psi = nn.Conv3d(in_channels=F_int,out_channels=1,kernel_size=1)
        relu = nn.LeakyReLU()
        activation = nn.Sigmoid()

        e = W_x(encoder_features)
        upsize = encoder_features.shape[2:]
        g = W_g(g)
        eg = relu(e+g)
        eg2 = psi(eg)
        ac = activation(eg2)
        alpha = F.interpolate(ac,size=upsize)


        encoder_features_gated= alpha*encoder_features

        # decoder part
# %%
        x_d =  x.clone()
        x_dall=[]
        for decoder, encoder_features in zip(model.decoders, encoders_features):
        
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x_d = decoder(encoder_features,x_d)
            x_dall.append(x_d)
        [a.shape for a in x_dall]# %%

# %%
        in_channels=1
        out_channels=2
        f_maps = 32
        final_sigmoid=False
        num_levels=5
        layer_order='clb'
        heavy=True
        deep_supervision=True
        basic_module=DoubleConv
        conv_kernel_size=3
        num_groups=8
        is_segmentation=True
        conv_padding=1
        pool_kernel_size=2

# %%
# %%

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

# %%
        # create encoder path
        model.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size,heavy=heavy)

# %%
        n_bottlenecks=0
        in_out_channels = 32
        b=    create_bottlenecks(f_maps[-1],basic_module,layer_order,n_bottlenecks=n_bottlenecks)
# %%
        # create decoder path
        model.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        upsample=True)


# %%
        x = xx.clone()
        # encoder part
        encoders_features = []
        for encoder in model.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        print(len(encoders_features))
        # decoder part

        x = model.bottlenecks(x)

        xxx = model.decoders[0](encoder_features[0], x)
# %%
# %% [markdown]
## Making Generator

# %%
        f_maps = 64
        num_levels = 4
        if isinstance(f_maps, int):
                f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
# %%
# %%
