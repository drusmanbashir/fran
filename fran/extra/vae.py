from fastai.test_utils import show_install
show_install()
from fastai.vision.all import *
path = untar_data(URLs.MNIST_SAMPLE)

btfms = aug_transforms()+[Normalize.from_stats(*mnist_stats)]
btfms

block = DataBlock(blocks=(ImageBlock(cls=PILImageBW), ImageBlock(cls=PILImageBW)),
                  get_items = get_image_files,
                  get_y = lambda o: o,
                  splitter=GrandparentSplitter(),
                  item_tfms=Resize(32),
                  batch_tfms = btfms,
)

block.summary(path)

dls = block.dataloaders(path, batch_size=256)
dls.show_batch()
arch = create_body(xresnet18, n_in=1).cuda()

x,y = dls.one_batch()
x.shape, y.shape

arch(x).shape

class UpsampleBlock(Module):
    def __init__(self, up_in_c:int, final_div:bool=True, blur:bool=False, leaky:float=None, **kwargs):
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, **kwargs)
        ni = up_in_c//2
        nf = ni if final_div else ni//2
        self.conv1 = ConvLayer(ni, nf, **kwargs)
        self.conv2 = ConvLayer(nf, nf, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, up_in:Tensor) -> Tensor:
        up_out = self.shuf(up_in)
        cat_x = self.relu(up_out)
        return self.conv2(self.conv1(cat_x))

def decoder_resnet(y_range, n_out=1):
        return nn.Sequential(UpsampleBlock(512),
                             UpsampleBlock(256),
                             UpsampleBlock(128),
                             UpsampleBlock(64),
                             UpsampleBlock(32),
                             nn.Conv2d(16, n_out, 1),
                             SigmoidRange(*y_range)
                             )

def autoencoder(encoder, y_range): return nn.Sequential(encoder, decoder_resnet(y_range))

y_range = (-3.,3.)
ac_resnet = autoencoder(arch, y_range).cuda()

dec = decoder_resnet(y_range).cuda()

dec(arch(x)).shape

learn = Learner(dls, ac_resnet, loss_func=MSELossFlat())

learn.fit_one_cycle(3)