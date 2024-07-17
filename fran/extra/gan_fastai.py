# %%
from functools import partial

from fastai.data.block import DataBlock, TransformBlock
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import *
from fastai.optimizer import RMSProp
from fastai.vision.augment import *
from fastai.vision.data import ImageBlock
from fastai.vision.gan import *
import matplotlib.pyplot as plt

import ipdb

from fran.architectures.unet3d.model import *
tr = ipdb.set_trace
import torch
import torch.nn as nn


bs = 128

size = 64

# %%
dblock = DataBlock(blocks = (TransformBlock, ImageBlock),
                   get_x = generate_noise,
                   get_items = get_image_files,
                   splitter = IndexSplitter([]),
                   item_tfms=Resize(size, method=ResizeMethod.Crop),
                   batch_tfms = Normalize.from_stats(torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])))

path = untar_data(URLs.LSUN_BEDROOMS)

dls = dblock.dataloaders(path, path=path, bs=bs)

dls.show_batch(max_n=16)
plt.show()

# %%
from fastai.callback.all import *

generator = basic_generator(64, n_channels=3, n_extra_layers=1)
critic    = basic_critic   (64, n_channels=3, n_extra_layers=1, act_cls=partial(nn.LeakyReLU, negative_slope=0.2))

learn = GANLearner.wgan(dls, generator, critic, opt_func = RMSProp)

learn.recorder.train_metrics=True
learn.recorder.valid_metrics=False

learn.fit(1, 2e-4, wd=0.)

learn.show_results(max_n=9, ds_idx=0)
