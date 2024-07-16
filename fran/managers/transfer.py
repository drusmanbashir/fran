## 

import ipdb
tr = ipdb.set_trace
import torch._dynamo

from fran.evaluation.losses import CombinedLoss, DeepSupervisionLoss
from fran.managers.data import ( DataManagerLBD, DataManagerPatch,
                                 DataManagerSource)
from fran.utils.fileio import load_yaml
from fran.utils.imageviewers import ImageMaskViewer

torch._dynamo.config.suppress_errors = True
from fran.managers.nep import NeptuneManager
import itertools as il
import operator
import warnings
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ( LearningRateMonitor,
                                         TQDMProgressBar, DeviceStatsMonitor)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fran.architectures.create_network import (create_model_from_conf, 
                                               pool_op_kernels_nnunet)
from torch import nn
import torch.nn.functional as F
from fran.transforms.spatialtransforms import one_hot
try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass

import torch
import torch
from fastcore.basics import store_attr
from fran.managers.training import TrainingManager, UNetTrainer, checkpoint_from_model_id

#TODO: fix LR  setup in Tranfer learning


class TrainingManagerTransfer(TrainingManager):
    def __init__(self, project, configs, run_name, freeze=None):
        assert freeze in [None,'encoder'],"Freeze either None or encoder"
        assert run_name is not None, "Please specificy a run to transfer learning from"
        super().__init__(project, configs, run_name)
        self.freeze=freeze
        self.run_name = None
    def init_dm_unet(self, epochs):
            self.N = self.load_trainer(max_epochs= epochs)
            self.D = self.init_dm(cache_rate)
            self.update_model()
            self.update_trainer()


    def update_trainer(self):
        self.N.model_params = self.configs['model_params']



    def update_model(self):
            if self.freeze=='encoder':
                self.freeze_encoder()
            self.replace_final_layer()

    def freeze_encoder(self):
        enc= self.N.model.conv_blocks_context
        for param in enc.parameters():
            param.requires_grad = False

    def replace_final_layer(self):
        out_ch_new = self.configs['model_params']['out_channels']
        for i, current in enumerate(self.N.model.seg_outputs):
            in_ch,out_ch_old, kernel,stride = current.in_channels,current.out_channels, current.kernel_size,current.stride
            newhead = nn.Conv3d(in_ch,out_ch_new,kernel,stride)
            self.N.model.seg_outputs[i]=newhead
        print("----------------------------------\nReplacing final layer outchannels ({0} to {1})\n---------------------------------)".format(out_ch_old,out_ch_new))

    def fit(self):
        self.trainer.fit(model=self.N, datamodule=self.D, ckpt_path=None)


# %%
if __name__ == "__main__":
    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *
    from torch.profiler import profile, record_function, ProfilerActivity
    project_title = "nodes"
    proj = Project(project_title=project_title)

    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    )
    configuration_filename = None

    config = ConfigMaker(
        proj, raytune=False
    ).config

    global_props = load_dict(proj.global_properties_filename)
    # conf['model_params']['lr']=1e-3

# %%
    device_id = 1
    run_name = None
    bs =5# if none, will get it from the conf file 
    run_name = "LITS-811"
    run_name ='LITS-919'
    run_name = "LITS-949"
    run_name = "LITS-911"
    # run_name ='LITS-836'
    compiled = False
    profiler=False

    batch_finder = False
    neptune = True
    tags = []
    cache_rate=0.0
    description = f" "
    Tm = TrainingManagerTransfer(project= proj, configs =config, run_name= run_name,freeze='encoder')
# %%
    Tm.setup(
        lr = 1e-3,
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=500,
        batchsize_finder=batch_finder,
        profiler=profiler,
        neptune=neptune,
        tags=tags,
        description=description,
        cache_rate=cache_rate
    )
# %%
    # Tm.D.batch_size=8
    Tm.N.compiled = compiled
# %%
    Tm.fit()
 
# %%
#SECTION:-------------------- Tinkering with N--------------------------------------------------------------------------------------

    Tm.N.model.seg_outputs[-1]
    N = Tm.N

    enc= N.model.conv_blocks_context
    for param in enc.parameters():
        param.requires_grad = False

# %%
    N.freeze()
    cc = list(N.children())
    ccc = list(cc[0].children())
    ccc[-1]

# %%
    Tm.D.setup()
    D = Tm.D
    ds = Tm.D.train_ds
    ds = Tm.D.valid_ds
# %%
    dl = Tm.D.train_dataloader()
    dl2 = Tm.D.val_dataloader()
    iteri = iter(dl)
    iteri2 = iter(dl2)
    batch = next(iteri2)
    Tm.N.model.to('cuda:1')
    pred = Tm.N(batch['image'].to(1))
    print(pred[0].shape)
#



# %%
    m =Tm.N.model
    [print(mm) for mm in m.named_modules()]
# %%

