# %%
import ipdb

from fran.configs.parser import ConfigMaker
from fran.managers.unet import UNetManager

tr = ipdb.set_trace
import torch._dynamo

torch._dynamo.config.suppress_errors = True
import warnings

import torch
from torch import nn

from fran.trainers.trainer import Trainer

# TODO: fix LR  setup in Tranfer learning


class TrainingManagerTransfer(Trainer):
    def __init__(self, project, config, run_name, freeze=None):
        assert freeze in [None, "encoder"], "Freeze either None or encoder"
        assert run_name is not None, "Please specificy a run to transfer learning from"
        super().__init__(
            project_title=project.project_title, configs=config, run_name=run_name
        )
        self.freeze = freeze

    def init_dm_unet(self, epochs, batch_size=None, override_dm_checkpoint=False):
        Ntmp = self.load_trainer(max_epochs=epochs, map_location="cpu")
        self.model_source = Ntmp.model
        self.N = self.init_trainer(epochs)
        self.D = self.init_dm()
        self.update_model()
        del self.model_source
        # self.update_trainer()

    def update_trainer(self):
        self.N.model_params = self.configs["model_params"]

    def update_model(self):
        self.replace_final_layer_src_model()
        self.copy_weights()
        if self.freeze == "encoder":
            self.freeze_encoder()

    def freeze_encoder(self):
        enc = self.N.model.conv_blocks_context
        for param in enc.parameters():
            param.requires_grad = False

    def replace_final_layer_src_model(self):
        out_ch_new = self.configs["model_params"]["out_channels"]
        for i, current in enumerate(self.model_source.seg_outputs):
            in_ch, out_ch_old, kernel, stride = (
                current.in_channels,
                current.out_channels,
                current.kernel_size,
                current.stride,
            )
            newhead = nn.Conv3d(in_ch, out_ch_new, kernel, stride, bias=False)
            self.model_source.seg_outputs[i] = newhead
        print(
            "----------------------------------\nReplacing final layer outchannels ({0} to {1})\n---------------------------------)".format(
                out_ch_old, out_ch_new
            )
        )

    def copy_weights(self):
        tot = 0
        failed = 0
        with torch.no_grad():
            for paramA, paramB in zip(
                self.N.model.parameters(), self.model_source.parameters()
            ):
                num = paramA.numel()
                try:
                    paramA.copy_(paramB)
                    tot += num
                except Exception as e:
                    print("!" * 40)
                    print(e)
                    failed += num
        print("Coped total: ", tot)
        print("Failed: ", failed)
        print("-" * 40)

    def fit(self):
        self.trainer.fit(model=self.N, datamodule=self.D, ckpt_path=None)


# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR>
if __name__ == "__main__":
    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")

    from fran.managers import Project
    from fran.utils.common import *

    P = Project("nodes")
    # conf['model_params']['lr']=1e-3

    # P.add_data([DS.totalseg])
    C = ConfigMaker(P, configuration_filename=None)
    C.setup(1)
    C.plans
    conf = C.configs
    print(conf["model_params"])

    plan = conf["plan_train"]
    pp(plan)

# %%
    device_id = 1
    run_name = None
    freeze = "encoder"
    freeze = None
    bs = 10  # if none, will get it from the conf file
    run_name = "LITS-811"
    run_name = "LITS-919"
    run_name = "LITS-949"
    run_name = "LITS-911"
    run_name = "LITS-1290"
    # run_name ='LITS-836'
    compiled = False
    profiler = False

    batch_finder = False
    neptune = True
    tags = []
    cache_rate = 0.0
    description = f"Transfer learning from {run_name}. Freeze: {freeze}"

    Tm = TrainingManagerTransfer(
        project=P, config=conf, run_name=run_name, freeze=freeze
    )
# %%
    Tm.setup(
        lr=1e-2,
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=500,
        batchsize_finder=batch_finder,
        profiler=profiler,
        neptune=neptune,
        tags=tags,
        description=description,
    )
# %%
    # Tm.D.batch_size=8
    Tm.N.compiled = compiled
# %%
    Tm.fit()

# %%
# SECTION:-------------------- Tinkering with N-------------------------------------------------------------------------------------- <CR>

    Tm.N.model.seg_outputs[-1]
    N = Tm.N

    enc = N.model.conv_blocks_context
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
    Tm.N.model.to("cuda:1")
    pred = Tm.N(batch["image"].to(1))
    print(pred[0].shape)
    #

# %%
    m1 = Tm.Ntmp.model
    m2 = Tm.N.model
    m2.load_state_dict(m1)
    m2.state_dict()
# %%
    tot = 0
    failed = 0
    with torch.no_grad():
        for paramA, paramB in zip(m2.parameters(), m1.parameters()):
            num = paramA.numel()
            try:
                paramA.copy_(paramB)
                tot += num
            except Exception as e:
                print(e)
                failed += num
    print("Coped total", tot)
    print("Failed: ", failed)
    print("-" * 40)

# %%
# %%
    m = Tm.N.model
    [print(mm) for mm in m.named_modules()]
    Tm.N.model.seg_outputs
    Tm.model_source.seg_outputs

# %%
    Tm.lr = 1e-3
    Tm.sync_dist = True
    epochs = 500
    Ntmp = Tm.load_trainer(max_epochs=epochs, map_location="cpu")
    Tm.model_source = Ntmp.model

    Tm.N = Tm.init_trainer(epochs)
    Tm.D = Tm.init_dm(cache_rate)
    Tm.update_model()
    # Tm.update_trainer()
# %%

    Tm.replace_final_layer_src_model()
    Tm.copy_weights()
    if Tm.freeze == "encoder":
        Tm.freeze_encoder()
