# %%
from __future__ import annotations

from pathlib import Path

import torch
import torch._dynamo
from fran.managers.unet import UNetManager
from fran.managers.wandb.wandb import WandbManager
from fran.callback.base import BatchSizeSafetyMargin
from fran.trainers.helpers import (
    available_checkpoint_epochs_for_run,
    normalize_checkpoint_path,
    select_source_ckpt,
    switch_ckpt_keys,
)
from fran.trainers.trainer_rt import BatchSizeFinderRT
from fran.trainers.trainer import Trainer
from utilz.stringz import headline

torch._dynamo.config.suppress_errors = True


class TrainerTransfer(Trainer):
    def __init__(
        self,
        project_title,
        configs,
        run_name=None,
        freeze=None,
        source_ckpt="interactive",
        resume_lr=None,
        ckpt=None,
        run_through: bool = False,
    ):
        """Initialize transfer training from a source run without resume state."""
        assert freeze in [None, "encoder"], "Freeze either None or encoder"
        assert source_ckpt in ["interactive", "last"]
        assert run_name is not None or ckpt is not None, (
            "Please specificy a run or checkpoint to transfer learning from"
        )
        assert not (resume_lr is not None and run_name is None), (
            "resume_lr requires run_name for transfer"
        )
        super().__init__(
            project_title=project_title,
            configs=configs,
            run_name=None,
            run_through=run_through,
        )
        self.freeze = freeze
        self.source_run_name = run_name
        self.resume_lr = float(resume_lr) if resume_lr is not None else None
        self.ckpt_source = self.resolve_source_checkpoint(
            run_name=run_name,
            ckpt=ckpt,
            resume_lr=resume_lr,
            source_ckpt=source_ckpt,
        )
        self.ckpt = None

    def init_dm_unet(self, epochs, batch_size=None, override_dm_checkpoint=False):
        """Build a fresh datamodule/model pair, then copy source weights into the model."""
        if override_dm_checkpoint:
            headline(
                "override_dm_checkpoint has no effect in transfer because no datamodule checkpoint is loaded."
            )
        source_manager = self.load_source_trainer(map_location="cpu")
        self.model_source = source_manager.model
        self.D = self.init_dm()
        self.N = self.init_trainer(epochs)
        self.update_model()
        del self.model_source

    def resolve_source_checkpoint(
        self,
        run_name,
        ckpt,
        resume_lr,
        source_ckpt,
    ):
        """Resolve the checkpoint that will be read for transfer weight loading."""
        if ckpt is not None:
            ckpt_path = Path(ckpt)
            if ckpt_path.exists() is False:
                raise RuntimeError(
                    f"No local checkpoint found for transfer source ckpt: {ckpt_path}"
                )
            # Future work: infer run_name from the ckpt folder layout plus project_title.
            raise NotImplementedError(
                "Explicit ckpt source selection for transfer is not implemented yet."
            )
        try:
            ckpts = available_checkpoint_epochs_for_run(run_name)
        except Exception as exc:
            raise RuntimeError(
                f"No local checkpoints found for transfer source run {run_name}."
            ) from exc
        if resume_lr is not None:
            return self.resolve_resume_lr_ckpt(
                run_name=run_name, resume_lr=resume_lr, ckpts=ckpts
            )
        source_ckpt_path = select_source_ckpt(run_name, source_ckpt)
        assert source_ckpt_path is not None, (
            f"No checkpoint found for source run: {run_name}"
        )
        return source_ckpt_path

    def resolve_resume_lr_ckpt(self, run_name: str, resume_lr: float, ckpts=None):
        """Select the first local checkpoint at or after the matched W&B LR shift epoch."""
        if ckpts is None:
            ckpts = available_checkpoint_epochs_for_run(run_name)
        logger = WandbManager(
            project=self.project,
            run_id=run_name,
            wb_mode="online",
            log_model_checkpoints=False,
        )
        shifts = logger.lr_shift_epoch_map(run_id=run_name)
        if len(shifts) == 0:
            raise RuntimeError(
                f"No logged LR shifts found for transfer source run {run_name}; "
                f"cannot resolve resume_lr={resume_lr}."
            )
        deltas = (shifts["lr-Adam"].astype(float) - float(resume_lr)).abs()
        row = shifts.iloc[deltas.argsort()[:1]].iloc[0]
        shift_epoch = int(row["epoch"])
        after = [(epoch, ckpt) for epoch, ckpt in ckpts if epoch >= shift_epoch]
        if len(after) == 0:
            raise RuntimeError(
                f"No local checkpoint found at or after epoch {shift_epoch} for "
                f"source run {run_name}."
            )
        chosen = after[0][1]
        chosen = normalize_checkpoint_path(chosen)
        headline(
            "transfer resume_lr={} matched logged lr {} (prev_lr {}) at epoch {}; selected {}".format(
                resume_lr,
                row["lr-Adam"],
                row["prev_lr"],
                shift_epoch,
                chosen,
            )
        )
        return chosen

    def load_source_trainer(self, map_location: str = "cpu"):
        """Load the source checkpoint for weight copying without resuming fit state."""
        try:
            source = UNetManager.load_from_checkpoint(
                self.ckpt_source,
                map_location=map_location,
                strict=True,
                weights_only=False,
            )
        except RuntimeError:
            switch_ckpt_keys(self.ckpt_source)
            source = UNetManager.load_from_checkpoint(
                self.ckpt_source,
                map_location=map_location,
                strict=True,
                weights_only=False,
            )
        headline(f"Source model loaded from checkpoint: {self.ckpt_source}")
        return source

    def update_model(self):
        """Copy compatible source weights into the fresh transfer model."""
        self.report_head_mismatch()
        self.copy_weights()
        if self.freeze == "encoder":
            self.freeze_encoder()

    def freeze_encoder(self):
        """Freeze encoder parameters after source weights have been copied."""
        enc = self.N.model.conv_blocks_context
        for param in enc.parameters():
            param.requires_grad = False

    def report_head_mismatch(self):
        """Log target/source head mismatch and leave target heads target-initialized."""
        out_ch_new = self.configs["model_params"]["out_channels"]
        out_ch_old = self.model_source.seg_outputs[-1].out_channels
        if out_ch_old != out_ch_new:
            headline(
                "Source final layer out_channels ({0}) != target ({1}); segmentation heads "
                "will be skipped by shape and remain target-initialized.".format(
                    out_ch_old, out_ch_new
                )
            )

    def copy_weights(self):
        """Copy only matching tensor shapes from the source model into the target model."""
        src_sd = self.model_source.state_dict()
        tgt_sd = self.N.model.state_dict()
        copied, skipped = 0, 0
        with torch.no_grad():
            for key, src_val in src_sd.items():
                tgt_val = tgt_sd.get(key)
                if tgt_val is None or tgt_val.shape != src_val.shape:
                    skipped += 1
                    continue
                tgt_sd[key].copy_(src_val)
                copied += 1
        self.N.model.load_state_dict(tgt_sd, strict=False)
        headline(
            f"Copied {copied} tensors from source model; skipped {skipped} tensors."
        )

    def fit(self):
        """Start transfer fitting without resuming optimizer or scheduler state."""
        try:
            self.trainer.fit(model=self.N, datamodule=self.D, ckpt_path=None)
        except KeyboardInterrupt:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass
            raise


class TrainerTransferRT(TrainerTransfer):
    """Compatibility shim for transfer learning in run-through mode."""

    def __init__(
        self,
        project_title,
        configs,
        run_name=None,
        freeze=None,
        source_ckpt="interactive",
        resume_lr=None,
        ckpt=None,
    ):
        """Initialize the transfer manager with run-through enabled."""
        super().__init__(
            project_title=project_title,
            configs=configs,
            run_name=run_name,
            freeze=freeze,
            source_ckpt=source_ckpt,
            resume_lr=resume_lr,
            ckpt=ckpt,
            run_through=True,
        )

    def init_cbs(
        self,
        cbs,
        wandb,
        batchsize_finder,
        profiler,
        tags,
        description="",
        early_stopping=True,
        early_stopping_monitor="val0_loss_dice",
        early_stopping_mode="min",
        early_stopping_patience=30,
        early_stopping_min_delta=0.0,
        lr_floor=None,
        wandb_grid_epoch_freq: int = 5,
        permanent_checkpoint_every_n_epochs: int = 100,
    ):
        cbs, logger, profiler = super().init_cbs(
            cbs=cbs,
            wandb=wandb,
            batchsize_finder=False,
            profiler=profiler,
            tags=tags,
            description=description,
            early_stopping=early_stopping,
            early_stopping_monitor=early_stopping_monitor,
            early_stopping_mode=early_stopping_mode,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            lr_floor=lr_floor,
            wandb_grid_epoch_freq=wandb_grid_epoch_freq,
            permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
        )
        if batchsize_finder:
            cbs[1:1] = [
                BatchSizeFinderRT(batch_arg_name="batch_size", mode="binsearch"),
                BatchSizeSafetyMargin(),
            ]
        return cbs, logger, profiler


# %%
# SECTION: -------------------- SETUP-------------------------------------------------------------------------------------- P = Project("nodes") <CR> <CR> <CR> <CR> <CR>
if __name__ == "__main__":
    from fran.configs.parser import ConfigMaker
    from fran.managers import Project
    from fran.utils.common import *
    from utilz.helpers import pp

    P = Project("lidc")
    P = Project("totalseg")
    P = Project("kits23")
    C = ConfigMaker(P)
    C.setup(2)

    conf = C.configs
    print(conf["model_params"])

    planT = conf["plan_train"]
    planV = conf["plan_valid"]
    pp(planT)

    planT["mode"]
    # add_plan_to_db(plan,"/r/datasets/preprocessed/totalseg/lbd/spc_100_100_100_plan5",P.db)
    # if (lm==3).any():
    #     print("Bad values 3 ->0")
    #     lm[lm==3]=1
    #     torch.save(lm, bad_case_fn)
    #
    # find_matching_fn(Path(bad_names[0])[0],fixed, tags=["all"])

    # fldr= DS.kits23.folder
    # fn = fldr/("label_analysis/lesion_stats.csv")
    # df = pd.read_csv(fn)
    # counts = df.groupby("case_id").size()
    # counts2 = counts.sort_values(ascending=False)
    # bb= counts2.index[:200]
# SECTION:-------------------- TRAINING-------------------------------------------------------------------------------------- <CR> <CR> <CR> devices = 2 <CR> <CR> <CR> <CR> <CR> <CR>
# %%
    bs = 16
    device_id = 0
    batchsize_finder = False
    batchsize_finder = True
    batch_tfms = True
    wandb = False
    wandb = True
    override_dm = False
    override_dm = True

    run_name = None
    run_name = "TOTALSEG-NJUGU"
    run_name = "KITS23-SIRIG"
    tags = []
    description = f""
    conf["dataset_params"]["fold"] = 0
    lr = None
    debug_ = False
    profiler = False
    compiled = False
    cbs = []
    wandb_grid_epoch_freq = 20
    val_every_n_epochs = 2
    train_indices = None
# %%
# SECTION:--------------------  TRAINING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    Tm = TrainerTransferRT(P.project_title, conf, run_name,resume_lr=0.01,source_ckpt="interactive",freeze=None)
# %%
    Tm.setup(
        compiled=compiled,
        train_indices=train_indices,
        batch_tfms=batch_tfms,
        cbs=cbs,
        debug=debug_,
        batch_size=bs,
        devices=[device_id],
        epochs=600,
        batchsize_finder=batchsize_finder,
        wandb=wandb,
        wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        tags=tags,
        description=description,
    )
# %%

    Tm.fit()
    # model(inputs)

# %%
    N = Tm.N
    D = Tm.D
    D.setup()
    D.prepare_data()
    tmt = D.train_manager
    tmv = D.valid_manager

# %%
    tmt.collate_fn
    tmv.collate_fn
    tmv.batch_size
    tmv.prepare_data()
# %%
    tmv.setup()
    dl = tmv.dl
    iteri = iter(dl)
# %%
    for i, batch in enumerate(iteri):
        print(batch["image"].shape)

# %%
    ds = tmv.ds
    ds[0]
    dici = ds[0]
    dici2 = ds[1]
# %%
    tmt.setup()
    dl2 = tmt.dl
    iteri2 = iter(dl2)
# %%
    for i, batch in enumerate(iteri2):
        print(batch["image"].shape)
# %%

    patch_overlap = 0
    mode = "constant"
    device = "cpu"
    sw_device = "cuda:1"
    bs = 1  # start lower if you are hitting OOM
# %%
# %%
