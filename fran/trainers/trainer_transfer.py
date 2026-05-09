from __future__ import annotations

import torch
import torch._dynamo
from fran.managers.unet import UNetManager
from fran.managers.wandb.wandb import WandbManager
from fran.trainers.base import (
    available_checkpoint_epochs_for_run,
    normalize_checkpoint_path,
    select_source_ckpt,
    switch_ckpt_keys,
)
from fran.trainers.trainer import Trainer
from torch import nn
from utilz.stringz import headline

torch._dynamo.config.suppress_errors = True


class TrainingManagerTransfer(Trainer):
    def __init__(
        self,
        project,
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
            project_title=project.project_title,
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
            # Future work: infer run_name from the ckpt folder layout plus project_title.
            raise NotImplementedError(
                "Explicit ckpt source selection for transfer is not implemented yet."
            )
        if resume_lr is not None:
            return self.resolve_resume_lr_ckpt(run_name=run_name, resume_lr=resume_lr)
        source_ckpt_path = select_source_ckpt(run_name, source_ckpt)
        assert source_ckpt_path is not None, (
            f"No checkpoint found for source run: {run_name}"
        )
        return source_ckpt_path

    def resolve_resume_lr_ckpt(self, run_name: str, resume_lr: float):
        """Select the first local checkpoint at or after the matched W&B LR shift epoch."""
        ckpts = available_checkpoint_epochs_for_run(run_name)
        logger = WandbManager(
            project=self.project,
            run_id=run_name,
            wb_mode="online",
            log_model_checkpoints=False,
        )
        shifts = logger.lr_shift_epoch_map(run_id=run_name)
        deltas = (shifts["lr-Adam"].astype(float) - float(resume_lr)).abs()
        row = shifts.iloc[deltas.argsort()[:1]].iloc[0]
        shift_epoch = int(row["epoch"])
        after = [(epoch, ckpt) for epoch, ckpt in ckpts if epoch >= shift_epoch]
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
        self.replace_final_layer_src_model()
        self.copy_weights()
        if self.freeze == "encoder":
            self.freeze_encoder()

    def freeze_encoder(self):
        """Freeze encoder parameters after source weights have been copied."""
        enc = self.N.model.conv_blocks_context
        for param in enc.parameters():
            param.requires_grad = False

    def replace_final_layer_src_model(self):
        """Swap source heads to the target output channel count before weight copy."""
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


class TrainingManagerTransferRT(TrainingManagerTransfer):
    """Compatibility shim for transfer learning in run-through mode."""

    def __init__(
        self,
        project,
        configs,
        run_name=None,
        freeze=None,
        source_ckpt="interactive",
        resume_lr=None,
        ckpt=None,
    ):
        """Initialize the transfer manager with run-through enabled."""
        super().__init__(
            project=project,
            configs=configs,
            run_name=run_name,
            freeze=freeze,
            source_ckpt=source_ckpt,
            resume_lr=resume_lr,
            ckpt=ckpt,
            run_through=True,
        )
