from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

import fran.trainers.transfer as transfer_module
from fran.trainers.transfer import TrainerTransfer, TrainerTransferRT


class _TinyModel(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.conv_blocks_context = nn.Sequential(nn.Conv3d(1, 2, kernel_size=1))
        self.seg_outputs = nn.ModuleList(
            [nn.Conv3d(2, out_channels, kernel_size=1, bias=False)]
        )


def _make_transfer(out_channels: int, source_out_channels: int):
    trainer = object.__new__(TrainerTransfer)
    trainer.configs = {"model_params": {"out_channels": out_channels}}
    trainer.freeze = None
    trainer.model_source = _TinyModel(source_out_channels)
    trainer.N = SimpleNamespace(model=_TinyModel(out_channels))
    return trainer


def test_update_model_preserves_target_head_on_out_channel_mismatch():
    trainer = _make_transfer(out_channels=2, source_out_channels=3)
    source_encoder = trainer.model_source.conv_blocks_context[0].weight.detach().clone()
    target_head_before = trainer.N.model.seg_outputs[0].weight.detach().clone()

    TrainerTransfer.update_model(trainer)

    target_head_after = trainer.N.model.seg_outputs[0].weight.detach()
    target_encoder_after = trainer.N.model.conv_blocks_context[0].weight.detach()

    assert torch.equal(target_head_after, target_head_before)
    assert torch.equal(target_encoder_after, source_encoder)


def test_update_model_copies_matching_head_weights():
    trainer = _make_transfer(out_channels=2, source_out_channels=2)
    source_head = trainer.model_source.seg_outputs[0].weight.detach().clone()

    TrainerTransfer.update_model(trainer)

    target_head_after = trainer.N.model.seg_outputs[0].weight.detach()
    assert torch.equal(target_head_after, source_head)


def test_resolve_resume_lr_ckpt_raises_for_empty_shift_history(monkeypatch):
    trainer = object.__new__(TrainerTransfer)
    trainer.project = SimpleNamespace()

    monkeypatch.setattr(
        transfer_module,
        "available_checkpoint_epochs_for_run",
        lambda run_name: [(5, Path("epoch=5.ckpt"))],
    )

    class FakeWandbManager:
        def __init__(self, **kwargs):
            pass

        def lr_shift_epoch_map(self, run_id):
            return pd.DataFrame(columns=["lr-Adam", "epoch", "prev_lr"])

    monkeypatch.setattr(transfer_module, "WandbManager", FakeWandbManager)

    with pytest.raises(RuntimeError, match="No logged LR shifts found"):
        TrainerTransfer.resolve_resume_lr_ckpt(trainer, "RUN123", 0.01)


def test_resolve_resume_lr_ckpt_raises_when_no_checkpoint_after_shift(monkeypatch):
    trainer = object.__new__(TrainerTransfer)
    trainer.project = SimpleNamespace()

    monkeypatch.setattr(
        transfer_module,
        "available_checkpoint_epochs_for_run",
        lambda run_name: [(5, Path("epoch=5.ckpt"))],
    )

    class FakeWandbManager:
        def __init__(self, **kwargs):
            pass

        def lr_shift_epoch_map(self, run_id):
            return pd.DataFrame(
                [{"lr-Adam": 0.01, "epoch": 10, "prev_lr": 0.02}]
            )

    monkeypatch.setattr(transfer_module, "WandbManager", FakeWandbManager)

    with pytest.raises(RuntimeError, match="No local checkpoint found at or after epoch 10"):
        TrainerTransfer.resolve_resume_lr_ckpt(trainer, "RUN123", 0.01)


def test_trainer_transfer_rt_uses_project_title_kwarg(monkeypatch):
    captured = {}

    def fake_init(
        self,
        project_title,
        configs,
        run_name=None,
        freeze=None,
        source_ckpt="interactive",
        resume_lr=None,
        ckpt=None,
        run_through=False,
    ):
        captured["project_title"] = project_title
        captured["configs"] = configs
        captured["run_name"] = run_name
        captured["freeze"] = freeze
        captured["source_ckpt"] = source_ckpt
        captured["resume_lr"] = resume_lr
        captured["ckpt"] = ckpt
        captured["run_through"] = run_through

    monkeypatch.setattr(transfer_module.TrainerTransfer, "__init__", fake_init)

    TrainerTransferRT(
        project_title="proj",
        configs={"dataset_params": {}, "model_params": {}},
        run_name="RUN123",
        freeze="encoder",
        source_ckpt="last",
        resume_lr=0.01,
        ckpt=None,
    )

    assert captured["project_title"] == "proj"
    assert captured["run_name"] == "RUN123"
    assert captured["freeze"] == "encoder"
    assert captured["source_ckpt"] == "last"
    assert captured["resume_lr"] == 0.01
    assert captured["run_through"] is True
