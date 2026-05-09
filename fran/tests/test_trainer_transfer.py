from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from fran.trainers import transfer


class _TinyTransferModel(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.conv_blocks_context = nn.Sequential(nn.Conv3d(1, 2, 1, bias=False))
        self.seg_outputs = nn.ModuleList([nn.Conv3d(2, out_channels, 1, bias=False)])


def test_transfer_resolve_resume_lr_rejects_empty_shift_history(monkeypatch):
    trainer = object.__new__(transfer.TrainerTransfer)
    trainer.project = SimpleNamespace(project_title="proj")

    class FakeWandbManager:
        def __init__(self, project, run_id, wb_mode, log_model_checkpoints):
            pass

        def lr_shift_epoch_map(self, run_id=None):
            return pd.DataFrame(
                columns=["trainer/global_step", "lr-Adam", "epoch", "prev_lr"]
            )

    monkeypatch.setattr(
        transfer,
        "available_checkpoint_epochs_for_run",
        lambda run_name: [(1, Path("epoch=1.ckpt"))],
    )
    monkeypatch.setattr(transfer, "WandbManager", FakeWandbManager)

    with pytest.raises(RuntimeError, match="No logged LR shifts found"):
        transfer.TrainerTransfer.resolve_resume_lr_ckpt(
            trainer, run_name="RUN-1", resume_lr=0.01
        )


def test_transfer_resolve_resume_lr_rejects_missing_local_checkpoint_after_shift(
    monkeypatch,
):
    trainer = object.__new__(transfer.TrainerTransfer)
    trainer.project = SimpleNamespace(project_title="proj")

    class FakeWandbManager:
        def __init__(self, project, run_id, wb_mode, log_model_checkpoints):
            pass

        def lr_shift_epoch_map(self, run_id=None):
            return pd.DataFrame(
                [
                    {
                        "trainer/global_step": 100,
                        "lr-Adam": 0.01,
                        "epoch": 5,
                        "prev_lr": 0.1,
                    }
                ]
            )

    monkeypatch.setattr(
        transfer,
        "available_checkpoint_epochs_for_run",
        lambda run_name: [(1, Path("epoch=1.ckpt"))],
    )
    monkeypatch.setattr(transfer, "WandbManager", FakeWandbManager)

    with pytest.raises(RuntimeError, match="No local checkpoint found at or after epoch 5"):
        transfer.TrainerTransfer.resolve_resume_lr_ckpt(
            trainer, run_name="RUN-1", resume_lr=0.01
        )


def test_transfer_update_model_keeps_target_head_initialized_on_out_channel_mismatch(
    monkeypatch,
):
    trainer = object.__new__(transfer.TrainerTransfer)
    trainer.freeze = None
    trainer.configs = {"model_params": {"out_channels": 3}}

    source_model = _TinyTransferModel(out_channels=2)
    target_model = _TinyTransferModel(out_channels=3)
    source_model.conv_blocks_context[0].weight.data.fill_(3.0)
    source_model.seg_outputs[0].weight.data.fill_(5.0)
    target_model.conv_blocks_context[0].weight.data.fill_(7.0)
    target_model.seg_outputs[0].weight.data.fill_(11.0)

    trainer.model_source = source_model
    trainer.N = SimpleNamespace(model=target_model)
    messages = []
    monkeypatch.setattr(transfer, "headline", messages.append)

    transfer.TrainerTransfer.update_model(trainer)

    assert torch.allclose(
        trainer.N.model.conv_blocks_context[0].weight,
        torch.full_like(trainer.N.model.conv_blocks_context[0].weight, 3.0),
    )
    assert torch.allclose(
        trainer.N.model.seg_outputs[0].weight,
        torch.full_like(trainer.N.model.seg_outputs[0].weight, 11.0),
    )
    assert any("remain target-initialized" in message for message in messages)


def test_training_manager_transfer_rt_passes_project_title_keyword(monkeypatch):
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

    monkeypatch.setattr(transfer.TrainerTransfer, "__init__", fake_init)

    transfer.TrainingManagerTransferRT(
        project_title="proj",
        configs={"dataset_params": {"fgbg_ratio": 1}, "model_params": {"lr": 1e-3}},
        run_name="RUN-1",
        resume_lr=0.01,
    )

    assert captured["project_title"] == "proj"
    assert captured["run_name"] == "RUN-1"
    assert captured["resume_lr"] == 0.01
    assert captured["run_through"] is True
