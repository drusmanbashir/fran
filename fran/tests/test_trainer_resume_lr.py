from pathlib import Path

import pandas as pd
import torch

from fran.trainers.trainer import Trainer


def _write_resume_ckpt(path: Path, epoch: int, lr: float, compiled: bool = True) -> None:
    torch.save(
        {
            "epoch": epoch,
            "optimizer_states": [{"param_groups": [{"lr": lr}]}],
            "lr_schedulers": [{"_last_lr": [lr]}],
            "hyper_parameters": {"configs": {"model_params": {"compiled": compiled}}},
        },
        path,
    )


def test_resolve_resume_lr_ckpt_uses_single_checkpoint_after_wandb_match(
    monkeypatch, tmp_path
):
    ckpt = tmp_path / "epoch=9-single.ckpt"
    _write_resume_ckpt(ckpt, epoch=9, lr=1e-3)

    class FakeWandbManager:
        def __init__(self, *args, **kwargs):
            pass

        def lr_shift_epoch_map(self, run_id=None):
            return pd.DataFrame(
                [
                    {
                        "trainer/global_step": 5,
                        "lr-Adam": 1e-4,
                        "epoch": 9,
                        "prev_lr": 1e-3,
                    }
                ]
            )

    monkeypatch.setattr("fran.trainers.trainer.WandbManager", FakeWandbManager)

    trainer = object.__new__(Trainer)
    trainer.project = object()
    trainer.ckpt = ckpt
    trainer.run_name = "RUN-1"
    trainer.available_checkpoint_epochs = lambda: [(9, ckpt)]

    resolved = Trainer.resolve_resume_lr_ckpt(trainer, 1e-4)

    assert resolved == ckpt
    assert trainer.ckpt == ckpt


def test_resolve_resume_lr_ckpt_chooses_first_checkpoint_after_shift(monkeypatch, tmp_path):
    ckpt_early = tmp_path / "epoch=399-snapshot.ckpt"
    ckpt_mid = tmp_path / "epoch=479-val.ckpt"
    ckpt_late = tmp_path / "epoch=599-last.ckpt"
    _write_resume_ckpt(ckpt_early, epoch=399, lr=1e-2)
    _write_resume_ckpt(ckpt_mid, epoch=479, lr=1e-3)
    _write_resume_ckpt(ckpt_late, epoch=599, lr=1e-4)

    class FakeWandbManager:
        def __init__(self, *args, **kwargs):
            pass

        def lr_shift_epoch_map(self, run_id=None):
            return pd.DataFrame(
                [
                    {
                        "trainer/global_step": 38521,
                        "lr-Adam": 1e-3,
                        "epoch": 403,
                        "prev_lr": 1e-2,
                    },
                    {
                        "trainer/global_step": 51781,
                        "lr-Adam": 1e-4,
                        "epoch": 533,
                        "prev_lr": 1e-3,
                    },
                ]
            )

    monkeypatch.setattr("fran.trainers.trainer.WandbManager", FakeWandbManager)

    trainer = object.__new__(Trainer)
    trainer.project = object()
    trainer.run_name = "RUN-1"
    trainer.ckpt = ckpt_late
    trainer.available_checkpoint_epochs = lambda: [
        (399, ckpt_early),
        (479, ckpt_mid),
        (599, ckpt_late),
    ]

    resolved = Trainer.resolve_resume_lr_ckpt(trainer, 1e-3)

    assert resolved == ckpt_mid
    assert trainer.ckpt == ckpt_mid


def test_set_lr_overwrites_source_checkpoint_when_resume_lr_was_not_used(tmp_path):
    ckpt = tmp_path / "epoch=9.ckpt"
    _write_resume_ckpt(ckpt, epoch=9, lr=1e-3)

    trainer = object.__new__(Trainer)
    trainer.ckpt = ckpt
    trainer.resume_lr = None
    trainer.configs = {"model_params": {"lr": 5e-3}}

    Trainer.set_lr(trainer, 1e-2)

    reloaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert reloaded["optimizer_states"][0]["param_groups"][0]["lr"] == 1e-2
    assert reloaded["lr_schedulers"][0]["_last_lr"][0] == 1e-2
    assert trainer.lr == 1e-2
    assert trainer.lr_override is None


def test_set_lr_uses_runtime_override_when_resume_lr_selected_checkpoint(tmp_path):
    ckpt = tmp_path / "epoch=9.ckpt"
    _write_resume_ckpt(ckpt, epoch=9, lr=1e-3)

    trainer = object.__new__(Trainer)
    trainer.ckpt = ckpt
    trainer.resume_lr = 1e-3
    trainer.configs = {"model_params": {"lr": 5e-3}}

    Trainer.set_lr(trainer, 1e-2)

    reloaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert reloaded["optimizer_states"][0]["param_groups"][0]["lr"] == 1e-3
    assert reloaded["lr_schedulers"][0]["_last_lr"][0] == 1e-3
    assert trainer.lr == 1e-2
    assert trainer.lr_override == 1e-2
