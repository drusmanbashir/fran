from pathlib import Path
from types import SimpleNamespace

import pytest
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


def test_trainer_init_allows_fresh_without_run_name_or_ckpt(monkeypatch):
    monkeypatch.setattr(
        "fran.trainers.trainer.Project",
        lambda project_title: SimpleNamespace(project_title=project_title),
    )

    trainer = Trainer(
        project_title="proj",
        configs={"dataset_params": {"fgbg_ratio": 1}, "model_params": {"lr": 1e-3}},
        run_name=None,
        ckpt=None,
    )

    assert trainer.run_name is None
    assert trainer.ckpt is None


def test_trainer_init_rejects_explicit_ckpt(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "fran.trainers.trainer.Project",
        lambda project_title: SimpleNamespace(project_title=project_title),
    )

    with pytest.raises(NotImplementedError):
        Trainer(
            project_title="proj",
            configs={"dataset_params": {"fgbg_ratio": 1}, "model_params": {"lr": 1e-3}},
            ckpt=tmp_path / "starter.ckpt",
        )


def test_trainer_set_lr_ignores_explicit_lr_on_resume(tmp_path):
    ckpt = tmp_path / "epoch=9.ckpt"
    _write_resume_ckpt(ckpt, epoch=9, lr=1e-3)

    trainer = object.__new__(Trainer)
    trainer.ckpt = ckpt
    trainer.configs = {"model_params": {"lr": 5e-3}}

    Trainer.set_lr(trainer, 1e-2)

    reloaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert reloaded["optimizer_states"][0]["param_groups"][0]["lr"] == 1e-3
    assert reloaded["lr_schedulers"][0]["_last_lr"][0] == 1e-3
    assert trainer.lr == 1e-3
