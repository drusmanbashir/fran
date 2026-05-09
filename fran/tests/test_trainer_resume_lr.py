from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.callbacks import BatchSizeFinder, EarlyStopping, ModelCheckpoint

from fran.managers.data.main import DataManagerPatch
from fran.trainers.trainer import Trainer
from fran.trainers.trainer import FranBatchSizeFinder
from fran.trainers.trainer_runthrough import CaseIDRecorderRT


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


def test_trainer_monitor_metric_name_remaps_for_run_through():
    trainer = object.__new__(Trainer)
    trainer.run_through = True

    assert Trainer.monitor_metric_name(trainer, "val0_loss") == "train0_loss"
    assert Trainer.monitor_metric_name(trainer, "train0_loss") == "train0_loss"


def test_trainer_resolve_datamanager_accepts_sourcepbd_in_run_through():
    trainer = object.__new__(Trainer)
    trainer.run_through = True

    dm_class = Trainer.resolve_datamanager(trainer, "sourcepbd")

    assert dm_class is DataManagerPatch


def test_trainer_init_cbs_uses_run_through_callback_semantics():
    trainer = object.__new__(Trainer)
    trainer.run_through = True
    trainer.debug = False
    trainer.project = SimpleNamespace(project_title="proj")
    trainer.configs = {
        "plan_train": {"vip_label": 1},
        "model_params": {"out_channels": 2},
    }

    cbs, logger, profiler = Trainer.init_cbs(
        trainer,
        cbs=[],
        wandb=False,
        batchsize_finder=True,
        profiler=False,
        tags=[],
        description="",
        early_stopping=True,
        early_stopping_monitor="val0_loss_dice",
        early_stopping_mode="min",
        early_stopping_patience=3,
        early_stopping_min_delta=0.0,
        lr_floor=None,
        wandb_grid_epoch_freq=5,
        permanent_checkpoint_every_n_epochs=25,
    )

    checkpoint_callbacks = [cb for cb in cbs if isinstance(cb, ModelCheckpoint)]
    early_stopping = [cb for cb in cbs if isinstance(cb, EarlyStopping)]
    batch_finders = [cb for cb in cbs if isinstance(cb, BatchSizeFinder)]

    assert logger is None
    assert profiler is None
    assert isinstance(cbs[0], CaseIDRecorderRT)
    assert type(batch_finders[0]) is BatchSizeFinder
    assert all(type(cb) is not FranBatchSizeFinder for cb in batch_finders)
    assert checkpoint_callbacks[0].monitor == "train0_loss"
    assert checkpoint_callbacks[0]._save_on_train_epoch_end is True
    assert early_stopping[0].monitor == "train0_loss_dice"
    assert early_stopping[0]._check_on_train_epoch_end is True
