import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from fran.managers.data.main import DataManagerPatch
from fran.trainers.trainer import Trainer
from fran.trainers.trainer_rt import BatchSizeFinderRT, CaseIDRecorderRT, TrainerRT


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
    trainer = object.__new__(TrainerRT)
    trainer.run_through = True
    trainer.debug = False
    trainer.project = SimpleNamespace(project_title="proj")
    trainer.configs = {
        "plan_train": {"vip_label": 1},
        "model_params": {"out_channels": 2},
    }

    cbs, logger, profiler = TrainerRT.init_cbs(
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
    batch_finders = [cb for cb in cbs if isinstance(cb, BatchSizeFinderRT)]

    assert logger is None
    assert profiler is None
    assert isinstance(cbs[0], CaseIDRecorderRT)
    assert type(batch_finders[0]) is BatchSizeFinderRT
    assert checkpoint_callbacks[0].monitor == "train0_loss"
    assert checkpoint_callbacks[0]._save_on_train_epoch_end is True
    assert early_stopping[0].monitor == "train0_loss_dice"
    assert early_stopping[0]._check_on_train_epoch_end is True


def test_trainer_rt_setup_omits_validation_and_early_stopping_kwargs():
    params = inspect.signature(TrainerRT.setup).parameters

    assert "val_indices" not in params
    assert "val_sampling" not in params
    assert "val_every_n_epochs" not in params
    assert "early_stopping" not in params
    assert "early_stopping_monitor" not in params
    assert "early_stopping_mode" not in params
    assert "early_stopping_patience" not in params
    assert "early_stopping_min_delta" not in params


def test_trainer_rt_setup_rejects_validation_and_early_stopping_kwargs():
    trainer = TrainerRT.__new__(TrainerRT)

    with pytest.raises(TypeError):
        TrainerRT.setup(trainer, val_every_n_epochs=2)

    with pytest.raises(TypeError):
        TrainerRT.setup(trainer, early_stopping=True)


def test_batch_size_finder_rt_restores_checkpoint_with_weights_only_false(monkeypatch):
    finder = BatchSizeFinderRT()
    calls = {}

    def fake_scale_batch_size_rt(
        trainer,
        mode,
        steps_per_trial,
        init_val,
        max_trials,
        batch_arg_name,
    ):
        trainer._checkpoint_connector.restore("probe.ckpt")
        calls["mode"] = mode
        calls["batch_arg_name"] = batch_arg_name
        return 6

    trainer = SimpleNamespace(
        _checkpoint_connector=SimpleNamespace(
            restore=lambda checkpoint_path=None, weights_only=None: calls.update(
                checkpoint_path=checkpoint_path, weights_only=weights_only
            )
        )
    )

    monkeypatch.setattr(
        "fran.trainers.trainer_rt._scale_batch_size_rt", fake_scale_batch_size_rt
    )

    finder.scale_batch_size(trainer, pl_module=object())

    assert calls["checkpoint_path"] == "probe.ckpt"
    assert calls["weights_only"] is False
    assert calls["mode"] == "power"
    assert calls["batch_arg_name"] == "batch_size"
    assert finder.optimal_batch_size == 6
