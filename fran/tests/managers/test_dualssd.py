from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import fran.managers.data.training as training_module
import fran.trainers.trainer as trainer_module
import fran.trainers.trainer_runthrough as trainer_runthrough_module
from fran.managers.data.dualssd import DataManagerSourceDualSSD
from fran.managers.data.training import DataManagerSource, DataManagerWhole
from fran.preprocessing.preprocessor import create_hdf5_shards
from fran.trainers.trainer import Trainer
from fran.trainers.trainer_runthrough import TrainerRT


def _write_case(output_folder: Path, case_id: str, shape=(32, 24, 16)):
    images = output_folder / "images"
    lms = output_folder / "lms"
    indices = output_folder / "indices"
    for folder in (images, lms, indices):
        folder.mkdir(parents=True, exist_ok=True)

    image = torch.zeros(shape, dtype=torch.float32)
    lm = torch.zeros(shape, dtype=torch.uint8)
    idx = {
        "lm_fg_indices": torch.arange(16, dtype=torch.int64),
        "lm_bg_indices": torch.arange(64, dtype=torch.int64),
        "meta": {"filename_or_obj": f"/tmp/{case_id}.nii.gz", "case_id": case_id},
    }

    torch.save(image, images / f"{case_id}.pt")
    torch.save(lm, lms / f"{case_id}.pt")
    torch.save(idx, indices / f"{case_id}.pt")


def _trainer_configs():
    return {
        "dataset_params": {
            "batch_size": 2,
            "cache_rate": 0.0,
            "device": "cpu",
            "ds_type": None,
        },
        "plan_train": {"mode": "source"},
        "plan_valid": {"mode": "whole"},
    }


def test_dualssd_copies_odd_hdf5_shards_and_rewrites_manifest(tmp_path, monkeypatch):
    rapid_access_folder = tmp_path / "rapid_access_folder"
    rapid_access_folder2 = tmp_path / "rapid_access_folder2"
    output_folder = rapid_access_folder / "project_a" / "fixed_spacing"

    monkeypatch.setitem(
        training_module.COMMON_PATHS,
        "rapid_access_folder",
        str(rapid_access_folder),
    )
    monkeypatch.setitem(
        training_module.COMMON_PATHS,
        "rapid_access_folder2",
        str(rapid_access_folder2),
    )

    for case_id in ("case_000", "case_001", "case_002"):
        _write_case(output_folder, case_id)

    shards = create_hdf5_shards(
        output_folder=output_folder,
        src_dims=(192, 192, 128),
        cases_per_shard=2,
    )

    manager = DataManagerSourceDualSSD.__new__(DataManagerSourceDualSSD)
    manager.data_folder = output_folder
    manager.plan = {"src_dims": (192, 192, 128)}
    staged = manager.copy_hdf5_shards_to_rapid_access_folder2(
        [{"case_id": case_id, "data_folder": str(output_folder)} for case_id in ("case_000", "case_001", "case_002")]
    )

    staged_folder = rapid_access_folder2 / "project_a" / "fixed_spacing"
    staged_manifest = staged_folder / "hdf5_shards" / "src_192_192_128" / "manifest.json"
    manifest = json.loads(staged_manifest.read_text())

    assert all(item["data_folder"] == str(staged_folder) for item in staged)
    assert manifest["shards"][0]["shard"] == str(shards[0])

    expected_staged_shard = (
        rapid_access_folder2
        / "project_a"
        / "fixed_spacing"
        / "hdf5_shards"
        / "src_192_192_128"
        / "shard_0001.h5"
    )
    assert manifest["shards"][1]["shard"] == str(expected_staged_shard)
    assert expected_staged_shard.exists()


def test_trainer_init_dm_uses_standard_datamodule_without_dualssd(monkeypatch):
    captured = {}

    class FakeDataManagerDual:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    class FakeDataManagerDualSSD:
        def __init__(self, **kwargs):
            raise AssertionError("dual-SSD datamodule should not be selected")

    monkeypatch.setattr(trainer_module, "DataManagerDual", FakeDataManagerDual)
    monkeypatch.setattr(trainer_module, "DataManagerDualSSD", FakeDataManagerDualSSD)
    monkeypatch.setattr(
        trainer_module,
        "infer_labels_and_update_out_channels",
        lambda dm, configs: None,
    )

    trainer = Trainer.__new__(Trainer)
    trainer.project = SimpleNamespace(project_title="proj")
    trainer.configs = _trainer_configs()
    trainer.train_indices = None
    trainer.val_indices = None
    trainer.val_sampling = 1.0
    trainer.val_every_n_epochs = 5
    trainer.dual_ssd = False

    dm = Trainer.init_dm(trainer)

    assert isinstance(dm, FakeDataManagerDual)
    assert captured["kwargs"]["manager_class_train"] is DataManagerSource
    assert captured["kwargs"]["manager_class_valid"] is DataManagerWhole


def test_trainer_rt_init_dm_wraps_manager_class_for_dualssd(monkeypatch):
    captured = {}

    class FakeDataManagerRT:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    class WrappedManager:
        pass

    monkeypatch.setattr(trainer_runthrough_module, "DataManagerRT", FakeDataManagerRT)
    monkeypatch.setattr(
        trainer_runthrough_module,
        "dual_ssd_manager_class",
        lambda manager_class: WrappedManager,
    )
    monkeypatch.setattr(
        trainer_runthrough_module,
        "infer_labels_and_update_out_channels",
        lambda dm, configs: None,
    )

    trainer = TrainerRT.__new__(TrainerRT)
    trainer.project = SimpleNamespace(project_title="proj")
    trainer.configs = _trainer_configs()
    trainer.train_indices = None
    trainer.debug = False
    trainer.dual_ssd = True

    dm = TrainerRT.init_dm(trainer)

    assert isinstance(dm, FakeDataManagerRT)
    assert captured["kwargs"]["manager_class"] is WrappedManager
