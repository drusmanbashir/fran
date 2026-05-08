import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

mnemonic = importlib.import_module("fran.run.inference.by_mnemonic")


def test_resolve_spec_tsl_uses_region_family_from_remapping(monkeypatch):
    monkeypatch.setattr(
        mnemonic,
        "load_best_runs",
        lambda path=mnemonic.BEST_RUNS_PATH: {
            "run_w": "TOTALSEG-FREHA",
            "kidneys": {"TSL": ["KITS2-bk"], "yolo": ["KITS23-SIRIG"], "k_largest": 2},
        },
    )
    monkeypatch.setattr(
        mnemonic,
        "load_run_metadata",
        lambda run_name: {"remapping_lbd_kbd": "TSL.full:TSL.kidney"},
    )

    spec = mnemonic.resolve_spec("kits23", "TSL")

    assert spec.inferer_cls is mnemonic.CascadeInferer
    assert spec.run_name == "KITS2-bk"
    assert spec.run_w == "TOTALSEG-FREHA"
    assert spec.localiser_labels == [1]
    assert spec.k_largest == 2


def test_resolve_spec_auto_prefers_yolo_when_tsl_missing(monkeypatch):
    monkeypatch.setattr(
        mnemonic,
        "load_best_runs",
        lambda path=mnemonic.BEST_RUNS_PATH: {
            "run_w": "TOTALSEG-FREHA",
            "kidneys": {"TSL": None, "yolo": ["KITS23-SIRIG"]},
        },
    )
    monkeypatch.setattr(
        mnemonic,
        "load_run_metadata",
        lambda run_name: {"mode": "kbd", "localiser_regions": "abdomen, pelvis"},
    )

    spec = mnemonic.resolve_spec("kidneys", None)

    assert spec.inferer_cls is mnemonic.CascadeInfererYOLO
    assert spec.run_name == "KITS23-SIRIG"
    assert spec.localiser_regions == ["abdomen", "pelvis"]


def test_resolve_input_folder_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        mnemonic.resolve_input_folder(None, None)

    with pytest.raises(ValueError, match="exactly one"):
        mnemonic.resolve_input_folder("/tmp/images", "kits23")


def test_main_uses_dataset_images_folder_for_yolo(monkeypatch, tmp_path):
    dataset_root = tmp_path / "kits23"
    images = dataset_root / "images"
    images.mkdir(parents=True)
    inferer_calls = []
    run_calls = []

    class FakeInferer:
        def __init__(self, **kwargs):
            inferer_calls.append(kwargs)

        def run(self, data, chunksize, overwrite):
            run_calls.append((data, chunksize, overwrite))

    monkeypatch.setattr(
        mnemonic,
        "load_best_runs",
        lambda path=mnemonic.BEST_RUNS_PATH: {
            "run_w": "TOTALSEG-FREHA",
            "kidneys": {"TSL": None, "yolo": ["KITS23-SIRIG"], "k_largest": 2},
        },
    )
    monkeypatch.setattr(
        mnemonic,
        "load_run_metadata",
        lambda run_name: {"mode": "kbd", "localiser_regions": "abdomen, pelvis"},
    )
    monkeypatch.setattr(mnemonic, "CascadeInfererYOLO", FakeInferer)
    monkeypatch.setattr(
        mnemonic,
        "DS",
        {"kits23": SimpleNamespace(folder=dataset_root)},
    )

    args = SimpleNamespace(
        mnemonic="kidneys",
        localiser_type="yolo",
        folder=None,
        dataset="kits23",
        gpus=[1],
        chunksize=3,
        patch_overlap=0.1,
        overwrite=True,
    )

    mnemonic.main(args)

    assert inferer_calls == [
        {
            "localiser_regions": ["abdomen", "pelvis"],
            "run_p": "KITS23-SIRIG",
            "devices": [1],
            "patch_overlap": 0.1,
            "save": True,
            "save_channels": False,
            "k_largest": 2,
        }
    ]
    assert run_calls == [([images], 3, True)]
