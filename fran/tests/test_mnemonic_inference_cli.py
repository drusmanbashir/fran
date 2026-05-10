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
            "whole": {"runs": ["TOTALSEG-FREHA"]},
            "kidneys": {
                "runs": {"TSL": ["KITS2-bk"], "yolo": ["KITS23-SIRIG"]},
                "k_largest": 2,
            },
        },
    )
    monkeypatch.setattr(
        mnemonic,
        "load_run_metadata",
        lambda run_name: {"mode": "lbd", "remapping_lbd_kbd": "TSL.full:TSL.kidney"},
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
            "whole": {"runs": ["TOTALSEG-FREHA"]},
            "kidneys": {"runs": {"TSL": None, "yolo": ["KITS23-SIRIG"]}},
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


def test_resolve_input_folders_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        mnemonic.resolve_input_folders(None, None)

    with pytest.raises(ValueError, match="exactly one"):
        mnemonic.resolve_input_folders("/tmp/images", ["kits23"])


def test_resolve_spec_auto_selects_first_ordered_run(monkeypatch, capsys):
    monkeypatch.setattr(
        mnemonic,
        "load_best_runs",
        lambda path=mnemonic.BEST_RUNS_PATH: {
            "whole": {"runs": ["TOTALSEG-FREHA"]},
            "kidneys": {"runs": {"TSL": ["KITS2-bk"], "yolo": ["KITS23-SIRIG"]}},
        },
    )
    monkeypatch.setattr(
        mnemonic,
        "load_run_metadata",
        lambda run_name: {
            "KITS2-bk": {"mode": "lbd", "remapping_lbd": "TSL.full:TSL.kidney"},
            "KITS23-SIRIG": {"mode": "rbd", "localiser_regions": "abdomen, pelvis"},
        }[run_name],
    )

    spec = mnemonic.resolve_spec("kidneys", None)

    assert spec.inferer_cls is mnemonic.CascadeInferer
    assert spec.run_name == "KITS2-bk"
    assert capsys.readouterr().out.strip().endswith(
        "{'run_names': ['KITS2-bk', 'KITS23-SIRIG'], 'selected_run': 'KITS2-bk'}"
    )


def test_resolve_spec_explicit_localiser_type_uses_registry_mode_bucket(monkeypatch):
    monkeypatch.setattr(
        mnemonic,
        "load_best_runs",
        lambda path=mnemonic.BEST_RUNS_PATH: {
            "whole": {"runs": ["TOTALSEG-FREHA"]},
            "kidneys": {"runs": {"TSL": ["KITS23-SIRIG"], "yolo": ["KITS2-bk"]}},
        },
    )

    monkeypatch.setattr(
        mnemonic,
        "load_run_metadata",
        lambda run_name: {
            "KITS23-SIRIG": {"mode": "kbd", "localiser_regions": "abdomen, pelvis"},
            "KITS2-bk": {"mode": "lbd", "remapping_lbd": "TSL.full:TSL.kidney"},
        }[run_name],
    )

    spec = mnemonic.resolve_spec("kidneys", "yolo")

    assert spec.inferer_cls is mnemonic.CascadeInfererYOLO
    assert spec.run_name == "KITS23-SIRIG"
    assert spec.localiser_regions == ["abdomen", "pelvis"]


def test_resolve_spec_prefers_minimal_for_nested_standalone_runs(monkeypatch):
    monkeypatch.setattr(
        mnemonic,
        "load_best_runs",
        lambda path=mnemonic.BEST_RUNS_PATH: {
            "totalseg": {"runs": {"full": ["FULL-RUN"], "minimal": ["MIN-RUN"]}},
        },
    )
    monkeypatch.setattr(mnemonic, "load_run_metadata", lambda run_name: {})
    monkeypatch.setattr(mnemonic, "resolve_standalone_inferer_cls", lambda run_name: mnemonic.WholeImageInferer)

    spec = mnemonic.resolve_spec("totalseg", None)

    assert spec.run_name == "FULL-RUN"
    assert spec.inferer_cls is mnemonic.WholeImageInferer


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
            "whole": {"runs": ["TOTALSEG-FREHA"]},
            "kidneys": {
                "runs": {"TSL": None, "yolo": ["KITS23-SIRIG"]},
                "k_largest": 2,
            },
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
        dataset=["kits23"],
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
