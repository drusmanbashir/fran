from pathlib import Path

import pandas as pd
import pytest

from fran.run.inference import by_mnemonic


def test_parser_requires_exactly_one_input_source():
    parser = by_mnemonic.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["kidneys"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["kidneys", "--folder", "/tmp/images", "--dataset", "kits23"]
        )


def test_resolve_input_folder_prefers_dataset_images(monkeypatch):
    class _Spec:
        folder = Path("/tmp/kits23")

    monkeypatch.setattr(by_mnemonic, "DS", {"kits23": _Spec()})
    assert by_mnemonic.resolve_input_folder(None, "kits23") == Path(
        "/tmp/kits23/images"
    )


def test_resolve_inference_spec_defaults_to_tsl_and_uses_minimal_run(monkeypatch):
    best_runs = {
        "run_w": "TOTALSEG-FREHA",
        "totalseg": {"full": ["FULL-RUN"], "minimal": ["MIN-RUN"]},
        "kidneys": {
            "TSL": ["KITS2-bk"],
            "yolo": ["KITS23-SIRIG"],
            "k_largest": 2,
        },
    }
    runs_registry = pd.DataFrame(
        [{"run_name": "KITS2-bk", "remapping_lbd_kbd_code": "TSL.label_full:TSL.label_minimal"}]
    )

    monkeypatch.setattr(by_mnemonic, "load_run_plan", lambda run_name: {"mode": "kbd"})

    spec = by_mnemonic.resolve_inference_spec(
        mnemonic_raw="kidneys",
        localiser_type=None,
        best_runs=best_runs,
        runs_registry=runs_registry,
        k_largest_override=None,
    )

    assert spec.localiser_type == "TSL"
    assert spec.run_p == "KITS2-bk"
    assert spec.run_w == "MIN-RUN"
    assert spec.localiser_labels == [4]
    assert spec.k_largest == 2


def test_resolve_inference_spec_uses_yolo_regions_from_registry(monkeypatch):
    best_runs = {
        "run_w": "TOTALSEG-FREHA",
        "totalseg": {"full": ["FULL-RUN"], "minimal": ["MIN-RUN"]},
        "kidneys": {"TSL": ["KITS2-bk"], "yolo": ["KITS23-SIRIG"]},
    }
    runs_registry = pd.DataFrame(
        [{"run_name": "KITS23-SIRIG", "localiser_regions": "abdomen, pelvis"}]
    )

    spec = by_mnemonic.resolve_inference_spec(
        mnemonic_raw="kidneys",
        localiser_type="yolo",
        best_runs=best_runs,
        runs_registry=runs_registry,
        k_largest_override=None,
    )

    assert spec.localiser_type == "yolo"
    assert spec.run_p == "KITS23-SIRIG"
    assert spec.localiser_regions == ["abdomen", "pelvis"]
