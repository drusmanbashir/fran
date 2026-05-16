import importlib

import pandas as pd


best_runs = importlib.import_module("fran.inference.best_runs")


def test_collect_run_targets_preserves_mnemonic_and_order():
    payload = {
        "whole": {"runs": ["TOTALSEG-NJUGU", "TOTALSEG-FREHA"]},
        "kidneys": {"runs": ["KITS23-KRETE", "KITS23-SIRIG"], "k_largest": 2},
        "totalseg": {"runs": {"full": ["LITS-1437"], "minimal": ["LITS-1439"]}},
    }

    assert best_runs.collect_run_targets(payload) == [
        ("whole", "TOTALSEG-NJUGU", None),
        ("whole", "TOTALSEG-FREHA", None),
        ("kidneys", "KITS23-KRETE", 2),
        ("kidneys", "KITS23-SIRIG", 2),
        ("totalseg", "LITS-1437", None),
        ("totalseg", "LITS-1439", None),
    ]


def test_update_runs_registry_writes_mnemonic_for_existing_and_pending(
    monkeypatch, tmp_path
):
    registry_csv = tmp_path / "runs_registry.csv"
    pd.DataFrame([{"run_name": "KITS23-KRETE", "mode": "kbd"}]).to_csv(
        registry_csv, index=False
    )

    monkeypatch.setattr(
        best_runs,
        "load_best_runs",
        lambda path=best_runs.BEST_RUNS_PATH: {
            "kidneys": {"runs": ["KITS23-KRETE", "KITS23-SIRIG"], "k_largest": 2},
            "whole": {"runs": ["TOTALSEG-NJUGU"]},
        },
    )
    monkeypatch.setattr(best_runs, "get_wandb_config", lambda: "token")
    monkeypatch.setattr(best_runs.wandb, "Api", lambda: object())
    monkeypatch.setattr(best_runs, "resolve_run", lambda api, run_name: object())
    monkeypatch.setattr(
        best_runs,
        "row_from_run",
        lambda run_name, mnemonic, k_largest, run: {
            "run_name": run_name,
            "mnemonic": mnemonic,
            "k_largest": k_largest,
            "mode": "from_run",
        },
    )

    df = best_runs.update_runs_registry(registry_csv=registry_csv)

    assert df.loc[df.run_name == "KITS23-KRETE", "mnemonic"].item() == "kidneys"
    assert df.loc[df.run_name == "KITS23-KRETE", "k_largest"].item() == 2
    assert df.loc[df.run_name == "KITS23-SIRIG", "mnemonic"].item() == "kidneys"
    assert df.loc[df.run_name == "KITS23-SIRIG", "k_largest"].item() == 2
    assert df.loc[df.run_name == "TOTALSEG-NJUGU", "mnemonic"].item() == "whole"
    assert pd.isna(df.loc[df.run_name == "TOTALSEG-NJUGU", "k_largest"].item())


def test_update_runs_registry_drops_legacy_slash_columns(monkeypatch, tmp_path):
    registry_csv = tmp_path / "runs_registry.csv"
    pd.DataFrame(
        [
            {
                "run_name": "KITS23-KRETE",
                "mode": "kbd",
                "remapping_imported/0": 1,
            }
        ]
    ).to_csv(registry_csv, index=False)

    monkeypatch.setattr(
        best_runs,
        "load_best_runs",
        lambda path=best_runs.BEST_RUNS_PATH: {
            "kidneys": {"runs": ["KITS23-KRETE"]},
        },
    )

    df = best_runs.update_runs_registry(registry_csv=registry_csv)

    assert "remapping_imported/0" not in df.columns
