from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from utilz.fileio import load_yaml

from fran.configs.mnemonics import Mnemonics
from fran.configs.parser import ConfigMaker
from fran.managers.project import Project
from fran.run.analyze_resample import process_plan


@dataclass
class PlanRun:
    project: str
    mnemonic: str
    plan_id: int
    ok: bool
    stage: str
    error: str | None = None


@dataclass
class ProjToAnalyzeReport:
    runs: list[PlanRun] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.runs if r.ok)

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.runs if not r.ok)


def _common_paths() -> dict[str, Any]:
    conf_root = Path(os.environ["FRAN_CONF"])
    return load_yaml(conf_root / "config.yaml")


def _projects_folder() -> Path:
    return Path(_common_paths()["projects_folder"])


def _experiment_configs_xlsx() -> Path:
    cfg = _common_paths()
    return Path(cfg["configurations_folder"]) / "experiment_configs.xlsx"


def _strict_mnemonic(raw: Any) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"mnemonic must be a string, got {type(raw).__name__}: {raw!r}")
    return Mnemonics.match(raw)


def _plans_by_mnemonic(xlsx: Path) -> dict[str, list[int]]:
    df = pd.read_excel(xlsx, sheet_name="plans", keep_default_na=False)
    out: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        m = str(row["mnemonic"]).strip().lower()
        pid = int(row["id"])
        out.setdefault(m, []).append(pid)
    for m in list(out.keys()):
        out[m] = sorted(set(out[m]))
    return out


def run_proj_to_analyze(
    run_analyze: bool = False,
    overwrite: bool = False,
    num_processes: int = 1,
    debug: bool = False,
) -> ProjToAnalyzeReport:
    report = ProjToAnalyzeReport()
    projects = sorted([p.name for p in _projects_folder().iterdir() if p.is_dir()])
    workbook_plans = _plans_by_mnemonic(_experiment_configs_xlsx())

    for project_title in projects:
        try:
            proj = Project(project_title=project_title)
        except Exception as e:
            msg = f"{project_title}: failed to load project: {e}"
            print(msg)
            report.errors.append(msg)
            continue

        try:
            mnemonic = _strict_mnemonic(proj.global_properties.get("mnemonic"))
        except Exception as e:
            msg = f"{project_title}: invalid mnemonic: {e}."
            print(msg)
            report.errors.append(msg)
            continue

        if not workbook_plans.get(mnemonic):
            msg = (
                f"{project_title}: no matching plan rows for mnemonic={mnemonic}."
            )
            print(msg)
            report.errors.append(msg)
            continue

        original_mnemonic = proj.global_properties.get("mnemonic")
        proj.global_properties["mnemonic"] = mnemonic
        try:
            cfg = ConfigMaker(proj)
        except Exception as e:
            msg = f"{project_title}/{mnemonic}: ConfigMaker init failed: {e}"
            print(msg)
            report.errors.append(msg)
            continue

        for plan_id in sorted(set(cfg.plans["plan_id"].tolist())):
            stage = "ConfigMaker.setup"
            try:
                cfg.setup(plan_id, verbose=False)
                if run_analyze:
                    stage = "analyze_resample.process_plan"
                    args = SimpleNamespace(
                        project_title=project_title,
                        plan=plan_id,
                        overwrite=overwrite,
                        num_processes=max(1, int(num_processes)),
                        debug=debug,
                    )
                    process_plan(args)
                report.runs.append(
                    PlanRun(
                        project=project_title,
                        mnemonic=mnemonic,
                        plan_id=int(plan_id),
                        ok=True,
                        stage=stage,
                    )
                )
            except Exception as e:
                report.runs.append(
                    PlanRun(
                        project=project_title,
                        mnemonic=mnemonic,
                        plan_id=int(plan_id),
                        ok=False,
                        stage=stage,
                        error=str(e),
                    )
                )
        proj.global_properties["mnemonic"] = original_mnemonic

    print(
        f"proj_to_analyze: plans_ok={report.n_ok} plans_failed={report.n_fail} "
        f"projects_skipped={len(report.skipped)}"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("proj_to_analyze")
    p.add_argument("--run-analyze", action="store_true", help="Run full analyze_resample process for each plan.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--debug", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    rep = run_proj_to_analyze(
        run_analyze=args.run_analyze,
        overwrite=args.overwrite,
        num_processes=args.num_processes,
        debug=args.debug,
    )
    return 1 if rep.n_fail > 0 or len(rep.errors) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
