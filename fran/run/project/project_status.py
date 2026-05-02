from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import pandas as pd

from fran.cpp.helpers import load_project_cfg
from fran.managers.project import COMMON_PATHS
from fran.utils.folder_names import FolderNames


def projects_root() -> Path:
    return Path(COMMON_PATHS["projects_folder"])


def should_ignore_project_dir(name: str) -> bool:
    lower_name = name.lower()
    return "test" in lower_name or len(name) < 3


def project_directories() -> list[Path]:
    root = projects_root()
    paths = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and not should_ignore_project_dir(path.name):
            paths.append(path)
    return paths


def project_names(print_stdout: bool = True) -> list[str]:
    names = []
    for path in project_directories():
        names.append(path.name)
    if print_stdout:
        for name in names:
            print(name)
    return names


def cases_in_folder(folder: str | Path) -> int:
    folder = Path(folder)
    return len(list((folder / "images").glob("*")))


def confirm_plan_status(project, plan) -> dict[str, bool]:
    n_cases = len(project)
    folders = FolderNames(project, plan).folders
    cases_in_src_folder = cases_in_folder(folders["data_folder_source"])
    src_fldr_full = n_cases == cases_in_src_folder

    mode = plan["mode"]
    if mode == "lbd":
        final_folder = folders["data_folder_lbd"]
    elif mode == "rbd":
        final_folder = folders["data_folder_rbd"]
    elif mode in ["patch", "pbd"]:
        final_folder = folders["data_folder_pbd"]
    elif mode == "whole":
        final_folder = folders["data_folder_whole"]
    elif mode in ["source", "sourcepbd"]:
        final_folder = folders["data_folder_source"]
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")

    cases_in_final_folder = cases_in_folder(final_folder)
    final_fldr_full = n_cases == cases_in_final_folder
    return {"src_fldr_full": src_fldr_full, "final_fldr_full": final_fldr_full}


def add_preprocess_status(cfg) -> None:
    preprocess = []

    for plan_id in cfg.plans.index:
        cfg.setup(plan_id, verbose=False)
        plan = cfg.configs["plan_train"]
        detailed_status = confirm_plan_status(cfg.project, plan)

        src_fldr_full = detailed_status["src_fldr_full"]
        final_fldr_full = detailed_status["final_fldr_full"]

        if src_fldr_full and final_fldr_full:
            status = "both"
        elif src_fldr_full or final_fldr_full:
            status = "one"
        else:
            status = "none"

        preprocess.append(status)

    cfg.plans["preprocessed"] = preprocess


def load_project_cfg_with_preprocess_status(project_name: str):
    proj, cfg = load_project_cfg(project_name)
    add_preprocess_status(cfg)
    return proj, cfg


def load_project_status(project_name: str) -> dict[str, object]:
    proj, cfg = load_project_cfg_with_preprocess_status(project_name)

    plan_rows = []
    for plan_id in cfg.plans.index:
        cfg.setup(plan_id, verbose=False)
        plan = cfg.configs["plan_train"]
        detailed_status = confirm_plan_status(cfg.project, plan)
        plan_rows.append(
            {
                "plan_id": int(plan_id),
                "status_source": "present"
                if detailed_status["src_fldr_full"]
                else "missing",
                "status_plan_ds": "present"
                if detailed_status["final_fldr_full"]
                else "missing",
            }
        )

    return {
        "project": proj.project_title,
        "num_cases": len(proj),
        "plans": plan_rows,
    }


def collect_project_statuses(
    names: list[str],
    num_processes: int | None = None,
) -> list[dict[str, object]]:
    if num_processes is None:
        num_processes = min(len(names), mp.cpu_count())
    num_processes = max(1, min(num_processes, len(names)))

    if num_processes == 1:
        results = []
        for name in names:
            results.append(load_project_status(name))
        return results

    with mp.Pool(processes=num_processes) as pool:
        return list(pool.imap(load_project_status, names))


def project_status_df(status: dict[str, object]) -> pd.DataFrame:
    rows = sorted(status["plans"], key=lambda row: row["plan_id"])
    return pd.DataFrame(rows, columns=["plan_id", "status_source", "status_plan_ds"])


def print_project_statuses(results: list[dict[str, object]]) -> None:
    for result in results:
        print(f"\n=== {result['project']} ===")
        print(f"num_cases: {result['num_cases']}")
        df = project_status_df(result)
        print(df.to_string(index=False))


def main(args: argparse.Namespace) -> list[dict[str, object]]:
    names = project_names(print_stdout=True)
    selected_names = args.projects or names

    print("all project statuses:")
    results = collect_project_statuses(selected_names, num_processes=args.num_processes)
    print_project_statuses(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List FRAN projects and print preprocessing status for each project."
    )
    parser.add_argument(
        "projects",
        nargs="*",
        help="Optional project names. Defaults to every non-ignored project folder.",
    )
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        default=None,
        help="Number of worker processes to use while loading project status.",
    )
    args = parser.parse_known_args()[0]
    # args.num_processes = 1
    main(args)
# %%
    project_name="kits23"

    proj, cfg = load_project_cfg_with_preprocess_status(project_name)
    add_preprocess_status(cfg)
# %%
    for plan_id, status in cfg.plans["preprocessed"].items():
        print(f"plan {plan_id}: {status}")
# %%
