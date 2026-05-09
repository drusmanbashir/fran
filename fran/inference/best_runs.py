from __future__ import annotations

import os
import shlex
from pathlib import Path

import pandas as pd
import torch
import wandb
from paramiko import SSHClient
from utilz.cprint import cprint
from utilz.fileio import load_yaml

from fran.callback.wandb.wandb import _resolve_run_by_name
from fran.configs.mnemonics import Mnemonics
from fran.inference.helpers import load_params
from fran.managers.wandb.wandb import download_path_no_wandb
from fran.managers.wandb.wandb import get_wandb_config
from fran.trainers.helpers import checkpoint_epoch
from fran.trainers.helpers import checkpoint_from_model_id
from fran.utils.common import COMMON_PATHS

BEST_RUNS_PATH = Path("/s/fran_storage/conf/best_runs.yaml")
COL_EXCEPTIONS = {
    "exact": {
        "tune",
        "quant",
        "manual_value",
        "plan_test",
        "tune_type",
        "vip_labels",
        "vip_label",
        "periodic_test",
        "periodict_test",
        "patch_dim0",
        "patch_dim1",
        "plan_valid",
    },
    "prefix": ("Unnamed",),
}


def load_best_runs(path: Path = BEST_RUNS_PATH) -> dict:
    return load_yaml(path)


def runs_registry_path() -> Path:
    root = Path(COMMON_PATHS["cold_storage_folder"])
    path = root / "conf" / "runs_registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    return path


def load_runs_registry(path: Path | None = None) -> pd.DataFrame:
    path = runs_registry_path() if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=["run_name"])
    df = pd.read_csv(path)
    if "run_name" not in df.columns:
        df["run_name"] = pd.Series(dtype="object")
    return df


def is_excluded_col(col: str) -> bool:
    if col in COL_EXCEPTIONS["exact"]:
        return True
    return any(col.startswith(prefix) for prefix in COL_EXCEPTIONS["prefix"])


def filter_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not is_excluded_col(str(key))}


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item]
    return [value]


def collect_run_names(best_runs: dict) -> list[str]:
    run_names = []
    for entry in best_runs.values():
        if not isinstance(entry, dict) or "runs" not in entry:
            continue
        runs = entry["runs"]
        if isinstance(runs, list):
            run_names.extend(item for item in runs if item)
            continue
        if isinstance(runs, dict):
            for value in runs.values():
                run_names.extend(_as_list(value))
    return list(dict.fromkeys(run_names))


def expand_wandb_config(payload: dict) -> dict:
    expanded = {}
    for key, value in payload.items():
        parts = str(key).split("/")
        cursor = expanded
        for part in parts[:-1]:
            if part not in cursor:
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return expanded


def resolve_entity() -> str:
    os.environ["WANDB_API_KEY"] = get_wandb_config()
    api = wandb.Api()
    return str(api.default_entity)


def wandb_project_paths(entity: str) -> list[str]:
    names = [mnemonic.wandb for mnemonic in Mnemonics._all]
    names = list(dict.fromkeys(str(name).lower() for name in names if name))
    return [f"{entity}/{name}" for name in names]


def resolve_run(api: wandb.Api, run_name: str):
    entity = resolve_entity()
    for project_name in wandb_project_paths(entity):
        try:
            return _resolve_run_by_name(api=api, project_name=project_name, run_name=run_name)
        except FileNotFoundError:
            continue
        except ValueError as exc:
            if "Could not find project" in str(exc):
                continue
            raise
    raise FileNotFoundError(f"Run '{run_name}' not found in mnemonic W&B projects")


def local_run_params_available(run_name: str) -> bool:
    ckpt_root = Path(COMMON_PATHS["checkpoints_parent_folder"])
    matches = [fld for fld in ckpt_root.rglob(run_name) if fld.is_dir()]
    return len(matches) == 1


def row_from_checkpoint(run_name: str) -> dict | None:
    if not local_run_params_available(run_name):
        return None
    configs = load_params(run_name)["configs"]
    row = {"run_name": run_name}
    if "plan_train" in configs and isinstance(configs["plan_train"], dict):
        row.update(
            {
                key: value
                for key, value in configs["plan_train"].items()
                if key != "run_name"
            }
        )
    if "source_plan_run" in configs and "source_plan_run" not in row:
        row["source_plan_run"] = configs["source_plan_run"]
    ckpt = checkpoint_from_model_id(run_name, normalize_keys=False)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    row["last_epoch"] = checkpoint_epoch(ckpt)
    row["last_lr"] = None
    for scheduler in state.get("lr_schedulers", []):
        if isinstance(scheduler, dict) and scheduler.get("_last_lr"):
            row["last_lr"] = scheduler["_last_lr"][0]
            break
        if isinstance(scheduler, dict) and scheduler.get("last_lr"):
            row["last_lr"] = scheduler["last_lr"][0]
            break
    row["train0_loss"] = None
    row["val0_loss"] = None
    return filter_row(row)


def remote_checkpoints_dir(run_name: str) -> str | None:
    config_hpc = load_yaml(Path(os.environ["FRAN_CONF"]) / "config_hpc.yaml")
    if "checkpoints_parent_folder" in config_hpc:
        remote_root = Path(config_hpc["checkpoints_parent_folder"])
    else:
        remote_root = Path(config_hpc["cold_storage_folder"]) / "checkpoints"
    hpc_settings = load_yaml(os.environ["HPC_SETTINGS"])
    client = SSHClient()
    client.load_system_host_keys()
    client.connect(
        hpc_settings["host"],
        username=hpc_settings["username"],
        password=hpc_settings["password"],
    )
    cmd = (
        f"find {shlex.quote(str(remote_root))} -type d "
        f"-path {shlex.quote(f'*/{run_name}/checkpoints')}"
    )
    try:
        _, stdout, _ = client.exec_command(cmd)
        matches = [line.strip() for line in stdout.readlines() if line.strip()]
    finally:
        client.close()
    if len(matches) == 0:
        return None
    return matches[0]


def resolve_checkpoint_row(run_name: str) -> dict | None:
    row = row_from_checkpoint(run_name)
    if row is not None:
        return row
    remote_dir = remote_checkpoints_dir(run_name)
    if remote_dir is None:
        return None
    config_hpc = load_yaml(Path(os.environ["FRAN_CONF"]) / "config_hpc.yaml")
    remote_cold = str(config_hpc["cold_storage_folder"]).rstrip("/")
    local_cold = str(COMMON_PATHS["cold_storage_folder"]).rstrip("/")
    if not remote_dir.startswith(remote_cold):
        raise ValueError(f"Cannot map remote checkpoint path to local cold storage: {remote_dir}")
    local_dir = Path(local_cold + remote_dir[len(remote_cold) :])
    download_path_no_wandb(remote_dir, local_dir)
    return row_from_checkpoint(run_name)


def history_tail_values(run, keys: list[str]) -> dict:
    values = {key: None for key in keys}
    for row in run.scan_history(keys=keys):
        for key in keys:
            if key in row and pd.notna(row[key]):
                values[key] = row[key]
    return values


def row_from_run(run_name: str, run) -> dict:
    plan_prefix = "configs/datamodule/plan_train/"
    config_payload = dict(run.config)
    raw_payload = dict(run.rawconfig)
    payload = expand_wandb_config(config_payload)
    row = {"run_name": run_name}
    if "plan_train" in payload and isinstance(payload["plan_train"], dict):
        row.update(payload["plan_train"])
    for source in (config_payload, raw_payload):
        for key, value in source.items():
            if str(key).startswith(plan_prefix):
                suffix = str(key).removeprefix(plan_prefix)
                if suffix == "run_name":
                    continue
                row[suffix] = value
    if "plan_train" not in payload and len(raw_payload) > 0:
        raw_nested = expand_wandb_config(raw_payload)
        if "plan_train" in raw_nested and isinstance(raw_nested["plan_train"], dict):
            row.update(raw_nested["plan_train"])
    row["last_epoch"] = run.summary.get("epoch")
    row["last_lr"] = run.summary.get("lr-Adam")
    if pd.isna(row["last_epoch"]) or pd.isna(row["last_lr"]):
        tail = history_tail_values(run, ["epoch", "lr-Adam"])
        if pd.isna(row["last_epoch"]):
            row["last_epoch"] = tail["epoch"]
        if pd.isna(row["last_lr"]):
            row["last_lr"] = tail["lr-Adam"]
    row.update(history_tail_values(run, ["train0_loss", "val0_loss"]))
    return filter_row(row)


def update_runs_registry(
    best_runs_path: Path = BEST_RUNS_PATH,
    registry_csv: Path | None = None,
) -> pd.DataFrame:
    best_runs = load_best_runs(best_runs_path)
    collected = collect_run_names(best_runs)
    collected_set = set(collected)

    registry_csv = runs_registry_path() if registry_csv is None else registry_csv
    df = load_runs_registry(registry_csv)
    df = df[df["run_name"].isin(collected_set)].reset_index(drop=True)

    existing = set(df["run_name"].tolist()) if len(df) else set()
    pending = [run_name for run_name in collected if run_name not in existing]

    if pending:
        os.environ["WANDB_API_KEY"] = get_wandb_config()
        api = wandb.Api()
        rows = []
        unresolved = []
        for run_name in pending:
            try:
                run = resolve_run(api, run_name)
            except FileNotFoundError:
                row = resolve_checkpoint_row(run_name)
                if row is None:
                    print(f"Skipping unresolved run: {run_name}")
                    unresolved.append(run_name)
                    continue
                rows.append(row)
                continue
            rows.append(row_from_run(run_name, run))
        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True, sort=False)
        if unresolved:
            cprint(
                "Unresolved runs remain in best_runs; remove these lines from best_runs:",
                color="yellow",
                bold=True,
            )
            for run_name in unresolved:
                cprint(run_name, color="yellow", bold=True)

    if "run_name" not in df.columns:
        df["run_name"] = pd.Series(dtype="object")
    df = df[[col for col in df.columns if not is_excluded_col(str(col))]]
    df.to_csv(registry_csv, index=False)
    return df


if __name__ == "__main__":
    df = update_runs_registry()
    print(df)
