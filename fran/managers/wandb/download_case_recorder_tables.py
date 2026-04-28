# Created by Codex
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import wandb
from fran.callback.wandb.wandb import _resolve_run_by_name
from fran.configs.mnemonics import Mnemonics
from fran.managers.project import Project
from fran.managers.wandb.wandb import get_wandb_config
from tqdm.auto import tqdm

TABLE_PATTERN = re.compile(
    r"^media/table/case_recorder/(?P<stage>[^/]+)/df_epoch_(?P<epoch>\d+)_.*\.table\.(?P<ext>csv|json)$"
)


def resolve_entity() -> str:
    api_token = get_wandb_config()
    os.environ["WANDB_API_KEY"] = api_token
    api = wandb.Api()
    return str(api.default_entity)


def resolve_wandb_project(project_title: str) -> str:
    return str(Mnemonics()[project_title].wandb)


def resolve_run(api: wandb.Api, project_title: str, run_name: str):
    entity = resolve_entity()
    wandb_project = resolve_wandb_project(project_title)
    project_name = f"{entity}/{wandb_project}"
    run = _resolve_run_by_name(api=api, project_name=project_name, run_name=run_name)
    return run


def resolve_out_root(project_title: str, run_name: str) -> Path:
    project = Project(project_title)
    out_root = Path(project.log_folder) / str(run_name)
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def iter_case_recorder_files(run, stages: set[str], epochs: set[int] | None):
    for remote_file in run.files():
        match = TABLE_PATTERN.match(remote_file.name)
        if match is None:
            continue
        stage = match.group("stage")
        epoch = int(match.group("epoch"))
        ext = match.group("ext")
        if stage not in stages:
            continue
        if epochs is not None and epoch not in epochs:
            continue
        yield remote_file, stage, epoch, ext


def destination_path(out_root: Path, stage: str, epoch: int, ext: str) -> Path:
    return out_root / stage / f"df_epoch_{epoch}.{ext}"


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


def download_run_config(
    run,
    out_root: str | Path,
    filename: str = "run_config.json",
    overwrite: bool = False,
) -> Path:
    out_root = Path(out_root)
    dest = out_root / filename
    if dest.exists() and not overwrite:
        print(f"Skipping existing {dest}")
        return dest
    run.load(force=True)
    run.load_full_data(force=True)
    payload = dict(run.config)
    raw_payload = dict(run.rawconfig)
    if len(payload) == 0:
        payload = raw_payload
    nested_payload = expand_wandb_config(payload)
    dest.write_text(json.dumps(nested_payload, indent=2, sort_keys=True) + "\n")
    print(f"Run config saved: {dest}")
    return dest


def compile_case_recorder_tables(run_folder: str | Path) -> pd.DataFrame:
    run_root = Path(run_folder)
    rows = []
    stage_dirs = []
    for stage_dir in sorted(run_root.iterdir()):
        if stage_dir.is_dir():
            stage_dirs.append(stage_dir)
    for stage_dir in stage_dirs:
        folder_name = stage_dir.name
        stage_value = folder_name
        json_files = sorted(stage_dir.glob("df_epoch_*.json"))
        for json_file in json_files:
            epoch = int(json_file.stem.split("_")[2])
            table = json.loads(json_file.read_text())
            columns = table["columns"]
            data = table["data"]
            frame = pd.DataFrame(data, columns=columns)
            for row in frame.itertuples(index=False):
                row_out = {
                    "stage": stage_value,
                    "epoch": epoch,
                    "case_id": row.case_id,
                    "label": row.label,
                    "loss_dice": row.loss_dice,
                }
                rows.append(row_out)
    compiled = pd.DataFrame(rows)
    output_file = run_root / "case_recorder_compiled.csv"
    compiled.to_csv(output_file, index=False)
    print(f"Compiled table saved: {output_file}")
    return compiled


def download_case_recorder_tables(
    run,
    out_root: str | Path,
    stages: set[str],
    epochs: set[int] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    out_root = Path(out_root)
    matches = list(iter_case_recorder_files(run, stages=stages, epochs=epochs))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No case_recorder tables found for run {run.path} with the requested filters."
        )

    print(f"Downloading {len(matches)} files")

    sorted_matches = sorted(matches, key=lambda item: (item[1], item[2], item[3]))
    for remote_file, stage, epoch, ext in tqdm(
        sorted_matches, total=len(sorted_matches), desc="case_recorder tables", unit="file"
    ):
        dest = destination_path(out_root=out_root, stage=stage, epoch=epoch, ext=ext)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not overwrite:
            print(f"Skipping existing {dest}")
            continue
        downloaded_obj = remote_file.download(
            root=str(dest.parent), replace=overwrite, exist_ok=True
        )
        downloaded = Path(downloaded_obj.name)
        if not downloaded.is_absolute():
            downloaded = Path.cwd() / downloaded
        if downloaded != dest:
            downloaded.rename(dest)
        print(f"{stage} epoch {epoch}: {dest}")
    compiled = compile_case_recorder_tables(out_root)
    return compiled


def main(args) -> None:
    api = wandb.Api()
    run = resolve_run(api=api, project_title=args.project, run_name=args.run_name)
    overwrite = False
    out_root = resolve_out_root(project_title=args.project, run_name=args.run_name)

    download_root = out_root.resolve()
    print(f"Resolved run: {run.path}")
    print(f"Download folder name: {download_root.name}")
    print(f"Download folder path: {download_root}")
    if args.download in ["config", "both"]:
        download_run_config(run=run, out_root=out_root, overwrite=overwrite)
    if args.download in ["tables", "both"]:
        epochs = None if args.epoch is None else set(args.epoch)
        stages = set(args.stage)
        download_case_recorder_tables(
            run=run,
            out_root=out_root,
            stages=stages,
            epochs=epochs,
            overwrite=overwrite,
        )

# %%
if __name__ == "__main__":
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Download W&B case_recorder dataframe tables for a run."
    )
    parser.add_argument(
        "--run-name",
        help="W&B run id or run name, for example KITS2-bah.",
    )
    parser.add_argument(
        "--project",
        help="Project title mnemonic, for example kits23.",
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["train", "valid", "train2"],
        choices=["train", "valid", "train2"],
        help="Stages to download for table downloads.",
    )
    parser.add_argument(
        "--epoch",
        nargs="+",
        type=int,
        default=None,
        help="Optional epochs for table downloads. Default downloads all logged epochs.",
    )
    parser.add_argument(
        "--download",
        default="tables",
        choices=["tables", "config", "both"],
        help="What to download for the run.",
    )
# %%
    args = parser.parse_known_args()[0]

# %%
    args.project = "kits23"
    args.run_name = "KITS23-SIRIG"
    args.download = "config"
# %%
    main(args)
# %%
