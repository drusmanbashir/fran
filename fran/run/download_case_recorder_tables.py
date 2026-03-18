# Created by Codex
from __future__ import annotations

import argparse
import re
from pathlib import Path

import wandb
from fran.managers.wandb import get_wandb_config

TABLE_PATTERN = re.compile(
    r"^media/table/case_recorder/(?P<stage>[^/]+)/df_epoch_(?P<epoch>\d+)_.*\.table\.(?P<ext>csv|json)$"
)


def build_parser() -> argparse.ArgumentParser:
    return parser


def resolve_entity() -> str:
    entity, _api_token = get_wandb_config()
    return entity


def iter_case_recorder_files(run, stages: set[str], epochs: set[int] | None):
    for remote_file in run.files():
        match = TABLE_PATTERN.match(remote_file.name)
        if match is None:
            continue
        stage = match.group("stage")
        epoch = int(match.group("epoch"))
        if stage not in stages:
            continue
        if epochs is not None and epoch not in epochs:
            continue
        yield remote_file, stage, epoch


def destination_path(out_root: Path, run, stage: str, epoch: int) -> Path:
    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run.name))
    ext = "csv"
    return out_root / run.project / safe_run_name / stage / f"df_epoch_{epoch}.{ext}"


def main(args) -> None:

    entity = resolve_entity()
    api = wandb.Api()
    run = resolve_run(
        api=api, entity=entity, project=args.project, run_name=args.run_name
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    epochs = None if args.epoch is None else set(args.epoch)
    stages = set(args.stage)

    matches = list(iter_case_recorder_files(run, stages=stages, epochs=epochs))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No case_recorder tables found for run {run.path} with the requested filters."
        )

    print(f"Resolved run: {run.path}")
    print(f"Downloading {len(matches)} files to {out_root.resolve()}")

    for remote_file, stage, epoch, ext in sorted(
        matches, key=lambda item: (item[1], item[2], item[3])
    ):
        dest = destination_path(
            out_root=out_root, run=run, stage=stage, epoch=epoch, ext=ext
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not args.overwrite:
            print(f"Skipping existing {dest}")
            continue
        remote_file.download(
            root=str(dest.parent), replace=args.overwrite, exist_ok=True
        )
        downloaded = dest.parent / Path(remote_file.name).name
        if downloaded != dest:
            downloaded.rename(dest)
        print(f"{stage} epoch {epoch}: {dest}")


if __name__ == "__main__":
# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
    from fran.managers.project import Project

    from managers.wandb import WandbManager

# %%
    parser = argparse.ArgumentParser(
        description="Download W&B case_recorder dataframe tables for a run."
    )
    parser.add_argument(
        "--run-name",
        help="W&B run id or run name, for example KITS2-bah.",
    )
    parser.add_argument(
        "--project",
        help="W&B project name, for example kits2.",
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["train", "valid", "train2"],
        choices=["train", "valid", "train2"],
        help="Stages to download.",
    )
    parser.add_argument(
        "--epoch",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of epochs to keep. Default downloads all logged epochs.",
    )
    parser.add_argument(
        "--out",
        default="wandb_tables",
        help="Destination root folder.",
    )

# %%
    args = parser.parse_known_args()[0]

    args.run_name = "KITS2-BAH"
    args.project = "kits2"
# %%
    main(args)
# %%

    proj = Project(args.project)
    man = WandbManager(project=proj, run_id = args.run_name)
    api = wandb.Api()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    epochs = None if args.epoch is None else set(args.epoch)
    stages = set(args.stage)

    matches = list(iter_case_recorder_files(run, stages=stages, epochs=epochs))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No case_recorder tables found for run {run.path} with the requested filters."
        )

    print(f"Resolved run: {run.path}")
    print(f"Downloading {len(matches)} files to {out_root.resolve()}")

    for remote_file, stage, epoch, ext in sorted(
        matches, key=lambda item: (item[1], item[2], item[3])
    ):
        dest = destination_path(
            out_root=out_root, run=run, stage=stage, epoch=epoch, ext=ext
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not args.overwrite:
            print(f"Skipping existing {dest}")
            continue
        remote_file.download(
            root=str(dest.parent), replace=args.overwrite, exist_ok=True
        )
        downloaded = dest.parent / Path(remote_file.name).name
        if downloaded != dest:
            downloaded.rename(dest)
        print(f"{stage} epoch {epoch}: {dest}")


# %%
