from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fran.data.dataregistry import DATASET_PATHS, DS
from fran.data.datasource import Datasource
from utilz.stringz import headline, info_from_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialise a datasource, optionally register it, then run datasource.process()."
    )
    parser.add_argument("folder", help="Datasource folder containing images/ and lms/")
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        default=1,
        help="Number of worker processes for datasource.process()",
    )
    return parser.parse_args()


def prompt_yes_no(question: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    response = input(question + suffix).strip().lower()
    if not response:
        return default
    return response in {"y", "yes"}


def prompt_text(
    question: str, default: str | None = None, allow_blank: bool = True
) -> str | None:
    suffix = f" [{default}]" if default not in (None, "") else ""
    response = input(f"{question}{suffix}: ").strip()
    if response:
        return response
    if default is not None:
        return default
    if allow_blank:
        return None
    print("A value is required.")
    return prompt_text(question, default=default, allow_blank=allow_blank)


def resolve_backup_root() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    codex_home = os.environ.get("CODEX_HOME")
    candidates = []
    if codex_home:
        candidates.append(Path(codex_home) / "agent_file_backups" / ts)
    candidates.append(Path("/tmp/agent_file_backups") / ts)
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            manifest = candidate / "manifest.tsv"
            manifest.write_text(
                "original_path\tbackup_path\toperation\ttimestamp\n",
                encoding="utf-8",
            )
            return candidate
        except OSError:
            continue
    raise RuntimeError(
        "Unable to create a backup directory for overwrite/delete operations."
    )


def backup_file(path: Path, operation: str, backup_root: Path) -> Path:
    backup_path = backup_root / path.name
    suffix_idx = 1
    while backup_path.exists():
        backup_path = backup_root / f"{path.stem}_{suffix_idx}{path.suffix}"
        suffix_idx += 1
    shutil.copy2(path, backup_path)
    manifest = backup_root / "manifest.tsv"
    timestamp = datetime.now().isoformat(timespec="seconds")
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(f"{path}\t{backup_path}\t{operation}\t{timestamp}\n")
    return backup_path


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"datasets": {}}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data.setdefault("datasets", {})
    return data


def save_registry(path: Path, data: dict[str, Any], backup_root: Path) -> None:
    if path.exists():
        backup_file(path, "overwrite", backup_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=False)


def find_registry_entry(folder: Path) -> tuple[str, Any] | None:
    for name in DS.names():
        spec = DS[name]
        if spec.folder.expanduser() == folder:
            return name, spec
    return None


def infer_dataset_name(folder: Path) -> str:
    images_dir = folder / "images"
    image_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    if not image_files:
        raise FileNotFoundError(f"No image files found in {images_dir}")
    try:
        parsed = info_from_filename(image_files[0].name)
        proj_title = parsed.get("proj_title")
        if proj_title:
            return proj_title
    except Exception:
        pass
    return folder.name


def build_registry_entry(
    folder: Path, dataset_name: str
) -> tuple[str, dict[str, Any], str | None]:
    key_default = dataset_name
    key = prompt_text("Registry key", default=key_default, allow_blank=False)
    ds_value = prompt_text("Value for ds", default=dataset_name, allow_blank=False)
    alias = prompt_text("Value for alias", allow_blank=True)
    ds_type = prompt_text("Value for ds_type", default="full", allow_blank=False)
    entry = {"ds": ds_value, "folder": str(folder)}
    entry["alias"] = alias
    if ds_type != "full":
        entry["ds_type"] = ds_type
    return key, entry, alias


def maybe_register_datasource(
    folder: Path,
    dataset_name: str,
    registry_paths: list[Path],
) -> str | None:
    existing = find_registry_entry(folder)
    if existing:
        reg_name, spec = existing
        print(
            f"Datasource already present in registry as '{reg_name}' "
            f"(folder={spec.folder}, alias={spec.alias}, ds_type={spec.ds_type})."
        )
        return spec.alias

    print("Datasource folder is not present in the current registry.")
    if not prompt_yes_no("Add it to one or more registry files?", default=True):
        return None

    key, entry, alias = build_registry_entry(folder, dataset_name)
    backup_root = resolve_backup_root()
    for path in registry_paths:
        if not prompt_yes_no(
            f"Write datasource entry to {path}?", default=path == registry_paths[0]
        ):
            continue
        registry = load_registry(path)
        datasets = registry.setdefault("datasets", {})
        if key in datasets:
            print(f"Registry key '{key}' already exists in {path}. Skipping this file.")
            continue
        datasets[key] = entry
        save_registry(path, registry, backup_root)
        print(f"Wrote datasource entry '{key}' to {path}")
    return alias


def maybe_overwrite_h5(h5_path: Path) -> None:
    if not h5_path.exists():
        return
    print(f"Existing datasource state found: {h5_path}")
    if not prompt_yes_no(
        "Overwrite existing h5 file and reinitialise datasource?", default=False
    ):
        print("Stopping without changes.")
        raise SystemExit(0)
    backup_root = resolve_backup_root()
    backup_path = backup_file(h5_path, "delete", backup_root)
    h5_path.unlink()
    print(f"Backed up existing h5 to {backup_path} and removed original.")


def validate_folder(folder: Path) -> None:
    if not folder.exists():
        raise FileNotFoundError(f"Datasource folder does not exist: {folder}")
    for child in ("images", "lms"):
        if not (folder / child).exists():
            raise FileNotFoundError(
                f"Expected datasource subfolder missing: {folder / child}"
            )


def main(args: argparse.Namespace) -> None:
    folder = Path(args.folder).expanduser().resolve()
    validate_folder(folder)
    dataset_name = infer_dataset_name(folder)
    bg_label = 0

    headline("Arguments")
    print(f"folder: {folder}")
    print(f"dataset_name: {dataset_name}")
    print(f"bg_label: {bg_label}")
    print(f"num_processes: {args.num_processes}")

    maybe_overwrite_h5(folder / "fg_voxels.h5")

    registry_paths = [
        Path(DATASET_PATHS).expanduser(),
        Path(Path(__file__).resolve().parents[2] / "datasets_hpc.yaml"),
    ]
    alias = maybe_register_datasource(folder, dataset_name, registry_paths)

    ds = Datasource(
        folder=folder,
        name=dataset_name,
        alias=alias,
        bg_label=bg_label,
    )
    multiprocess = args.num_processes > 1
    ds.process(num_processes=args.num_processes, multiprocess=multiprocess)


if __name__ == "__main__":
    import sys

    # sample: python /home/ub/code/fran/fran/run/datasource_init.py /path/to/datasource -n 8
    try:
        main(parse_args())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
