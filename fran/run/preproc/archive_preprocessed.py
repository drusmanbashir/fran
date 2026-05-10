from __future__ import annotations

import argparse
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fran.managers.project import Project
from fran.run.misc.resolve_plan_folder import resolve_plan_folders
from fran.utils.common import COMMON_PATHS
from tqdm.auto import tqdm

RAPID_ACCESS_ROOT = Path(COMMON_PATHS["rapid_access_folder"])
ARCHIVE_ROOT = Path(COMMON_PATHS["cold_storage_folder"]) / "archived"
ARCHIVABLE_PLAN_KEYS = (
    "data_folder_lbd",
    "data_folder_whole",
    "data_folder_pbd",
    "data_folder_rbd",
)
DEFAULT_COPY_WORKERS = min(32, max(1, os.cpu_count() or 1))


def _relative_from_root(folder: str | Path, root: Path) -> Path:
    folder_path = Path(folder).expanduser().resolve(strict=False)
    root_path = root.resolve(strict=False)
    try:
        relative = folder_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(
            f"Folder must live under {root_path}: {folder_path}"
        ) from exc
    if len(relative.parts) != 3:
        raise ValueError(
            f"Folder must be a direct preprocessed leaf under "
            f"{root_path}/<project>/<kind>/<folder>: {folder_path}"
        )
    return relative


def _folder_to_relative(folder: str | Path) -> Path:
    return _relative_from_root(folder, RAPID_ACCESS_ROOT)


def _archive_folder_to_relative(folder: str | Path) -> Path:
    return _relative_from_root(folder, ARCHIVE_ROOT)


def archive_path_for_rapid_folder(folder: str | Path) -> Path:
    return ARCHIVE_ROOT / _folder_to_relative(folder)


def rapid_path_for_archive_folder(folder: str | Path) -> Path:
    return RAPID_ACCESS_ROOT / _archive_folder_to_relative(folder)


def rapid_folder_missing_or_empty(folder: str | Path) -> bool:
    folder_path = Path(folder)
    if not folder_path.exists():
        return True
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Rapid-access source is not a folder: {folder_path}")
    return next(folder_path.iterdir(), None) is None


def _copy_one_file(job: tuple[str, str]) -> int:
    source_name, destination_name = job
    shutil.copy2(source_name, destination_name)
    return 1


def _copytree_via_temp(source: Path, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.parent / (
        f"{destination.name}.tmp-{uuid.uuid4().hex[:8]}"
    )
    try:
        temp_destination.mkdir()
        copy_jobs = []
        for path in source.rglob("*"):
            destination_path = temp_destination / path.relative_to(source)
            if path.is_dir():
                destination_path.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                copy_jobs.append((str(path), str(destination_path)))

        workers = min(DEFAULT_COPY_WORKERS, max(1, len(copy_jobs)))
        print(f"copying={source} -> {temp_destination}")
        print(f"files={len(copy_jobs)} workers={workers}")

        if copy_jobs:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_copy_one_file, job) for job in copy_jobs]
                with tqdm(total=len(futures), desc=source.name, unit="file") as progress:
                    for future in as_completed(futures):
                        future.result()
                        progress.update(1)
        temp_destination.rename(destination)
    except Exception:
        print(f"copy_failed={source}")
        if temp_destination.exists():
            shutil.rmtree(temp_destination)
        raise
    return destination


def archive_folder(relative: Path) -> int:
    source = RAPID_ACCESS_ROOT / relative
    target = ARCHIVE_ROOT / relative
    if not source.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source is not a folder: {source}")
    print("action=archive")
    print(f"relative={relative}")
    print(f"source={source}")
    print(f"target={target}")
    _copytree_via_temp(source, target)
    print(f"removing_source={source}")
    shutil.rmtree(source)
    print(f"archived_to={target}")
    return 0


def unarchive_folder(relative: Path, remove_archive: bool = False) -> int:
    source = ARCHIVE_ROOT / relative
    target = RAPID_ACCESS_ROOT / relative
    if not source.exists():
        raise FileNotFoundError(f"Archive folder does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Archive source is not a folder: {source}")
    print("action=unarchive")
    print(f"relative={relative}")
    print(f"source={source}")
    print(f"target={target}")
    _copytree_via_temp(source, target)
    if remove_archive:
        print(f"removing_archive={source}")
        shutil.rmtree(source)
    else:
        print(f"archive_retained={source}")
    print(f"restored_to={target}")
    return 0


def ensure_rapid_data_folder(folder: str | Path) -> Path:
    rapid_folder = Path(folder)
    if not rapid_folder_missing_or_empty(rapid_folder):
        return rapid_folder
    archive_folder = archive_path_for_rapid_folder(rapid_folder)
    if not archive_folder.exists():
        return rapid_folder
    if rapid_folder.exists():
        rapid_folder.rmdir()
    relative = _folder_to_relative(rapid_folder)
    unarchive_folder(relative)
    return rapid_folder


def _project_relatives(project_title: str, action: str) -> list[Path]:
    project = Project(project_title=project_title)
    rapid_roots = (
        Path(project.lbd_folder),
        Path(project.rbd_folder),
        Path(project.pbd_folder),
        Path(project.whole_images_folder),
    )
    if action == "archive":
        roots = rapid_roots
        relative_fn = _folder_to_relative
    else:
        roots = [ARCHIVE_ROOT / project_title / root.name for root in rapid_roots]
        relative_fn = _archive_folder_to_relative

    relatives = []
    for root in roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir():
                relatives.append(relative_fn(child))
    return relatives


def _resolve_relatives_from_args(args: argparse.Namespace) -> list[Path]:
    has_folder = args.folder is not None
    has_project = args.project_title is not None
    has_plan_num = args.plan_num is not None
    has_key = args.key is not None

    if has_folder:
        if has_project or has_plan_num or has_key:
            raise SystemExit(
                "Choose either --folder or --project-title/--plan-num/--key."
            )
        return [_folder_to_relative(args.folder)]

    if not has_project:
        raise SystemExit("Provide --folder or --project-title.")

    if has_plan_num:
        if not has_key:
            raise SystemExit("--key is required when --plan-num is provided.")
        folder = resolve_plan_folders(args.project_title, args.plan_num)[args.key]
        return [_folder_to_relative(folder)]

    if has_key:
        raise SystemExit("--key requires --plan-num.")

    return _project_relatives(args.project_title, args.action)


def main(args: argparse.Namespace) -> int:
    relatives = _resolve_relatives_from_args(args)
    if args.action == "archive":
        for relative in relatives:
            archive_folder(relative)
        return 0
    for relative in relatives:
        unarchive_folder(relative, remove_archive=args.remove_archive)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Archive or restore FRAN preprocessed folders between rapid-access "
            "storage and cold storage."
        )
    )
    parser.add_argument("action", choices=["archive", "unarchive"])
    parser.add_argument(
        "--folder",
        help=(
            "Direct rapid-access folder to archive or restore, for example "
            "<rapid_access>/<project>/<kind>/<folder>."
        ),
    )
    parser.add_argument(
        "--project-title",
        help=(
            "FRAN project title. Without --plan-num, operate on every immediate "
            "child folder under the project's plan roots."
        ),
    )
    parser.add_argument("--plan-num", type=int, help="FRAN plan id")
    parser.add_argument("--key", choices=ARCHIVABLE_PLAN_KEYS)
    parser.add_argument(
        "--remove-archive",
        action="store_true",
        help="Delete the archive copy after a successful unarchive.",
    )
    args = parser.parse_known_args()[0]
    raise SystemExit(main(args))
