from __future__ import annotations

import argparse
from pathlib import Path

from fran.managers.project import COMMON_PATHS, Project
from utilz.stringz import headline


def _project_titles() -> list[str]:
    projects_folder = Path(COMMON_PATHS["projects_folder"])
    if not projects_folder.exists():
        return []
    return sorted(path.name for path in projects_folder.iterdir() if path.is_dir())


def _delete_project(title: str) -> bool:
    headline(f"Deleting project {title}")
    project = Project(title)
    return bool(project.delete(interactive=False))


def main(args: argparse.Namespace) -> None:
    if args.all:
        titles = _project_titles()
    else:
        titles = args.titles

    if not titles:
        raise SystemExit("No project titles supplied")

    for title in titles:
        _delete_project(title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete FRAN project files non-interactively.")
    parser.add_argument(
        "titles",
        nargs="*",
        help="Project title(s) to delete.",
    )
    parser.add_argument(
        "-t",
        "--title",
        "--project-title",
        dest="title",
        action="append",
        default=[],
        help="Project title to delete. Can be repeated.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete every project folder in COMMON_PATHS['projects_folder'].",
    )
    args = parser.parse_args()
    args.titles = [*args.title, *args.titles]
    main(args)
