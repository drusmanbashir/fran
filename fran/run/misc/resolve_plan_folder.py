from __future__ import annotations

import argparse
import json

from fran.configs.parser import ConfigMaker
from fran.managers import Project
from fran.utils.folder_names import FolderNames


def resolve_plan_folders(project_title: str, plan_num: int) -> dict[str, str]:
    project = Project(project_title=project_title)
    cfg = ConfigMaker(project)
    cfg.setup(plan_num, plan_num, verbose=False)
    plan = cfg.configs["plan_train"]
    folders = FolderNames(project, plan).folders
    return {key: str(value) for key, value in folders.items()}


def main(args: argparse.Namespace) -> int:
    folders = resolve_plan_folders(args.project_title, args.plan_num)
    if args.key is not None:
        print(folders[args.key])
        return 0
    print(json.dumps(folders, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resolve FRAN plan folder paths for a project and plan id."
    )
    parser.add_argument("project_title", help="FRAN project title")
    parser.add_argument("plan_num", type=int, help="FRAN plan id")
    parser.add_argument(
        "--key",
        choices=[
            "data_folder_source",
            "data_folder_lbd",
            "data_folder_whole",
            "data_folder_pbd",
            "data_folder_sourcepbd",
            "data_folder_rbd",
        ],
        help="Optional single folder key to print instead of the full mapping.",
    )
    raise SystemExit(main(parser.parse_args()))
