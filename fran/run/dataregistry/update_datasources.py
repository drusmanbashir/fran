from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from fran.data.datasource import Datasource


def datasets_conf_path() -> Path:
    return Path(os.environ["FRAN_CONF"]) / "datasets.yaml"


def load_datasets() -> dict:
    data = yaml.safe_load(datasets_conf_path().read_text()) or {}
    return data["datasets"] if "datasets" in data else data


def resolve_dataset(name: str, entry: dict) -> tuple[str, Path]:
    ds_name = entry["ds"] if "ds" in entry else name
    folder_raw = entry["folder"] if "folder" in entry else entry["local_folder"]
    folder = Path(os.path.expandvars(os.path.expanduser(folder_raw))).resolve()
    return ds_name, folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="update_datasources",
        description="Initialize or update datasource fg_voxels.h5 files from $FRAN_CONF/datasets.yaml.",
    )
    parser.add_argument("dataset_names", nargs="*", help="Optional dataset keys from datasets.yaml.")
    parser.add_argument("-n", "--num-processes", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--return-voxels", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    datasets = load_datasets()
    requested = args.dataset_names if args.dataset_names else sorted(datasets.keys())

    summaries = []
    for dataset_key in requested:
        entry = datasets[dataset_key]
        ds_name, folder = resolve_dataset(dataset_key, entry)
        ds = Datasource(folder=folder, name=ds_name)
        h5_exists_before = ds.h5_fname.exists()
        summary = ds.update_datasource(
            return_voxels=args.return_voxels,
            num_processes=args.num_processes,
            multiprocess=args.num_processes > 1,
            dry_run=args.dry_run,
        )
        summary["dataset_key"] = dataset_key
        summary["dataset_name"] = ds_name
        summary["folder"] = str(folder)
        summary["h5_exists_before"] = h5_exists_before
        summary["action"] = "update" if h5_exists_before else "init"
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    print(json.dumps({"processed": len(summaries), "datasets": [s["dataset_key"] for s in summaries]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
