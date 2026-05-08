import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import ipdb
tr = ipdb.set_trace

import pandas as pd
from label_analysis.totalseg import TotalSegmenterLabels
from utilz.fileio import load_yaml

from fran.configs.mnemonics import Mnemonics
from fran.data.dataregistry import DS
from fran.inference.base import BaseInferer
from fran.inference.cascade import CascadeInferer, WholeImageInferer
from fran.inference.cascade_yolo import CascadeInfererYOLO
from fran.inference.helpers import load_params
from fran.run.inference.inference import resolve_inferer_cls
from fran.utils.common import COMMON_PATHS


BEST_RUNS_PATH = Path("/s/fran_storage/conf/best_runs.yaml")
RUNS_REGISTRY_NAME_COLS = ("run_name", "run_id", "model_id", "id")
MODE_ALIASES = {"kbd": "rbd"}
TSL_FAMILY_BY_MNEMONIC = {
    "kidneys": "kidney",
    "liver": "liver",
    "lungs": "lung",
    "pancreas": "pancreas",
    "colon": "colon",
}


@dataclass(frozen=True)
class InferenceSpec:
    inferer_cls: type
    run_name: str
    run_w: str | None = None
    localiser_labels: list[int] | None = None
    localiser_regions: list[str] | None = None
    k_largest: int | None = None


def load_best_runs(path: Path = BEST_RUNS_PATH) -> dict:
    return load_yaml(path)


def runs_registry_path() -> Path | None:
    if "COLD_STORAGE" not in os.environ:
        return None
    path = Path(os.environ["COLD_STORAGE"]) / "conf" / "runs_registry.csv"
    if not path.exists():
        return None
    return path


def load_runs_registry(path: Path | None = None) -> pd.DataFrame | None:
    path = runs_registry_path() if path is None else path
    if path is None:
        return None
    return pd.read_csv(path)


def registry_name_col(df: pd.DataFrame) -> str:
    for col in RUNS_REGISTRY_NAME_COLS:
        if col in df.columns:
            return col
    raise ValueError("runs_registry.csv missing run name column")


def local_run_params_available(run_name: str) -> bool:
    ckpt_root = Path(COMMON_PATHS["checkpoints_parent_folder"])
    matches = [fld for fld in ckpt_root.rglob(run_name) if fld.is_dir()]
    return len(matches) == 1


def parse_run_metadata_value(value):
    if pd.isna(value):
        return None
    return value


def run_metadata_from_registry(run_name: str) -> dict | None:
    df = load_runs_registry()
    if df is None:
        return None
    name_col = registry_name_col(df)
    rows = df[df[name_col] == run_name]
    if len(rows) == 0:
        return None
    row = rows.iloc[0].to_dict()
    return {key: parse_run_metadata_value(value) for key, value in row.items()}


def run_metadata_from_checkpoint(run_name: str) -> dict | None:
    if not local_run_params_available(run_name):
        return None
    configs = load_params(run_name)["configs"]
    plan = dict(configs["plan_train"])
    if "source_plan_run" in configs and "source_plan_run" not in plan:
        plan["source_plan_run"] = configs["source_plan_run"]
    return plan


def load_run_metadata(run_name: str) -> dict:
    metadata = run_metadata_from_registry(run_name)
    if metadata is not None:
        return metadata
    metadata = run_metadata_from_checkpoint(run_name)
    if metadata is not None:
        return metadata
    return {}


def canonical_mnemonic(raw: str, best_runs: dict) -> str:
    raw = raw.strip().lower()
    if raw in best_runs:
        return raw
    try:
        return Mnemonics.match(raw)
    except ValueError:
        for name, entry in best_runs.items():
            if not isinstance(entry, dict):
                continue
            aliases = entry["mnemonics"] if "mnemonics" in entry else []
            if raw in aliases:
                return name
        raise


def nonempty_runs(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [item for item in value if item]


def normalize_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    return MODE_ALIASES[mode] if mode in MODE_ALIASES else mode


def parse_regions(value) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(region).strip() for region in value if str(region).strip()]
    return [region for region in str(value).replace(" ", "").split(",") if region]


def remapping_family(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            family = remapping_family(item)
            if family is not None:
                return family
        return None
    if isinstance(value, dict):
        return None
    text = str(value).replace(" ", "")
    if "TSL." not in text or ":" not in text:
        return None
    return text.split(":")[1].replace("TSL.", "")


def family_from_metadata(metadata: dict) -> str | None:
    for key in (
        "remapping_lbd_kbd",
        "remapping_lbd_rbd",
        "remapping_lbd",
        "remapping_whole",
        "remapping_source",
    ):
        if key in metadata:
            family = remapping_family(metadata[key])
            if family is not None:
                return family
    return None


def tsl_region_labels(family: str) -> list[int]:
    tsl = TotalSegmenterLabels()
    if family == "label_region":
        return sorted(set(tsl.label_region))
    structure = getattr(tsl, family)
    return list(structure.label_region)


def resolve_tsl_localiser_labels(mnemonic: str, run_name: str) -> list[int]:
    metadata = load_run_metadata(run_name)
    family = family_from_metadata(metadata)
    if family is not None:
        return tsl_region_labels(family)
    if mnemonic == "nodes":
        return tsl_region_labels("label_region")
    if mnemonic not in TSL_FAMILY_BY_MNEMONIC:
        raise ValueError(f"No TSL label family mapping for mnemonic={mnemonic}")
    return tsl_region_labels(TSL_FAMILY_BY_MNEMONIC[mnemonic])


def choose_localiser_type(entry: dict, explicit: str | None, mnemonic: str) -> str | None:
    if explicit is not None:
        return explicit
    if mnemonic == "totalseg":
        return None
    if nonempty_runs(entry["TSL"] if "TSL" in entry else None):
        return "TSL"
    if nonempty_runs(entry["yolo"] if "yolo" in entry else None):
        return "yolo"
    return None


def resolve_input_folder(folder: str | None, dataset: str | None) -> Path:
    if (folder is None) == (dataset is None):
        raise ValueError("Pass exactly one of --folder or --dataset")
    if folder is not None:
        return Path(folder)
    return DS[dataset].folder / "images"


def resolve_yolo_regions(run_name: str) -> list[str]:
    metadata = load_run_metadata(run_name)
    if "mode" in metadata:
        mode = normalize_mode(metadata["mode"])
        if mode != "rbd":
            raise ValueError(f"Expected YOLO mode for {run_name}, found {metadata['mode']}")
    if "localiser_regions" not in metadata:
        raise ValueError(f"Missing localiser_regions for YOLO run {run_name}")
    return parse_regions(metadata["localiser_regions"])


def resolve_standalone_run(entry: dict) -> str:
    if "minimal" in entry and entry["minimal"]:
        return entry["minimal"][0]
    if "full" in entry and entry["full"]:
        return entry["full"][0]
    raise ValueError("No standalone run configured")


def resolve_standalone_inferer_cls(run_name: str):
    metadata = load_run_metadata(run_name)
    mode = normalize_mode(metadata["mode"]) if "mode" in metadata else None
    if mode == "source":
        return BaseInferer
    if mode == "whole":
        return WholeImageInferer
    inferer_cls, _ = resolve_inferer_cls(run_name)
    return inferer_cls


def resolve_spec(mnemonic_raw: str, localiser_type: str | None) -> InferenceSpec:
    best_runs = load_best_runs()
    mnemonic = canonical_mnemonic(mnemonic_raw, best_runs)
    entry = best_runs[mnemonic]
    localiser_type = choose_localiser_type(entry, localiser_type, mnemonic)

    if localiser_type == "yolo":
        run_name = nonempty_runs(entry["yolo"])[0]
        return InferenceSpec(
            inferer_cls=CascadeInfererYOLO,
            run_name=run_name,
            localiser_regions=resolve_yolo_regions(run_name),
            k_largest=entry["k_largest"] if "k_largest" in entry else None,
        )

    if localiser_type == "TSL":
        run_name = nonempty_runs(entry["TSL"])[0]
        return InferenceSpec(
            inferer_cls=CascadeInferer,
            run_name=run_name,
            run_w=best_runs["whole"],
            localiser_labels=resolve_tsl_localiser_labels(mnemonic, run_name),
            k_largest=entry["k_largest"] if "k_largest" in entry else None,
        )

    run_name = resolve_standalone_run(entry)
    inferer_cls = resolve_standalone_inferer_cls(run_name)
    return InferenceSpec(inferer_cls=inferer_cls, run_name=run_name)


def build_inferer(spec: InferenceSpec, gpus: list[int], patch_overlap: float):
    if spec.inferer_cls is CascadeInfererYOLO:
        return CascadeInfererYOLO(
            localiser_regions=spec.localiser_regions,
            run_p=spec.run_name,
            devices=gpus,
            patch_overlap=patch_overlap,
            save=True,
            save_channels=False,
            k_largest=spec.k_largest,
        )
    if spec.inferer_cls is CascadeInferer:
        return CascadeInferer(
            run_w=spec.run_w,
            run_p=spec.run_name,
            localiser_labels=spec.localiser_labels,
            devices=gpus,
            patch_overlap=patch_overlap,
            save=True,
            save_channels=False,
            save_localiser=False,
            k_largest=spec.k_largest,
        )
    if spec.inferer_cls in (BaseInferer, WholeImageInferer):
        return spec.inferer_cls(
            run_name=spec.run_name,
            devices=gpus,
            save=True,
            save_channels=False,
        )
    raise ValueError(f"Unsupported inferer class {spec.inferer_cls}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Mnemonic-driven inference runner")
    parser.add_argument("mnemonic")
    parser.add_argument("--localiser-type", choices=["yolo", "TSL"], default=None)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder")
    source.add_argument("--dataset", choices=sorted(DS.names()))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--patch-overlap", type=float, default=0.2)
    parser.add_argument("-o", "--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(args=None):
    args = parse_args(args) if isinstance(args, list) or args is None else args
    input_folder = resolve_input_folder(args.folder, args.dataset)
    spec = resolve_spec(args.mnemonic, args.localiser_type)
    inferer = build_inferer(spec, args.gpus, args.patch_overlap)
    print(
        {
            "mnemonic": args.mnemonic,
            "run_name": spec.run_name,
            "run_w": spec.run_w,
            "inferer": spec.inferer_cls.__name__,
            "input_folder": str(input_folder),
        }
    )
    inferer.run([input_folder], chunksize=args.chunksize, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
