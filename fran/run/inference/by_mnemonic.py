import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

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
LOCALISER_TYPES = {"yolo", "tsl"}
TSL_FAMILY_BY_MNEMONIC = {
    "kidneys": "kidney",
    "liver": "liver",
    "lungs": "lung",
    "pancreas": "pancreas",
    "colon": "colon",
}
RUN_NAME_PATTERN = re.compile(r"^[A-Z0-9]+-[A-Z0-9]+$")


@dataclass(frozen=True)
class InferenceSpec:
    inferer_cls: type
    run_name: str
    run_w: str | None = None
    localiser_labels: list[int] | None = None
    localiser_regions: list[str] | None = None
    k_largest: int | None = None


@dataclass(frozen=True)
class TargetRun:
    mnemonic: str | None
    run_name: str
    k_largest: int | None = None


def load_best_runs(path: Path = BEST_RUNS_PATH) -> dict:
    return load_yaml(path)


def looks_like_run_name(value: str) -> bool:
    return bool(RUN_NAME_PATTERN.fullmatch(value.strip()))


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


def ordered_runs(runs) -> list[str]:
    if isinstance(runs, dict):
        ordered = []
        for value in runs.values():
            ordered.extend(nonempty_runs(value))
        return ordered
    return nonempty_runs(runs)


def best_runs_entry_for_run_name(run_name: str, best_runs: dict) -> tuple[str | None, dict | None]:
    for name, entry in best_runs.items():
        if not isinstance(entry, dict) or "runs" not in entry:
            continue
        runs = entry["runs"]
        if run_name in nonempty_runs(runs):
            return name, entry
        if not isinstance(runs, dict):
            continue
        for value in runs.values():
            if run_name in nonempty_runs(value):
                return name, entry
    return None, None


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


def resolve_tsl_localiser_labels(mnemonic: str | None, run_name: str) -> list[int]:
    metadata = load_run_metadata(run_name)
    family = family_from_metadata(metadata)
    if family is not None:
        return tsl_region_labels(family)
    if mnemonic == "nodes":
        return tsl_region_labels("label_region")
    if mnemonic is None:
        raise ValueError(
            f"Cannot resolve fallback TSL label family for run_name={run_name}"
        )
    if mnemonic not in TSL_FAMILY_BY_MNEMONIC:
        raise ValueError(f"No TSL label family mapping for mnemonic={mnemonic}")
    return tsl_region_labels(TSL_FAMILY_BY_MNEMONIC[mnemonic])

def resolve_input_images(folder: str | None, datasets: list[str] | None) -> list[Path]:
    if (folder is None) == (datasets is None):
        raise ValueError("Pass exactly one of --folder or --dataset")
    roots = [Path(folder)] if folder is not None else [
        Path(item) if "/" in item else DS[item].folder / "images" for item in datasets
    ]
    images = []
    for root in roots:
        if root.is_dir():
            images.extend(sorted(p for p in root.glob("*") if p.is_file()))
        else:
            images.append(root)
    return images


def resolve_yolo_regions(run_name: str) -> list[str]:
    metadata = load_run_metadata(run_name)
    if "mode" in metadata:
        mode = normalize_mode(metadata["mode"])
        if mode != "rbd":
            raise ValueError(f"Expected YOLO mode for {run_name}, found {metadata['mode']}")
    if "localiser_regions" not in metadata:
        raise ValueError(f"Missing localiser_regions for YOLO run {run_name}")
    return parse_regions(metadata["localiser_regions"])


def resolve_standalone_run(runs) -> str:
    if isinstance(runs, list):
        return nonempty_runs(runs)[0]
    if "minimal" in runs and runs["minimal"]:
        return runs["minimal"][0]
    if "full" in runs and runs["full"]:
        return runs["full"][0]
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


def run_mode(run_name: str) -> str:
    metadata = load_run_metadata(run_name)
    if "mode" in metadata:
        return normalize_mode(metadata["mode"])
    _, mode = resolve_inferer_cls(run_name)
    return normalize_mode(mode)


def resolve_k_largest(entry: dict | None) -> int | None:
    if entry is None or "k_largest" not in entry:
        return None
    return entry["k_largest"]


def resolve_target_run(mnemonic_raw: str, best_runs: dict) -> TargetRun:
    token = mnemonic_raw.strip()
    if looks_like_run_name(token):
        mnemonic, entry = best_runs_entry_for_run_name(token, best_runs)
        return TargetRun(
            mnemonic=mnemonic,
            run_name=token,
            k_largest=resolve_k_largest(entry),
        )
    mnemonic = canonical_mnemonic(token, best_runs)
    entry = best_runs[mnemonic]
    return TargetRun(
        mnemonic=mnemonic,
        run_name=resolve_standalone_run(entry["runs"]),
        k_largest=resolve_k_largest(entry),
    )


def resolve_default_run_w(best_runs: dict) -> str:
    return best_runs["whole"]["runs"][0]


def resolve_yolo_spec(run_name: str, k_largest: int | None) -> InferenceSpec:
    return InferenceSpec(
        inferer_cls=CascadeInfererYOLO,
        run_name=run_name,
        localiser_regions=resolve_yolo_regions(run_name),
        k_largest=k_largest,
    )


def resolve_tsl_spec(
    mnemonic: str | None,
    run_name: str,
    run_w: str,
    k_largest: int | None,
) -> InferenceSpec:
    if run_mode(run_w) != "whole":
        raise ValueError(f"--run-w must resolve to whole run, found {run_w}")
    return InferenceSpec(
        inferer_cls=CascadeInferer,
        run_name=run_name,
        run_w=run_w,
        localiser_labels=resolve_tsl_localiser_labels(mnemonic, run_w),
        k_largest=k_largest,
    )


def native_localiser_type(run_name: str) -> str | None:
    mode = run_mode(run_name)
    if mode == "rbd":
        return "yolo"
    if mode in ("lbd", "pbd"):
        return "tsl"
    return None


def resolve_native_spec(target: TargetRun, best_runs: dict) -> InferenceSpec:
    localiser_type = native_localiser_type(target.run_name)
    if localiser_type == "yolo":
        return resolve_yolo_spec(target.run_name, target.k_largest)
    if localiser_type == "tsl":
        return resolve_tsl_spec(
            target.mnemonic,
            target.run_name,
            resolve_default_run_w(best_runs),
            target.k_largest,
        )
    inferer_cls = resolve_standalone_inferer_cls(target.run_name)
    return InferenceSpec(
        inferer_cls=inferer_cls,
        run_name=target.run_name,
        k_largest=target.k_largest,
    )


def resolve_override_spec(
    target: TargetRun,
    best_runs: dict,
    localiser_type: str | None,
    run_w: str | None,
) -> InferenceSpec | None:
    if run_w is not None and localiser_type == "yolo":
        raise ValueError("--run-w only supports tsl localiser override")
    if run_w is not None or localiser_type == "tsl":
        return resolve_tsl_spec(
            target.mnemonic,
            target.run_name,
            resolve_default_run_w(best_runs) if run_w is None else run_w,
            target.k_largest,
        )
    if localiser_type == "yolo":
        return resolve_yolo_spec(target.run_name, target.k_largest)
    return None


def resolve_spec(
    mnemonic_raw: str, localiser_type: str | None = None, run_w: str | None = None
) -> InferenceSpec:
    best_runs = load_best_runs()
    target = resolve_target_run(mnemonic_raw, best_runs)
    native_type = native_localiser_type(target.run_name)
    native_spec = resolve_native_spec(target, best_runs)
    if run_w is not None or localiser_type is not None:
        if native_type is None:
            raise ValueError(
                f"Run {target.run_name} is standalone; localiser override unsupported"
            )
        override_spec = resolve_override_spec(target, best_runs, localiser_type, run_w)
        print(
            {
                "selected_run": target.run_name,
                "native_localiser_type": native_type,
                "override_localiser_type": localiser_type if run_w is None else "tsl",
                "override_run_w": run_w,
            }
        )
        return override_spec
    print(
        {
            "selected_run": target.run_name,
            "native_localiser_type": native_type,
            "override_localiser_type": None,
            "override_run_w": None,
        }
    )
    return native_spec


def build_inferer(spec: InferenceSpec, gpus: list[int], patch_overlap: float):
    common_kwargs = dict(devices=gpus, save=True, save_channels=False)
    if spec.inferer_cls is CascadeInfererYOLO:
        return CascadeInfererYOLO(
            localiser_regions=spec.localiser_regions,
            run_p=spec.run_name,
            patch_overlap=patch_overlap,
            k_largest=spec.k_largest,
            **common_kwargs,
        )
    if spec.inferer_cls is CascadeInferer:
        return CascadeInferer(
            run_w=spec.run_w,
            run_p=spec.run_name,
            localiser_labels=spec.localiser_labels,
            patch_overlap=patch_overlap,
            save_localiser=False,
            k_largest=spec.k_largest,
            **common_kwargs,
        )
    if spec.inferer_cls in (BaseInferer, WholeImageInferer):
        return spec.inferer_cls(
            run_name=spec.run_name,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported inferer class {spec.inferer_cls}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Mnemonic-driven inference runner")
    parser.add_argument("mnemonic")
    parser.add_argument("--localiser-type", type=str.lower, choices=sorted(LOCALISER_TYPES))
    parser.add_argument("--run-w")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder")
    source.add_argument("--dataset", nargs="+")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--patch-overlap", type=float, default=0.2)
    parser.add_argument("-o", "--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(args=None):
    args = parse_args(args) if isinstance(args, list) or args is None else args
    input_images = resolve_input_images(args.folder, args.dataset)
    spec = resolve_spec(args.mnemonic, args.localiser_type, args.run_w)
    inferer = build_inferer(spec, args.gpus, args.patch_overlap)
    print(
        {
            "mnemonic": args.mnemonic,
            "run_name": spec.run_name,
            "run_w": spec.run_w,
            "inferer": spec.inferer_cls.__name__,
            "input_images_count": len(input_images),
        }
    )
    inferer.run(input_images, chunksize=args.chunksize, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
