import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from label_analysis.totalseg import TotalSegmenterLabels
from utilz.cprint import cprint
from utilz.fileio import load_yaml

from fran.configs.helpers import is_excel_None
from fran.data.dataregistry import DS
from fran.inference.helpers import load_params
from fran.utils.common import COMMON_PATHS

LOCALISER_TYPES = ("whole", "yolo")

cs = COMMON_PATHS["cold_storage_folder"]
best_runs = load_yaml(Path(cs) / "conf" / "best_runs.yaml")
MNEMONICS = [run for run in best_runs.keys() if run != "yolo"]
MNEMONIC_TSL_STRUCTURE_MAPPING = {
    "kidneys": "kidney",
    "lungs": "lung",
}
MNEMONIC_YOLO_REGION_MAPPING = {
    "kidney": "abdomen,pelvis",
    "kidneys": "abdomen,pelvis",
    "lung": "chest",
    "lungs": "chest",
    "liver": "abdomen",
    "nodes": "chest,abdomen,pelvis",
}


@dataclass(frozen=True)
class InferenceSpec:
    inferer_cls: type
    run_name: str
    run_w: str | None = None
    localiser_labels: list[int] | None = None
    localiser_regions: list[str] | None = None
    k_largest: int | None = None
    localiser_run_name: str | None = None


def build_inferer(
    spec: InferenceSpec,
    gpus: list[int],
    safe_mode=True,
    patch_overlap: float | None = None,
    save=True,
    save_channels=False,
):
    common_kwargs = {
        "devices": gpus,
        "save": save,
        "save_channels": save_channels,
    }
    cls_name = spec.inferer_cls.__name__
    if cls_name == "CascadeInfererYOLO":
        return spec.inferer_cls(
            localiser_regions=spec.localiser_regions,
            run_p=spec.run_name,
            patch_overlap=patch_overlap,
            k_largest=spec.k_largest,
            safe_mode=safe_mode,
            yolo_run_key=spec.localiser_run_name,
            **common_kwargs,
        )
    if cls_name == "CascadeInferer":
        return spec.inferer_cls(
            run_w=spec.run_w,
            run_p=spec.run_name,
            localiser_labels=spec.localiser_labels,
            patch_overlap=patch_overlap,
            save_localiser=False,
            safe_mode=safe_mode,
            k_largest=spec.k_largest,
            **common_kwargs,
        )
    if cls_name in ("BaseInferer", "WholeImageInferer"):
        return spec.inferer_cls(run_name=spec.run_name, **common_kwargs)
    raise ValueError(f"Unsupported inferer class {spec.inferer_cls}")


def needs_localiser(mnemonic: str = None, mode: str = None) -> bool:
    assert not (mnemonic and mode), "Must provide either mnemonic or mode, not both"
    if mnemonic:
        return mnemonic != "totalseg"
    if mode in ("lbd", "pbd", "rbd"):
        return True
    return False


def is_runname(input_str: str) -> bool:
    return bool(re.match(r"^[A-Z0-9]+-[A-Z0-9]+$", input_str))


def body_part_choices() -> list[str]:
    return [key for key in best_runs.keys() if key != "yolo"]


def canonicalise_body_part(text: str) -> str:
    token = str(text).strip().lower()
    if token in body_part_choices():
        return token
    raise ValueError(f"Unknown body part '{text}'")


def _flatten_runs(entry) -> list[str]:
    if isinstance(entry, str):
        return [entry]
    if isinstance(entry, list):
        return [str(item) for item in entry]
    if isinstance(entry, dict):
        runs = []
        for key, value in entry.items():
            if key == "k_largest":
                continue
            if key == "runs":
                runs.extend(_flatten_runs(value))
            else:
                runs.extend(_flatten_runs(value))
        return runs
    return []


def run_names_for_body_part(body_part: str) -> list[str]:
    canonical = canonicalise_body_part(body_part)
    runs = []
    seen = set()
    for run_name in _flatten_runs(best_runs[canonical]):
        if run_name not in seen:
            seen.add(run_name)
            runs.append(run_name)
    return runs


def localiser_run_choices(localiser_type: str) -> list[str]:
    if localiser_type == "whole":
        return list(best_runs["totalseg"][localiser_type]["runs"])
    if localiser_type == "yolo":
        return list(best_runs["yolo"].keys())
    raise ValueError(f"Unsupported localiser type {localiser_type}")


def mnemonic_to_run(mnemonic: str) -> str:
    canonical = canonicalise_body_part(mnemonic)
    entry = best_runs[canonical]
    if canonical == "totalseg":
        cprint("Using totalseg FULL run", "green")
        return entry["full"]["runs"][0]
    return run_names_for_body_part(canonical)[0]


def get_run_w(localiser_type: str, localiser_run_name: str | None = None) -> str:
    if localiser_type == "yolo":
        return localiser_run_name or "ab_ch_ne_pe"
    if localiser_run_name:
        return localiser_run_name
    return best_runs["totalseg"][localiser_type]["runs"][0]


def infer_localiser_type(mode: str) -> str:
    if mode == "rbd":
        return "yolo"
    if mode in ("lbd", "pbd"):
        return "whole"
    raise ValueError(f"Unsupported mode {mode} for inferring localiser type")


def row_from_local_params(
    run_name: str,
    mnemonic: str | None = None,
    k_largest: int | None = None,
):
    params = load_params(run_name)
    mode = params["configs"]["plan_train"]["mode"]
    row = pd.Series(
        {
            "run_name": run_name,
            "mnemonic": mnemonic,
            "mode": mode,
            "k_largest": k_largest,
        }
    )
    remapping_key = f"remapping_{mode}"
    row[remapping_key] = params["configs"]["plan_train"][remapping_key]
    return row


def get_run_row(
    run_name: str,
    mnemonic: str | None = None,
    k_largest: int | None = None,
):
    registry = Path(cs) / "conf" / "runs_registry.csv"
    df = pd.read_csv(registry)
    try:
        row = df[df.run_name == run_name].iloc[0]
    except IndexError as exc:
        try:
            return row_from_local_params(
                run_name=run_name,
                mnemonic=mnemonic,
                k_largest=k_largest,
            )
        except Exception as fallback_exc:
            raise ValueError(f"Run name {run_name} not found in registry") from fallback_exc
    k_largest = row["k_largest"]
    row["k_largest"] = None if is_excel_None(k_largest) else int(k_largest)
    return row


def tsl_label_loc(mnemonic: str, row_w):
    labels = TotalSegmenterLabels()
    if mnemonic == "nodes":
        return sorted(set(labels.label_region))
    mnemonic_key = MNEMONIC_TSL_STRUCTURE_MAPPING.get(mnemonic, mnemonic)
    structure = getattr(labels, mnemonic_key)
    mode_w = row_w["mode"]
    remapping_key = f"remapping_{mode_w}"
    remapping = row_w[remapping_key]
    remapping_ast = ast.literal_eval(remapping)
    if isinstance(remapping_ast, list) and len(remapping_ast) == 1:
        remapping_ast = remapping_ast[0]
    elif not isinstance(remapping_ast, dict):
        raise NotImplementedError(f"Unexpected remapping format: {remapping_ast}")

    labels_out_loc = set(remapping_ast.values())
    if len(labels_out_loc) == 20:
        tsl_mapping = "label_minimal"
    elif len(labels_out_loc) == 11:
        tsl_mapping = "label_organ"
    elif len(labels_out_loc) == 7:
        tsl_mapping = "label_region"
    else:
        raise NotImplementedError(
            f"Unexpected number of unique labels in remapping: {len(labels_out_loc)}"
        )
    return getattr(structure, tsl_mapping)


def resolve_selected_run(
    mnemonic_or_run: str,
    run_name: str | None = None,
) -> tuple[str, object, str]:
    if run_name:
        row = get_run_row(run_name, mnemonic=canonicalise_body_part(mnemonic_or_run))
        mnemonic_true = row["mnemonic"] or canonicalise_body_part(mnemonic_or_run)
        return run_name, row, mnemonic_true
    if is_runname(mnemonic_or_run):
        row = get_run_row(mnemonic_or_run)
        return mnemonic_or_run, row, row["mnemonic"]
    mnemonic_true = canonicalise_body_part(mnemonic_or_run)
    entry = best_runs[mnemonic_true]
    k_largest = entry.get("k_largest") if isinstance(entry, dict) else None
    errors = []
    for candidate_run in run_names_for_body_part(mnemonic_true):
        try:
            row = get_run_row(
                candidate_run,
                mnemonic=mnemonic_true,
                k_largest=k_largest,
            )
            row["mnemonic"] = row["mnemonic"] or mnemonic_true
            return candidate_run, row, row["mnemonic"]
        except Exception as exc:
            errors.append(f"{candidate_run}: {exc}")
    raise ValueError(
        f"No resolvable runs found for mnemonic {mnemonic_true}. Tried: {errors}"
    )


def resolve_inference_spec(
    mnemonic_or_run: str,
    run_name: str | None = None,
    localiser_type: str | None = None,
    localiser_run_name: str | None = None,
) -> InferenceSpec:
    run_name_resolved, row, mnemonic_true = resolve_selected_run(
        mnemonic_or_run=mnemonic_or_run,
        run_name=run_name,
    )
    needs_loc = needs_localiser(mnemonic=mnemonic_true)
    if needs_loc and not localiser_type:
        localiser_type = infer_localiser_type(row["mode"])
    k_largest = row["k_largest"]

    cprint(f"Resolved run name: {run_name_resolved}", "green")
    cprint(f"Resolved mnemonic: {mnemonic_true}", "green")
    cprint(f"K-largest : {k_largest}", "green")
    cprint(f"Localiser type: {localiser_type}", "green")

    if needs_loc:
        if localiser_type == "whole":
            from fran.inference.cascade import CascadeInferer

            run_w = get_run_w(localiser_type, localiser_run_name=localiser_run_name)
            row_w = get_run_row(run_w)
            label_loc = tsl_label_loc(mnemonic_true, row_w)
            cprint(f"RUN W: {run_w}", "green")
            return InferenceSpec(
                inferer_cls=CascadeInferer,
                run_name=run_name_resolved,
                run_w=run_w,
                localiser_labels=label_loc,
                k_largest=k_largest,
                localiser_run_name=run_w,
            )
        if localiser_type == "yolo":
            from fran.inference.cascade_yolo import CascadeInfererYOLO

            if mnemonic_true not in MNEMONIC_YOLO_REGION_MAPPING:
                raise ValueError(f"No YOLO localiser mapping for {mnemonic_true}")
            regions = MNEMONIC_YOLO_REGION_MAPPING[mnemonic_true]
            return InferenceSpec(
                inferer_cls=CascadeInfererYOLO,
                run_name=run_name_resolved,
                localiser_regions=[region.strip() for region in regions.split(",")],
                k_largest=k_largest,
                localiser_run_name=localiser_run_name,
            )
        raise ValueError(f"Unsupported localiser type {localiser_type}")

    mode = row["mode"]
    cprint(f"Mode: {mode}", "green")
    if mode == "whole":
        from fran.inference.cascade import WholeImageInferer

        return InferenceSpec(
            inferer_cls=WholeImageInferer,
            run_name=run_name_resolved,
        )
    if mode == "source":
        from fran.inference.base import BaseInferer

        return InferenceSpec(
            inferer_cls=BaseInferer,
            run_name=run_name_resolved,
        )
    raise ValueError(f"Unsupported standalone mode {mode}")


def resolve_mnemonic_run(args):
    return resolve_selected_run(
        mnemonic_or_run=args.mnemonic,
        run_name=getattr(args, "run_name", None),
    )


def resolve_inferer_cls(args):
    return resolve_inference_spec(
        mnemonic_or_run=args.mnemonic,
        run_name=getattr(args, "run_name", None),
        localiser_type=args.localiser_type,
        localiser_run_name=getattr(args, "localiser_run_name", None),
    )


def resolve_input_images(folder: list[str] | None, datasets: list[str] | None) -> list[Path]:
    if (folder is None) == (datasets is None):
        raise ValueError("Pass exactly one of --folder or --dataset")

    def supported_image_files(image_dir: Path) -> list[Path]:
        image_files = [fn for fn in image_dir.glob("*") if fn.is_file()]
        return sorted(
            [
                fn
                for fn in image_files
                if str(fn).endswith((".nii.gz", ".nii", ".nrrd"))
            ]
        )

    img_fns = []
    if folder is not None:
        for fldr in folder:
            imgs = supported_image_files(Path(fldr))
            img_fns.extend(imgs)
        return img_fns

    for item in datasets:
        ds = DS[item].folder / "images"
        imgs = supported_image_files(ds)
        img_fns.extend(imgs)
    return img_fns


def main(args):
    spec = resolve_inferer_cls(args)
    input_images = resolve_input_images(args.folder, args.dataset)
    inferer = build_inferer(spec, gpus=args.gpus, patch_overlap=args.patch_overlap)
    inferer.run(input_images, overwrite=args.overwrite, chunksize=args.chunksize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mnemonic-driven inference runner")
    parser.add_argument("mnemonic")
    parser.add_argument(
        "--localiser-type", type=str.lower, choices=list(LOCALISER_TYPES)
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder", nargs="+")
    source.add_argument("--dataset", nargs="+")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--patch-overlap", type=float, default=0.2)
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_known_args()[0]
    main(args)
