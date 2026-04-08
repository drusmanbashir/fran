#!/usr/bin/env python3
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

# Use local label_analysis codebase requested by user.
LABEL_ANALYSIS_ROOT = Path.home() / "code" / "label_analysis"
sys.path.insert(0, str(LABEL_ANALYSIS_ROOT))

from label_analysis.geometry_itk import LabelMapGeometryITK  # noqa: E402
from label_analysis.helpers import get_labels  # noqa: E402


def _case_id_from_name(fn: Path) -> str:
    name = fn.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return fn.stem


def _one_file_rows(lm_fn: Path) -> list[dict]:
    lm = sitk.ReadImage(str(lm_fn))
    labels = get_labels(lm)

    # geometry_itk currently crashes on empty masks; skip cleanly.
    if not labels:
        return [
            {
                "case_id": _case_id_from_name(lm_fn),
                "lm_fn": str(lm_fn),
                "is_empty": True,
                "label_org": None,
                "label_cc": None,
                "cent_x": None,
                "cent_y": None,
                "cent_z": None,
                "bbox_x": None,
                "bbox_y": None,
                "bbox_z": None,
                "bbox_sx": None,
                "bbox_sy": None,
                "bbox_sz": None,
                "flatness": None,
                "feret": None,
                "major_axis": None,
                "minor_axis": None,
                "least_axis": None,
                "volume_mm3": None,
            }
        ]

    L = LabelMapGeometryITK(lm)
    rows = []
    for _, r in L.nbrhoods.iterrows():
        cent = r["cent"]
        bbox = r["bbox"]
        rows.append(
            {
                "case_id": _case_id_from_name(lm_fn),
                "lm_fn": str(lm_fn),
                "is_empty": False,
                "label_org": int(r["label_org"]),
                "label_cc": int(r["label_cc"]),
                "cent_x": float(cent[0]),
                "cent_y": float(cent[1]),
                "cent_z": float(cent[2]),
                "bbox_x": int(bbox[0]),
                "bbox_y": int(bbox[1]),
                "bbox_z": int(bbox[2]),
                "bbox_sx": int(bbox[3]),
                "bbox_sy": int(bbox[4]),
                "bbox_sz": int(bbox[5]),
                "flatness": float(r["flatness"]),
                "feret": float(r["feret"]),
                "major_axis": float(r["major_axis"]),
                "minor_axis": float(r["minor_axis"]),
                "least_axis": float(r["least_axis"]),
                "volume_mm3": float(r["volume_mm3"]),
            }
        )
    return rows


def main(lm_folder: Path, nproc: int, out_csv: Path | None) -> Path:
    lm_folder = lm_folder.expanduser().resolve()
    if out_csv is None:
        out_csv = lm_folder.parent / f"{lm_folder.name}_labelmap_geometry_itk.csv"
    else:
        out_csv = out_csv.expanduser().resolve()

    lm_files = sorted(list(lm_folder.glob("*.nii.gz")) + list(lm_folder.glob("*.nii")))
    if len(lm_files) == 0:
        raise FileNotFoundError(f"No .nii/.nii.gz files found in {lm_folder}")

    tasks = [(fn,) for fn in lm_files]
    rows_all = []
    with mp.Pool(processes=max(1, int(nproc))) as pool:
        for rows in pool.starmap(_one_file_rows, tasks):
            rows_all.extend(rows)

    df = pd.DataFrame(rows_all)
    df.sort_values(
        ["case_id", "label_org", "label_cc"], inplace=True, na_position="last"
    )
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(f"Rows: {len(df)} | Files: {len(lm_files)}")
    return out_csv


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--lm-folder",
        type=Path,
        default=Path("/s/fran_storage/predictions/lidc/LIDC-0021/"),
    )
    p.add_argument("--nproc", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    main(args.lm_folder, args.nproc, args.out_csv)
