#!/usr/bin/env python3
"""Read ITK/SimpleITK label files and print unique labels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def get_labels(image: sitk.Image) -> list[int]:
    arr = sitk.GetArrayFromImage(image)
    return sorted(np.unique(arr).astype(int).tolist())


def read_itk_labels(folder_path: str | Path, extensions: list[str] | None = None) -> None:
    folder = Path(folder_path)
    if extensions is None:
        extensions = [".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".dcm"]
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    files: list[Path] = []
    for ext in extensions:
        files.extend(folder.glob(f"*{ext}"))
    files = sorted(set(files))
    if not files:
        print(f"No ITK files found in {folder}")
        return

    print(f"Found {len(files)} ITK files in {folder}")
    for fn in files:
        try:
            img = sitk.ReadImage(str(fn))
            print(f"{fn.name}: labels={get_labels(img)} size={img.GetSize()} spacing={img.GetSpacing()}")
        except Exception as e:
            print(f"{fn.name}: ERROR {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fran/extra/read_itk_labels.py <folder_path>")
        raise SystemExit(1)
    read_itk_labels(sys.argv[1])
