from __future__ import annotations

import os
from pathlib import Path

import pytest

from fran.data.datasource import Datasource
from fran.preprocessing.datasetanalyzers import import_h5py


pytestmark = pytest.mark.skipif(
    "FRAN_CONF" not in os.environ,
    reason="Datasource smoke tests require FRAN_CONF dataset registry.",
)


def _h5_case(h5f, case_id: str):
    ds = h5f.create_dataset(case_id, data=[1.0])
    ds.attrs["spacing"] = [1.0, 1.0, 1.0]
    ds.attrs["labels"] = [1]
    ds.attrs["numel_fg"] = 1
    ds.attrs["mean_fg"] = 1.0
    ds.attrs["min_fg"] = 1.0
    ds.attrs["max_fg"] = 1.0
    ds.attrs["std_fg"] = 0.0


def _link_case(src_folder: Path, dst_folder: Path, fname: str):
    for subfolder in ("images", "lms"):
        dst_subfolder = dst_folder / subfolder
        dst_subfolder.mkdir(parents=True, exist_ok=True)
        (dst_subfolder / fname).symlink_to(src_folder / subfolder / fname)


def test_update_datasource_smoke_with_short_dataset(tmp_path):
    from fran.data.dataregistry import DS

    src_folder = DS["kits23_short"].folder
    if not src_folder.exists():
        pytest.skip(f"Short test dataset is unavailable: {src_folder}")

    case_fnames = sorted((src_folder / "images").glob("*.nii.gz"))[:2]
    if len(case_fnames) < 2:
        pytest.skip(f"Need at least two short dataset cases in: {src_folder}")

    for case_fn in case_fnames:
        _link_case(src_folder, tmp_path, case_fn.name)

    h5py = import_h5py()
    with h5py.File(tmp_path / "fg_voxels.h5", "w") as h5f:
        _h5_case(h5f, "kits23_00000")
        _h5_case(h5f, "kits23_stale")

    ds = Datasource(tmp_path, name="kits23")
    summary = ds.update_datasource(dry_run=True)

    assert summary["added_case_ids"] == ["kits23_00001"]
    assert summary["removed_case_ids"] == ["kits23_stale"]
    assert summary["kept_case_ids"] == ["kits23_00000"]
    assert summary["processed_case_ids"] == []

    with h5py.File(tmp_path / "fg_voxels.h5", "r") as h5f:
        assert "kits23_stale" in h5f


def test_update_datasource_deletes_stale_h5_case_without_processing(tmp_path):
    from fran.data.dataregistry import DS

    src_folder = DS["kits23_short"].folder
    if not src_folder.exists():
        pytest.skip(f"Short test dataset is unavailable: {src_folder}")

    case_fnames = sorted((src_folder / "images").glob("*.nii.gz"))[:2]
    if len(case_fnames) < 2:
        pytest.skip(f"Need at least two short dataset cases in: {src_folder}")

    for case_fn in case_fnames:
        _link_case(src_folder, tmp_path, case_fn.name)

    h5py = import_h5py()
    with h5py.File(tmp_path / "fg_voxels.h5", "w") as h5f:
        _h5_case(h5f, "kits23_00000")
        _h5_case(h5f, "kits23_00001")
        _h5_case(h5f, "kits23_stale")

    ds = Datasource(tmp_path, name="kits23")
    summary = ds.update_datasource()

    assert summary["added_case_ids"] == []
    assert summary["removed_case_ids"] == ["kits23_stale"]
    assert summary["kept_case_ids"] == ["kits23_00000", "kits23_00001"]
    assert summary["processed_case_ids"] == []

    with h5py.File(tmp_path / "fg_voxels.h5", "r") as h5f:
        assert sorted(h5f.keys()) == ["kits23_00000", "kits23_00001"]
