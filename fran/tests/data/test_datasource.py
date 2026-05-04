from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from fran.data.datasource import Datasource
from fran.preprocessing.datasetanalyzers import import_h5py


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


def _datasource_output(case_id: str, voxels: torch.Tensor):
    return {
        "case": {
            "case_id": case_id,
            "properties": {
                "spacing": [1.0, 1.0, 1.0],
                "labels": [1],
                "numel_fg": voxels.numel(),
                "mean_fg": float("nan") if voxels.numel() == 0 else 1.5,
                "min_fg": float("nan") if voxels.numel() == 0 else 1.0,
                "max_fg": float("nan") if voxels.numel() == 0 else 2.0,
                "std_fg": float("nan") if voxels.numel() == 0 else 0.5,
            },
        },
        "voxels": voxels,
    }


def _make_test_datasource(tmp_path: Path, outputs):
    ds = Datasource.__new__(Datasource)
    ds.h5_fname = tmp_path / "fg_voxels.h5"
    ds.outputs = outputs
    return ds


def test_dump_to_h5_writes_empty_foreground_voxels_and_attrs(tmp_path):
    ds = _make_test_datasource(
        tmp_path,
        [_datasource_output("case_empty", torch.tensor([], dtype=torch.float32))],
    )

    ds.dump_to_h5()

    h5py = import_h5py()
    with h5py.File(ds.h5_fname, "r") as h5f:
        stored = h5f["case_empty"]
        assert stored.shape == (0,)
        assert str(stored.dtype) == "float32"
        assert stored.attrs["numel_fg"] == 0
        assert list(stored.attrs["spacing"]) == [1.0, 1.0, 1.0]
        assert list(stored.attrs["labels"]) == [1]

    loaded = ds._load_raw_dataset_properties(["case_empty"])
    assert len(loaded) == 1
    assert loaded[0]["case_id"] == "case_empty"
    assert loaded[0]["properties"]["spacing"] == [1.0, 1.0, 1.0]
    assert loaded[0]["properties"]["labels"] == [1]
    assert loaded[0]["properties"]["numel_fg"] == 0
    assert loaded[0]["properties"]["mean_fg"] == pytest.approx(float("nan"), nan_ok=True)
    assert loaded[0]["properties"]["min_fg"] == pytest.approx(float("nan"), nan_ok=True)
    assert loaded[0]["properties"]["max_fg"] == pytest.approx(float("nan"), nan_ok=True)
    assert loaded[0]["properties"]["std_fg"] == pytest.approx(float("nan"), nan_ok=True)


def test_dump_to_h5_preserves_non_empty_voxels(tmp_path):
    voxels = torch.tensor([1.0, 2.0], dtype=torch.float32)
    ds = _make_test_datasource(tmp_path, [_datasource_output("case_full", voxels)])

    ds.dump_to_h5()

    h5py = import_h5py()
    with h5py.File(ds.h5_fname, "r") as h5f:
        stored = h5f["case_full"][:]
        assert stored.tolist() == [1.0, 2.0]
        assert h5f["case_full"].attrs["numel_fg"] == 2


@pytest.mark.skipif(
    "FRAN_CONF" not in os.environ,
    reason="Datasource smoke tests require FRAN_CONF dataset registry.",
)
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


@pytest.mark.skipif(
    "FRAN_CONF" not in os.environ,
    reason="Datasource smoke tests require FRAN_CONF dataset registry.",
)
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
