from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from fran.preprocessing.fixed_size2 import FixedSizeDataGenerator
from fran.preprocessing.helpers import import_h5py
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.preprocessing.patch import PatchDataGenerator
from fran.preprocessing.preprocessor import Preprocessor, create_hdf5_shards


def _write_case(output_folder, case_id: str, shape, fg_len: int, bg_len: int):
    images = output_folder / "images"
    lms = output_folder / "lms"
    indices = output_folder / "indices"
    for folder in (images, lms, indices):
        folder.mkdir(parents=True, exist_ok=True)

    image = torch.zeros(shape, dtype=torch.float32)
    lm = torch.zeros(shape, dtype=torch.uint8)
    lm.view(-1)[: min(10, lm.numel())] = 1
    idx = {
        "lm_fg_indices": torch.arange(fg_len, dtype=torch.int64),
        "lm_bg_indices": torch.arange(bg_len, dtype=torch.int64),
        "meta": {"filename_or_obj": f"/tmp/{case_id}.nii.gz", "case_id": case_id},
    }

    torch.save(image, images / f"{case_id}.pt")
    torch.save(lm, lms / f"{case_id}.pt")
    torch.save(idx, indices / f"{case_id}.pt")


def _prepare_three_cases(output_folder):
    _write_case(output_folder, "case_000", (220, 40, 20), fg_len=1024, bg_len=300000)
    _write_case(output_folder, "case_001", (48, 32, 16), fg_len=2048, bg_len=8192)
    _write_case(output_folder, "case_002", (36, 24, 12), fg_len=512, bg_len=4096)


def test_create_hdf5_shards_schema_chunks_and_manifest(tmp_path):
    output_folder = tmp_path / "preprocessed"
    _prepare_three_cases(output_folder)

    shards = create_hdf5_shards(
        output_folder=output_folder,
        src_dims=(192, 192, 128),
        cases_per_shard=2,
        compression="gzip",
        compression_opts=1,
    )

    assert [pth.name for pth in shards] == ["shard_0000.h5", "shard_0001.h5"]

    h5py = import_h5py()
    with h5py.File(shards[0], "r") as h5f:
        assert h5f.attrs["format"] == "fran_hdf5_shards_v1"
        assert list(h5f.attrs["src_dims"]) == [192, 192, 128]
        assert h5f.attrs["cases_per_shard"] == 2
        assert set(h5f["cases"].keys()) == {"case_000", "case_001"}

        case0 = h5f["cases"]["case_000"]
        assert case0["image"].dtype == np.dtype(np.float32)
        assert case0["lm"].dtype == np.dtype(np.uint8)
        assert case0["lm_fg_indices"].dtype == np.dtype(np.int64)
        assert case0["lm_bg_indices"].dtype == np.dtype(np.int64)
        assert case0["image"].chunks == (192, 40, 20)
        assert case0["lm"].chunks == (192, 40, 20)
        assert case0["lm_fg_indices"].chunks == (1024,)
        assert case0["lm_bg_indices"].chunks == (262144,)
        assert case0.attrs["source_meta_filename_or_obj"] == "/tmp/case_000.nii.gz"

    manifest_fn = output_folder / "hdf5_shards" / "src_192_192_128" / "manifest.json"
    with open(manifest_fn, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["format"] == "fran_hdf5_shards_v1"
    assert manifest["num_cases"] == 3
    assert manifest["num_shards"] == 2
    assert manifest["cases_per_shard"] == 2
    assert manifest["shards"][0]["case_ids"] == ["case_000", "case_001"]
    assert manifest["shards"][1]["case_ids"] == ["case_002"]


def test_create_hdf5_shards_skip_if_manifest_exists(tmp_path):
    output_folder = tmp_path / "preprocessed"
    _prepare_three_cases(output_folder)

    initial = create_hdf5_shards(
        output_folder=output_folder,
        src_dims=(192, 192, 128),
        cases_per_shard=2,
    )
    manifest_fn = output_folder / "hdf5_shards" / "src_192_192_128" / "manifest.json"
    before_mtime = manifest_fn.stat().st_mtime_ns

    second = create_hdf5_shards(
        output_folder=output_folder,
        src_dims=(192, 192, 128),
        cases_per_shard=1,
        overwrite=False,
    )
    after_mtime = manifest_fn.stat().st_mtime_ns

    assert [pth.name for pth in second] == [pth.name for pth in initial]
    assert after_mtime == before_mtime


def test_create_hdf5_shards_handles_empty_index_arrays(tmp_path):
    output_folder = tmp_path / "preprocessed"
    _write_case(output_folder, "case_fg_empty", (40, 32, 16), fg_len=0, bg_len=128)
    _write_case(output_folder, "case_bg_empty", (40, 32, 16), fg_len=256, bg_len=0)

    shards = create_hdf5_shards(
        output_folder=output_folder,
        src_dims=(192, 192, 128),
        cases_per_shard=2,
    )
    assert len(shards) == 1

    h5py = import_h5py()
    with h5py.File(shards[0], "r") as h5f:
        case_fg_empty = h5f["cases"]["case_fg_empty"]
        fg_ds = case_fg_empty["lm_fg_indices"]
        assert fg_ds.shape == (0,)
        assert fg_ds.dtype == np.dtype(np.int64)
        assert fg_ds[:].size == 0
        assert fg_ds.compression == "gzip"
        assert fg_ds.compression_opts == 1

        case_bg_empty = h5f["cases"]["case_bg_empty"]
        bg_ds = case_bg_empty["lm_bg_indices"]
        assert bg_ds.shape == (0,)
        assert bg_ds.dtype == np.dtype(np.int64)
        assert bg_ds[:].size == 0
        assert bg_ds.compression == "gzip"
        assert bg_ds.compression_opts == 1


def _make_preprocessor(output_folder, src_dims):
    pre = Preprocessor.__new__(Preprocessor)
    pre.output_folder = output_folder
    pre.plan = {"mode": "lbd", "src_dims": src_dims}
    pre.hdf5_shards = True
    pre.df_hdf5 = pd.DataFrame({"case_id": ["case_000", "case_001", "case_002"]})
    return pre


class _DummyPreprocessor(Preprocessor):
    def set_input_output_folders(self, data_folder, output_folder):
        self.data_folder = Path(data_folder) if data_folder is not None else None
        self.output_folder = Path(output_folder)


class _DummyRBDPreprocessor(_DummyPreprocessor):
    subfolder_key = "data_folder_rbd"


def test_preprocessor_init_defaults_hdf5_shards_off(tmp_path):
    pre = _DummyPreprocessor(
        project=SimpleNamespace(),
        plan={"mode": "lbd", "src_dims": (24, 24, 12)},
        data_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
    )

    assert pre.hdf5_shards is False


def test_preprocessor_only_enables_hdf5_shards_for_matching_plan_mode(tmp_path):
    pre_enabled = _DummyRBDPreprocessor(
        project=SimpleNamespace(),
        plan={"mode": "rbd", "src_dims": (24, 24, 12)},
        data_folder=tmp_path / "input_rbd",
        output_folder=tmp_path / "output_rbd",
        hdf5_shards=True,
    )
    pre_disabled = _DummyRBDPreprocessor(
        project=SimpleNamespace(),
        plan={"mode": "lbd", "src_dims": (24, 24, 12)},
        data_folder=tmp_path / "input_lbd",
        output_folder=tmp_path / "output_lbd",
        hdf5_shards=True,
    )

    assert pre_enabled.hdf5_shards is True
    assert pre_disabled.hdf5_shards is False


def test_labelbounded_init_only_enables_hdf5_shards_for_lbd_mode(tmp_path):
    pre_enabled = LabelBoundedDataGenerator(
        project=SimpleNamespace(),
        plan={"mode": "lbd", "src_dims": (24, 24, 12)},
        data_folder=tmp_path / "input_lbd",
        output_folder=tmp_path / "output_lbd",
    )
    pre_disabled = LabelBoundedDataGenerator(
        project=SimpleNamespace(),
        plan={"mode": "rbd", "src_dims": (24, 24, 12)},
        data_folder=tmp_path / "input_rbd",
        output_folder=tmp_path / "output_rbd",
    )

    assert pre_enabled.hdf5_shards is True
    assert pre_disabled.hdf5_shards is False


def test_preprocessor_uses_plan_src_dims_when_hdf5_shards_enabled(tmp_path):
    output_folder = tmp_path / "preprocessed"
    _prepare_three_cases(output_folder)
    pre = _make_preprocessor(output_folder, (24, 24, 12))

    shards = pre._maybe_create_hdf5_shards(df_hdf5_run=pre.df_hdf5)

    assert [pth.name for pth in shards] == ["shard_0000.h5"]
    manifest_fn = output_folder / "hdf5_shards" / "src_24_24_12" / "manifest.json"
    with open(manifest_fn, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["src_dims"] == [24, 24, 12]
    assert manifest["num_cases"] == 3


def test_preprocessor_can_explicitly_opt_out_of_hdf5_shard_creation(tmp_path):
    output_folder = tmp_path / "preprocessed"
    _prepare_three_cases(output_folder)
    pre = _make_preprocessor(output_folder, (24, 24, 12))

    pre.hdf5_shards = False
    shards = pre._maybe_create_hdf5_shards(df_hdf5_run=pre.df_hdf5)

    assert shards == []
    manifest_fn = output_folder / "hdf5_shards" / "src_24_24_12" / "manifest.json"
    assert manifest_fn.exists() is False


def test_patch_and_whole_generators_force_hdf5_shards_off(tmp_path, monkeypatch):
    class _FolderNamesStub:
        def __init__(self, project, plan):
            root = project.test_root
            self.folders = {
                "data_folder_whole": str(root / "whole"),
                "data_folder_pbd": str(root / "pbd"),
                "data_folder_lbd": str(root / "lbd"),
            }

    monkeypatch.setattr("fran.preprocessing.fixed_size2.FolderNames", _FolderNamesStub)
    monkeypatch.setattr("fran.preprocessing.patch.FolderNames", _FolderNamesStub)

    project = SimpleNamespace(test_root=tmp_path)
    whole = FixedSizeDataGenerator(
        project=project,
        plan={"mode": "whole", "src_dims": (24, 24, 12)},
        data_folder=tmp_path / "input_whole",
    )
    patch = PatchDataGenerator(
        project=project,
        plan={"mode": "patch", "src_dims": (24, 24, 12)},
        data_folder=tmp_path / "input_patch",
    )

    assert whole.hdf5_shards is False
    assert patch.hdf5_shards is False
