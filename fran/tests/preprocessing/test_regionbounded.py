from pathlib import Path

import pandas as pd
import pytest
import torch
from monai.data.meta_tensor import MetaTensor

from fran.preprocessing import regionbounded


def _metatensor(data, filename):
    return MetaTensor(
        data,
        meta={
            "affine": torch.eye(4),
            "filename_or_obj": filename,
        },
    )


def test_region_generator_adds_bbox_fn_by_case_id(tmp_path, monkeypatch):
    loc_folder = tmp_path / "localisers"
    loc_folder.mkdir()
    bbox_a = loc_folder / "drli_001.txt"
    bbox_b = loc_folder / "drli_002.txt"
    bbox_a.write_text("width: (0.1 0.5)\nap: (0.1 0.5)\nheight: (0.1 0.5)\n")
    bbox_b.write_text("width: (0.2 0.6)\nap: (0.2 0.6)\nheight: (0.2 0.6)\n")

    def fake_parent_create_data_df(self):
        self.df = pd.DataFrame({"case_id": ["drli_001", "drli_002"]})

    monkeypatch.setattr(
        regionbounded.LabelBoundedDataGenerator,
        "create_data_df",
        fake_parent_create_data_df,
    )

    generator = regionbounded.RegionBoundedDataGenerator.__new__(
        regionbounded.RegionBoundedDataGenerator
    )
    generator.data_folder = tmp_path

    generator.create_data_df()

    assert generator.df["bbox_fn"].tolist() == [bbox_a.resolve(), bbox_b.resolve()]


def test_region_generator_errors_on_duplicate_bbox_matches(tmp_path, monkeypatch):
    loc_folder = tmp_path / "localisers"
    loc_folder.mkdir()
    (loc_folder / "a.txt").write_text("width: (0.1 0.5)\n")
    (loc_folder / "b.txt").write_text("width: (0.1 0.5)\n")

    def fake_parent_create_data_df(self):
        self.df = pd.DataFrame({"case_id": ["case_001"]})

    monkeypatch.setattr(
        regionbounded.LabelBoundedDataGenerator,
        "create_data_df",
        fake_parent_create_data_df,
    )
    monkeypatch.setattr(
        regionbounded.RegionBoundedDataGenerator,
        "_case_id_keys_from_bbox_file",
        staticmethod(lambda fn: {"case_001"}),
    )

    generator = regionbounded.RegionBoundedDataGenerator.__new__(
        regionbounded.RegionBoundedDataGenerator
    )
    generator.data_folder = tmp_path

    with pytest.raises(ValueError, match="Duplicate bbox matches"):
        generator.create_data_df()


def test_rbd_worker_data_dict_loads_bbox(monkeypatch):
    bbox = {"width": (0.1, 0.5), "ap": (0.2, 0.7), "height": (0.0, 1.0)}
    bbox_fn = Path("/tmp/case_001.txt")
    seen = []

    def fake_bbox_from_file(fn):
        seen.append(fn)
        return bbox

    monkeypatch.setattr(regionbounded, "bbox_from_file", fake_bbox_from_file)
    worker = regionbounded._RBDSamplerWorkerBase.__new__(
        regionbounded._RBDSamplerWorkerBase
    )
    row = pd.Series(
        {
            "case_id": "case_001",
            "image": Path("/tmp/image.pt"),
            "lm": Path("/tmp/lm.pt"),
            "ds": "drli",
            "remapping": None,
            "bbox_fn": bbox_fn,
        }
    )

    data = worker._create_data_dict(row)

    assert seen == [bbox_fn]
    assert data["bbox"] == bbox
    assert data["bbox_fn"] == bbox_fn
    assert data["case_id"] == "case_001"


def test_crop_by_yolo_default_margin_recovers_cropped_label():
    image = _metatensor(torch.zeros((10, 12, 8)), "image.pt")
    lm = _metatensor(torch.zeros((10, 12, 8)), "lm.pt")
    lm[1:9, 2:10, 1:7] = 1
    bbox = {"width": (0.3, 0.6), "ap": (0.3, 0.6), "height": (0.3, 0.6)}

    transform = regionbounded.CropByYolo()
    out = transform(
        {
            "case_id": "case_001",
            "image": image,
            "lm": lm,
            "bbox": bbox,
            "bbox_fn": Path("/tmp/case_001.txt"),
        }
    )

    assert transform.margin == 20
    assert torch.count_nonzero(out["lm"]) == torch.count_nonzero(lm)
    assert out["image"].shape == out["lm"].shape
