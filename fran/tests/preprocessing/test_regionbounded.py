from pathlib import Path

import pandas as pd
import pytest
import torch
from monai.data.meta_tensor import MetaTensor

from fran.preprocessing import regionbounded
from fran.transforms.spatialtransforms import CropForegroundMinShaped


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
    generator.output_folder = tmp_path

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
    generator.output_folder = tmp_path

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


def test_rbd_worker_passes_plan_expand_by_to_crop_by_yolo(monkeypatch):
    def fake_parent_create_transforms(self, device):
        self.transforms_dict = {}

    monkeypatch.setattr(
        regionbounded.RayWorkerBase,
        "create_transforms",
        fake_parent_create_transforms,
    )

    worker = regionbounded._RBDSamplerWorkerBase.__new__(
        regionbounded._RBDSamplerWorkerBase
    )
    worker.plan = {"expand_by": 14, "src_dims": (32, 32, 32)}

    worker.create_transforms(device="cpu")

    assert worker.cropper_yolo.margin == 14
    assert worker.CropByYolo.cropper_yolo.margin == 14
    assert worker.transforms_dict["CropByYolo"] is worker.CropByYolo


def test_rbd_worker_uses_zero_margin_when_expand_by_is_zero(monkeypatch):
    def fake_parent_create_transforms(self, device):
        self.transforms_dict = {}

    monkeypatch.setattr(
        regionbounded.RayWorkerBase,
        "create_transforms",
        fake_parent_create_transforms,
    )

    worker = regionbounded._RBDSamplerWorkerBase.__new__(
        regionbounded._RBDSamplerWorkerBase
    )
    worker.plan = {"expand_by": 0, "src_dims": (32, 32, 32)}

    worker.create_transforms(device="cpu")

    assert worker.cropper_yolo.margin == 0
    assert worker.CropByYolo.cropper_yolo.margin == 0


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
    assert "_preprocess_events" in out
    assert out["_preprocess_events"][0]["error_type"] == "CropByYolo"
    assert "bbox_txt_path=/tmp/case_001.txt" in out["_preprocess_events"][0]["error_message"]


def test_crop_by_yolo_with_fg_fallback_uses_fg_crop_when_yolo_loses_fg():
    image = _metatensor(torch.zeros((12, 12, 12)), "image.pt")
    lm = _metatensor(torch.zeros((12, 12, 12), dtype=torch.uint8), "lm.pt")
    lm[4:9, 4:9, 4:9] = 1
    fg_before = int(torch.count_nonzero(lm).item())
    bbox = {"width": (0.0, 0.1), "ap": (0.0, 0.1), "height": (0.0, 0.1)}

    cropper_yolo = regionbounded.CropByYolo(
        keys=["image", "lm"],
        lm_key="lm",
        bbox_key="bbox",
        margin=0,
        sanitize=True,
    )
    cropper_fg = CropForegroundMinShaped(
        keys=["image", "lm"],
        source_key="lm",
        min_shape=(4, 4, 4),
        margin=0,
    )
    transform = regionbounded.CropByYoloWithForegroundFallbackd(
        keys=["image", "lm"],
        lm_key="lm",
        bbox_key="bbox",
        cropper_yolo=cropper_yolo,
        cropper_fg=cropper_fg,
    )

    out = transform(
        {
            "case_id": "case_00486",
            "image": image,
            "lm": lm,
            "bbox": bbox,
            "bbox_fn": Path("/tmp/case_00486.txt"),
        }
    )

    assert out["image"].ndim == 3
    assert out["lm"].ndim == 3
    assert int(torch.count_nonzero(out["lm"]).item()) == fg_before
    events = out.get("_preprocess_events", [])
    event_types = [ev["error_type"] for ev in events]
    assert "CropByYolo" in event_types
    assert "CropByYoloFallback" in event_types
    fallback_messages = [
        ev["error_message"] for ev in events if ev["error_type"] == "CropByYoloFallback"
    ]
    assert any("verified_fg_preserved=True" in msg for msg in fallback_messages)


def test_crop_by_yolo_with_fg_fallback_noop_when_yolo_preserves_fg():
    image = _metatensor(torch.zeros((10, 10, 10)), "image.pt")
    lm = _metatensor(torch.zeros((10, 10, 10), dtype=torch.uint8), "lm.pt")
    lm[2:8, 2:8, 2:8] = 1
    bbox = {"width": (0.1, 0.9), "ap": (0.1, 0.9), "height": (0.1, 0.9)}
    fg_before = int(torch.count_nonzero(lm).item())

    cropper_yolo = regionbounded.CropByYolo(
        keys=["image", "lm"],
        lm_key="lm",
        bbox_key="bbox",
        margin=20,
        sanitize=True,
    )
    cropper_fg = CropForegroundMinShaped(
        keys=["image", "lm"],
        source_key="lm",
        min_shape=(4, 4, 4),
        margin=0,
    )
    transform = regionbounded.CropByYoloWithForegroundFallbackd(
        keys=["image", "lm"],
        lm_key="lm",
        bbox_key="bbox",
        cropper_yolo=cropper_yolo,
        cropper_fg=cropper_fg,
    )

    out = transform(
        {
            "case_id": "case_ok",
            "image": image,
            "lm": lm,
            "bbox": bbox,
            "bbox_fn": Path("/tmp/case_ok.txt"),
        }
    )

    assert int(torch.count_nonzero(out["lm"]).item()) == fg_before
    events = out.get("_preprocess_events", [])
    assert all(ev["error_type"] != "CropByYoloFallback" for ev in events)
