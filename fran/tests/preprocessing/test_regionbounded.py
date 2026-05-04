from pathlib import Path

import pandas as pd
import pytest
import torch
from monai.data.meta_tensor import MetaTensor

from fran.preprocessing import regionbounded
from fran.transforms.spatialtransforms import CropByYoloWithForegroundFallbackd


def _metatensor(data, filename):
    return MetaTensor(
        data,
        meta={
            "affine": torch.eye(4),
            "filename_or_obj": filename,
        },
    )


def test_region_generator_attaches_cached_bbox_json_by_case_id(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cached_localiser"
    cache_dir.mkdir()
    bbox_a = cache_dir / "drli_001.json"
    bbox_b = cache_dir / "drli_002.json"
    bbox_a.write_text(
        '{"ap": {"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}, '
        '"lat": {"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}}\n'
    )
    bbox_b.write_text(
        '{"ap": {"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}, '
        '"lat": {"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}}\n'
    )

    class FakeLocaliserInferer:
        def __init__(self, *args, **kwargs):
            self.output_folder = cache_dir
            self.yolo_state_dict = {"data": {"names": ["abdomen"]}}

        def run(self, imgs, overwrite=False):
            raise AssertionError("run() should not be reached when cached bbox JSON exists")

    monkeypatch.setattr(regionbounded, "LocaliserInfererPT", FakeLocaliserInferer)
    monkeypatch.setattr(
        regionbounded,
        "standardize_bboxes",
        lambda *args, **kwargs: {
            "width": (0.1, 0.5),
            "ap": (0.2, 0.6),
            "height": (0.3, 0.7),
        },
    )

    generator = regionbounded.RegionBoundedDataGenerator.__new__(
        regionbounded.RegionBoundedDataGenerator
    )
    generator.output_folder = tmp_path
    generator.devices = ["cpu"]
    generator.plan = {"localiser_regions": "abdomen"}
    generator.df = pd.DataFrame(
        {
            "case_id": ["drli_001", "drli_002"],
            "image": [tmp_path / "drli_001.nii.gz", tmp_path / "drli_002.nii.gz"],
            "bbox_fn": [None, None],
            "bbox": [None, None],
        }
    )

    generator.maybe_infer_bboxes()

    assert generator.df["bbox_fn"].tolist() == [bbox_a.resolve(), bbox_b.resolve()]
    assert generator.df["bbox"].tolist() == [
        {"width": (0.1, 0.5), "ap": (0.2, 0.6), "height": (0.3, 0.7)},
        {"width": (0.1, 0.5), "ap": (0.2, 0.6), "height": (0.3, 0.7)},
    ]


def test_region_generator_errors_on_duplicate_bbox_matches(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cached_localiser"
    cache_dir.mkdir()
    (cache_dir / "a.json").write_text(
        '{"ap": {"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}, '
        '"lat": {"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}}\n'
    )
    (cache_dir / "b.json").write_text(
        '{"ap": {"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}, '
        '"lat": {"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}}\n'
    )

    class FakeLocaliserInferer:
        def __init__(self, *args, **kwargs):
            self.output_folder = cache_dir
            self.yolo_state_dict = {"data": {"names": ["abdomen"]}}

        def run(self, imgs, overwrite=False):
            raise AssertionError("run() should not be reached for duplicate bbox matches")

    monkeypatch.setattr(regionbounded, "LocaliserInfererPT", FakeLocaliserInferer)
    monkeypatch.setattr(
        regionbounded.RegionBoundedDataGenerator,
        "_case_id_keys_from_bbox_file",
        staticmethod(lambda fn: {"case_001"}),
    )
    monkeypatch.setattr(
        regionbounded,
        "standardize_bboxes",
        lambda *args, **kwargs: {
            "width": (0.1, 0.5),
            "ap": (0.2, 0.6),
            "height": (0.3, 0.7),
        },
    )

    generator = regionbounded.RegionBoundedDataGenerator.__new__(
        regionbounded.RegionBoundedDataGenerator
    )
    generator.output_folder = tmp_path
    generator.devices = ["cpu"]
    generator.plan = {"localiser_regions": "abdomen"}
    generator.df = pd.DataFrame(
        {
            "case_id": ["case_001"],
            "image": [tmp_path / "case_001.nii.gz"],
            "bbox_fn": [None],
            "bbox": [None],
        }
    )

    with pytest.raises(ValueError, match="Duplicate bbox matches"):
        generator.maybe_infer_bboxes()


def test_region_generator_uses_cached_bbox_json_before_inference(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cached_localiser"
    cache_dir.mkdir()
    cached_json = cache_dir / "kits23_00097.json"
    cached_json.write_text(
        '{"ap": {"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}, '
        '"lat": {"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}}\n'
    )

    run_calls = []

    class FakeLocaliserInferer:
        def __init__(self, *args, **kwargs):
            self.output_folder = cache_dir
            self.yolo_state_dict = {"data": {"names": ["abdomen"]}}

        def run(self, imgs, overwrite=False):
            run_calls.append((list(imgs), overwrite))
            return []

    monkeypatch.setattr(
        regionbounded, "LocaliserInfererPT", FakeLocaliserInferer
    )
    monkeypatch.setattr(
        regionbounded,
        "standardize_bboxes",
        lambda *args, **kwargs: {
            "width": (0.1, 0.5),
            "ap": (0.2, 0.6),
            "height": (0.3, 0.7),
        },
    )

    generator = regionbounded.RegionBoundedDataGenerator.__new__(
        regionbounded.RegionBoundedDataGenerator
    )
    generator.output_folder = tmp_path
    generator.devices = ["cpu"]
    generator.plan = {"localiser_regions": "abdomen"}
    generator.df = pd.DataFrame(
        {
            "case_id": ["kits23_00097"],
            "image": [tmp_path / "kits23_00097.nii.gz"],
            "bbox_fn": [None],
            "bbox": [None],
        }
    )

    generator.maybe_infer_bboxes()

    assert run_calls == []
    assert generator.df.loc[0, "bbox_fn"] == cached_json.resolve()
    assert generator.df.loc[0, "bbox"] == {
        "width": (0.1, 0.5),
        "ap": (0.2, 0.6),
        "height": (0.3, 0.7),
    }


def test_region_generator_attaches_empty_cached_bbox_sentinel_without_inference(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cached_localiser"
    cache_dir.mkdir()
    cached_json = cache_dir / "kits23_00097.json"
    cached_json.write_text(
        '{"ap": {"xyxy": [], "conf": [], "cls": [], "orig_shape": [100, 80], '
        '"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}, '
        '"lat": {"xyxy": [], "conf": [], "cls": [], "orig_shape": [100, 80], '
        '"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}}\n'
    )

    run_calls = []

    class FakeLocaliserInferer:
        def __init__(self, *args, **kwargs):
            self.output_folder = cache_dir
            self.yolo_state_dict = {"data": {"names": ["abdomen"]}}

        def run(self, imgs, overwrite=False):
            run_calls.append((list(imgs), overwrite))
            return []

    monkeypatch.setattr(regionbounded, "LocaliserInfererPT", FakeLocaliserInferer)
    generator = regionbounded.RegionBoundedDataGenerator.__new__(
        regionbounded.RegionBoundedDataGenerator
    )
    generator.output_folder = tmp_path
    generator.devices = ["cpu"]
    generator.plan = {"localiser_regions": "abdomen"}
    generator.df = pd.DataFrame(
        {
            "case_id": ["kits23_00097"],
            "image": [tmp_path / "kits23_00097.nii.gz"],
            "bbox_fn": [None],
            "bbox": [None],
        }
    )

    generator.maybe_infer_bboxes()

    assert run_calls == []
    assert generator.df.loc[0, "bbox_fn"] == cached_json.resolve()
    assert generator.df.loc[0, "bbox"] == {"empty_bbox": True}


def test_region_generator_ignores_unrelated_cached_bbox_jsons(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cached_localiser"
    cache_dir.mkdir()
    cached_json = cache_dir / "kits23_00097.json"
    cached_json.write_text(
        '{"ap": {"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}, '
        '"lat": {"meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}}\n'
    )
    unrelated_json = cache_dir / "totalseg_s0003.json"
    unrelated_json.write_text('{"broken": true}\n')

    run_calls = []

    class FakeLocaliserInferer:
        def __init__(self, *args, **kwargs):
            self.output_folder = cache_dir
            self.yolo_state_dict = {"data": {"names": ["abdomen"]}}

        def run(self, imgs, overwrite=False):
            run_calls.append((list(imgs), overwrite))
            return []

    monkeypatch.setattr(regionbounded, "LocaliserInfererPT", FakeLocaliserInferer)
    monkeypatch.setattr(
        regionbounded,
        "standardize_bboxes",
        lambda *args, **kwargs: {
            "width": (0.1, 0.5),
            "ap": (0.2, 0.6),
            "height": (0.3, 0.7),
        },
    )

    generator = regionbounded.RegionBoundedDataGenerator.__new__(
        regionbounded.RegionBoundedDataGenerator
    )
    generator.output_folder = tmp_path
    generator.devices = ["cpu"]
    generator.plan = {"localiser_regions": "abdomen"}
    generator.df = pd.DataFrame(
        {
            "case_id": ["kits23_00097"],
            "image": [tmp_path / "kits23_00097.nii.gz"],
            "bbox_fn": [None],
            "bbox": [None],
        }
    )

    generator.maybe_infer_bboxes()

    assert run_calls == []
    assert generator.df.loc[0, "bbox_fn"] == cached_json.resolve()
    assert generator.df.loc[0, "bbox"] == {
        "width": (0.1, 0.5),
        "ap": (0.2, 0.6),
        "height": (0.3, 0.7),
    }


def test_region_generator_errors_when_cached_bbox_json_has_no_requested_class_match(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cached_localiser"
    cache_dir.mkdir()
    cached_json = cache_dir / "kits23_00097.json"
    cached_json.write_text(
        '{"ap": {"cls": [0], "conf": [0.9], "xyxy": [[10, 20, 30, 40]], '
        '"orig_shape": [100, 80], "meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}, '
        '"lat": {"cls": [0], "conf": [0.8], "xyxy": [[15, 25, 35, 45]], '
        '"orig_shape": [100, 80], "meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]}}}\n'
    )

    class FakeLocaliserInferer:
        def __init__(self, *args, **kwargs):
            self.output_folder = cache_dir
            self.yolo_state_dict = {"data": {"names": ["kidney", "abdomen"]}}

        def run(self, imgs, overwrite=False):
            raise AssertionError("run() should not be reached when cached bbox JSON fails")

    monkeypatch.setattr(
        regionbounded, "LocaliserInfererPT", FakeLocaliserInferer
    )

    generator = regionbounded.RegionBoundedDataGenerator.__new__(
        regionbounded.RegionBoundedDataGenerator
    )
    generator.output_folder = tmp_path
    generator.devices = ["cpu"]
    generator.plan = {"localiser_regions": "abdomen"}
    generator.df = pd.DataFrame(
        {
            "case_id": ["kits23_00097"],
            "image": [tmp_path / "kits23_00097.nii.gz"],
            "bbox_fn": [None],
            "bbox": [None],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "Failed to standardize cached bbox JSON for RBD preprocessing\\. "
            "case_id=kits23_00097 .*requested_classes=\\[1\\] "
            "detected_classes=\\[0\\]"
        ),
    ):
        generator.maybe_infer_bboxes()


def test_rbd_worker_data_dict_loads_bbox(monkeypatch):
    bbox = {"width": (0.1, 0.5), "ap": (0.2, 0.7), "height": (0.0, 1.0)}
    bbox_fn = Path("/tmp/case_001.json")
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
            "bbox": bbox,
        }
    )

    data = worker._create_data_dict(row)

    assert data["bbox"] == bbox
    assert data["bbox_fn"] == bbox_fn
    assert data["case_id"] == "case_001"


def test_rbd_worker_uses_fallback_crop_wrapper_in_preprocessing_path(monkeypatch):
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

    assert isinstance(worker.CropByYolo, CropByYoloWithForegroundFallbackd)
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

    assert isinstance(worker.CropByYolo, CropByYoloWithForegroundFallbackd)
    assert worker.CropByYolo.cropper_yolo.margin == 0


def test_rbd_worker_crop_transform_preserves_fg_on_cpu(monkeypatch):
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
    worker.plan = {"expand_by": 0, "src_dims": (4, 4, 4)}

    worker.create_transforms(device="cpu")

    image = _metatensor(torch.zeros((1, 12, 12, 12)), "image.pt")
    lm = _metatensor(torch.zeros((1, 12, 12, 12), dtype=torch.uint8), "lm.pt")
    lm[:, 4:9, 4:9, 4:9] = 1
    fg_before = int(torch.count_nonzero(lm).item())

    out = worker.transforms_dict["CropByYolo"](
        {
            "case_id": "case_cpu",
            "image": image,
            "lm": lm,
            "bbox": {"width": (0.0, 0.1), "ap": (0.0, 0.1), "height": (0.0, 0.1)},
            "bbox_fn": Path("/tmp/case_cpu.json"),
            "_preprocess_events": [],
        }
    )

    assert int(torch.count_nonzero(out["lm"]).item()) == fg_before
    assert out["image"].device.type == "cpu"
    assert out["lm"].device.type == "cpu"
    assert any(
        event["error_type"] == "CropByYoloFallback"
        for event in out["_preprocess_events"]
    )


def test_crop_by_yolo_default_margin_recovers_cropped_label():
    image = _metatensor(torch.zeros((1, 10, 12, 8)), "image.pt")
    lm = _metatensor(torch.zeros((1, 10, 12, 8)), "lm.pt")
    lm[:, 1:9, 2:10, 1:7] = 1
    bbox = {"width": (0.3, 0.6), "ap": (0.3, 0.6), "height": (0.3, 0.6)}

    transform = regionbounded.CropByYolo()
    out = transform(
        {
            "case_id": "case_001",
            "image": image,
            "lm": lm,
            "bbox": bbox,
            "bbox_fn": Path("/tmp/case_001.json"),
            "_preprocess_events": [],
        }
    )

    assert transform.margin == 20
    assert torch.count_nonzero(out["lm"]) == torch.count_nonzero(lm)
    assert out["image"].shape == out["lm"].shape
    assert "_preprocess_events" in out
    assert out["_preprocess_events"][0]["error_type"] == "CropByYolo"
    assert "bbox_source_path=/tmp/case_001.json" in out["_preprocess_events"][0]["error_message"]


def test_crop_by_yolo_with_fg_fallback_uses_fg_crop_when_yolo_loses_fg():
    image = _metatensor(torch.zeros((1, 12, 12, 12)), "image.pt")
    lm = _metatensor(torch.zeros((1, 12, 12, 12), dtype=torch.uint8), "lm.pt")
    lm[:, 4:9, 4:9, 4:9] = 1
    fg_before = int(torch.count_nonzero(lm).item())
    bbox = {"width": (0.0, 0.1), "ap": (0.0, 0.1), "height": (0.0, 0.1)}
    transform = CropByYoloWithForegroundFallbackd(
        min_shape=(4, 4, 4),
        keys=["image", "lm"],
        lm_key="lm",
        bbox_key="bbox",
        margin=0,
    )

    out = transform(
        {
            "case_id": "case_00486",
            "image": image,
            "lm": lm,
            "bbox": bbox,
            "bbox_fn": Path("/tmp/case_00486.json"),
            "_preprocess_events": [],
        }
    )

    assert out["image"].ndim == 4
    assert out["lm"].ndim == 4
    assert int(torch.count_nonzero(out["lm"]).item()) == fg_before
    events = out.get("_preprocess_events", [])
    event_types = [ev["error_type"] for ev in events]
    assert "CropByYoloFallback" in event_types
    fallback_messages = [
        ev["error_message"] for ev in events if ev["error_type"] == "CropByYoloFallback"
    ]
    assert fallback_messages == ["fg loss"]


def test_crop_by_yolo_with_fg_fallback_bypasses_yolo_for_empty_bbox():
    image = _metatensor(torch.zeros((1, 12, 12, 12)), "image.pt")
    lm = _metatensor(torch.zeros((1, 12, 12, 12), dtype=torch.uint8), "lm.pt")
    lm[:, 4:9, 4:9, 4:9] = 1
    fg_before = int(torch.count_nonzero(lm).item())
    transform = CropByYoloWithForegroundFallbackd(
        min_shape=(4, 4, 4),
        keys=["image", "lm"],
        lm_key="lm",
        bbox_key="bbox",
        margin=0,
    )

    out = transform(
        {
            "case_id": "case_empty",
            "image": image,
            "lm": lm,
            "bbox": {"empty_bbox": True},
            "bbox_fn": Path("/tmp/case_empty.json"),
            "_preprocess_events": [],
        }
    )

    assert out["image"].ndim == 4
    assert out["lm"].ndim == 4
    assert int(torch.count_nonzero(out["lm"]).item()) == fg_before
    events = out.get("_preprocess_events", [])
    assert events[-1]["error_type"] == "CropByYoloFallback"
    assert events[-1]["error_message"] == "empty bbox"


def test_crop_by_yolo_with_fg_fallback_noop_when_yolo_preserves_fg():
    image = _metatensor(torch.zeros((1, 10, 10, 10)), "image.pt")
    lm = _metatensor(torch.zeros((1, 10, 10, 10), dtype=torch.uint8), "lm.pt")
    lm[:, 2:8, 2:8, 2:8] = 1
    bbox = {"width": (0.1, 0.9), "ap": (0.1, 0.9), "height": (0.1, 0.9)}
    fg_before = int(torch.count_nonzero(lm).item())
    transform = CropByYoloWithForegroundFallbackd(
        min_shape=(4, 4, 4),
        keys=["image", "lm"],
        lm_key="lm",
        bbox_key="bbox",
        margin=20,
    )
    data = {
        "case_id": "case_ok",
        "image": image,
        "lm": lm,
        "bbox": bbox,
        "bbox_fn": Path("/tmp/case_ok.json"),
        "_preprocess_events": [],
    }

    out = transform(dict(data))
    yolo_out = transform.cropper_yolo(dict(data))

    assert int(torch.count_nonzero(out["lm"]).item()) == fg_before
    assert torch.equal(out["image"], yolo_out["image"])
    assert torch.equal(out["lm"], yolo_out["lm"])
    events = out.get("_preprocess_events", [])
    assert all(ev["error_type"] != "CropByYoloFallback" for ev in events)
