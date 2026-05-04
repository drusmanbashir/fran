from pathlib import Path
from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from fran.inference.helpers import load_oriented_images
from localiser.inference import base as localiser_base
from localiser.inference.localiserinferer import LocaliserInferer
from monai.data import MetaTensor
from fran.inference import cascade_yolo


def _projection_tensor(start, filename):
    data = torch.arange(start, start + 12, dtype=torch.float32).reshape(1, 3, 2, 2)
    return MetaTensor(data, meta={"filename_or_obj": Path(filename)})


def test_collate_projections_stacks_once_and_keeps_case_metadata(monkeypatch):
    monkeypatch.setattr(
        localiser_base,
        "letterbox_meta",
        lambda image: {
            "letterbox_padded": ((0, 0), (1, 1), (2, 2)),
            "letterbox_orig_size": (2, 2),
            "letterbox_resized_size": (4, 4),
        },
    )
    batch = [
        {
            "image1": _projection_tensor(0, "case_001.nii.gz"),
            "image2": _projection_tensor(12, "case_001.nii.gz"),
            "image_orig": "orig-1",
        },
        {
            "image1": _projection_tensor(24, "case_002.nii.gz"),
            "image2": _projection_tensor(36, "case_002.nii.gz"),
            "image_orig": "orig-2",
        },
    ]

    out = localiser_base.collate_projections(batch)

    assert set(out) == {"image", "image_orig", "projection_meta"}
    assert out["image"].shape == (4, 3, 2, 2)
    assert out["image_orig"] == ["orig-1", "orig-2"]
    assert out["projection_meta"][0]["case_index"] == 0
    assert out["projection_meta"][0]["projection_index"] == 0
    assert out["projection_meta"][1]["projection_index"] == 1
    assert out["projection_meta"][2]["case_index"] == 1


def test_load_oriented_images_returns_channel_first_ras_tensor(tmp_path):
    image = sitk.GetImageFromArray(np.arange(24, dtype=np.int16).reshape(2, 3, 4))
    image_path = tmp_path / "case_001.nii.gz"
    sitk.WriteImage(image, str(image_path))

    out = load_oriented_images(image_path)

    assert len(out) == 1
    assert set(out[0]) == {"image"}
    assert out[0]["image"].shape == (1, 4, 3, 2)
    assert nib.aff2axcodes(out[0]["image"].meta["affine"].cpu().numpy()) == (
        "R",
        "A",
        "S",
    )


def test_load_oriented_images_preserves_case_record_keys(tmp_path):
    image = sitk.GetImageFromArray(np.arange(24, dtype=np.int16).reshape(2, 3, 4))
    image_path = tmp_path / "case_001.nii.gz"
    sitk.WriteImage(image, str(image_path))

    out = load_oriented_images(
        [{"case_id": "case_001", "image": image_path, "bbox": {"width": (0.1, 0.9)}}]
    )

    assert out[0]["case_id"] == "case_001"
    assert out[0]["bbox"] == {"width": (0.1, 0.9)}
    assert out[0]["image"].shape == (1, 4, 3, 2)


def test_load_case_original_uses_shared_loader_and_returns_contiguous_cpu(
    monkeypatch,
):
    inferer = LocaliserInferer.__new__(LocaliserInferer)
    seen = []
    image = torch.arange(24, dtype=torch.float32).reshape(1, 4, 3, 2).transpose(1, 2)

    monkeypatch.setattr(
        "localiser.inference.localiserinferer.load_oriented_images",
        lambda source: seen.append(source) or [{"image": image}],
    )

    out = inferer.load_case_original(Path("/tmp/case_001.nii.gz"))

    assert seen == [Path("/tmp/case_001.nii.gz")]
    assert torch.equal(out, image)
    assert out.device.type == "cpu"
    assert out.is_contiguous()


def test_package_preds_rebuilds_single_projection_images_from_combined_batch():
    inferer = localiser_base.LocaliserInferer.__new__(localiser_base.LocaliserInferer)
    batch = {
        "pred": ["lat-1", "ap-1", "lat-2", "ap-2"],
        "projection_meta": [
            {"projection_key": "image1", "case_index": 0},
            {"projection_key": "image2", "case_index": 0},
            {"projection_key": "image1", "case_index": 1},
            {"projection_key": "image2", "case_index": 1},
        ],
        "image": torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(4, 3, 2, 2),
        "image_orig": ["orig-1", "orig-2"],
    }

    outputs = inferer.package_preds(batch)

    assert len(outputs) == 2
    assert torch.equal(outputs[0]["image"], batch["image"][0:2])
    assert torch.equal(outputs[0]["image1"], batch["image"][0:1])
    assert torch.equal(outputs[0]["image2"], batch["image"][1:2])
    assert outputs[0]["image_orig"] == "orig-1"
    assert torch.equal(outputs[1]["image"], batch["image"][2:4])
    assert torch.equal(outputs[1]["image1"], batch["image"][2:3])
    assert torch.equal(outputs[1]["image2"], batch["image"][3:4])
    assert outputs[1]["image_orig"] == "orig-2"


def test_dataloader_num_workers_uses_zero_when_preprocess_moves_to_cuda():
    inferer = localiser_base.LocaliserInferer.__new__(localiser_base.LocaliserInferer)
    inferer.keys_preproc = "E,Dev,O"

    assert inferer.dataloader_num_workers([1, 2, 3, 4]) == 0


def test_dataloader_num_workers_keeps_cpu_worker_heuristic_without_dev():
    inferer = localiser_base.LocaliserInferer.__new__(localiser_base.LocaliserInferer)
    inferer.keys_preproc = "E,O"

    assert inferer.dataloader_num_workers([1, 2, 3, 4, 5, 6, 7, 8]) == 0


def test_maybe_filter_images_uses_cached_json_case_ids(tmp_path, monkeypatch):
    inferer = LocaliserInferer.__new__(LocaliserInferer)
    monkeypatch.setattr(
        LocaliserInferer,
        "output_folder",
        property(lambda self: tmp_path),
    )
    (tmp_path / "kits23_00097.json").write_text("{}\n")

    images = [
        Path("/tmp/kits23_00097.nii.gz"),
        Path("/tmp/kits23_00098.nii.gz"),
    ]

    filtered = inferer.maybe_filter_images(images, overwrite=False)

    assert filtered == [Path("/tmp/kits23_00098.nii.gz")]


def test_create_workspace_uses_fixed_path_and_resets_contents(tmp_path):
    inferer = LocaliserInferer.__new__(LocaliserInferer)
    inferer.temp_root = tmp_path / "localiser-work-root"

    workspace = inferer.create_workspace()
    stale_file = workspace / "manifests" / "stale.json"
    stale_file.write_text("{}\n")

    workspace_again = inferer.create_workspace()

    assert workspace == inferer.temp_root / "workspace"
    assert workspace_again == workspace
    assert workspace_again.exists()
    assert (workspace_again / "images" / "lat").is_dir()
    assert (workspace_again / "images" / "ap").is_dir()
    assert (workspace_again / "manifests").is_dir()
    assert not stale_file.exists()


def test_predict_from_workspace_reloads_case_originals_per_chunk(monkeypatch):
    inferer = LocaliserInferer.__new__(LocaliserInferer)
    inferer.bs = 2
    inferer.fabric_device = "cpu"
    inferer.fabric = SimpleNamespace(autocast=lambda: nullcontext())
    inferer.save_jpg = False
    inferer.mem_quota = 0.0

    loaded_sources = []

    monkeypatch.setattr(
        inferer,
        "load_projection_jpg",
        lambda jpg_path: torch.ones(3, 2, 2, dtype=torch.float32),
    )
    monkeypatch.setattr(
        inferer,
        "load_case_original",
        lambda source: loaded_sources.append(Path(source).name) or f"orig:{Path(source).name}",
    )
    monkeypatch.setattr(
        inferer,
        "model",
        lambda image_batch_device, verbose=False: [
            torch.tensor(float(index)) for index in range(len(image_batch_device))
        ],
    )
    monkeypatch.setattr(inferer, "combine_bboxes", lambda out: out)
    monkeypatch.setattr(inferer, "postprocess", lambda out: out)
    monkeypatch.setattr(inferer, "save_bboxes_final", lambda out: None)
    monkeypatch.setattr(inferer, "system_mem_remaining", lambda: 1.0)
    monkeypatch.setattr(inferer, "delete_image_orig", lambda outputs: None)

    def package_preds(self, batch):
        return [
            {"case_index": case_index, "image_orig": image_orig}
            for case_index, image_orig in enumerate(batch["image_orig"])
        ]

    inferer.package_preds = MethodType(package_preds, inferer)

    case_records = [
        {
            "case_id": "case_001",
            "source": Path("/tmp/case_001.nii.gz"),
            "projections": [
                {"jpg_path": Path("/tmp/case_001_lat.jpg"), "meta": {}, "projection_key": "image1", "orientation": "lat"},
                {"jpg_path": Path("/tmp/case_001_ap.jpg"), "meta": {}, "projection_key": "image2", "orientation": "ap"},
            ],
        },
        {
            "case_id": "case_002",
            "source": Path("/tmp/case_002.nii.gz"),
            "projections": [
                {"jpg_path": Path("/tmp/case_002_lat.jpg"), "meta": {}, "projection_key": "image1", "orientation": "lat"},
                {"jpg_path": Path("/tmp/case_002_ap.jpg"), "meta": {}, "projection_key": "image2", "orientation": "ap"},
            ],
        },
    ]

    outputs = inferer.predict_from_workspace(case_records)

    assert loaded_sources == ["case_001.nii.gz", "case_002.nii.gz"]
    assert [out["image_orig"] for out in outputs] == [
        "orig:case_001.nii.gz",
        "orig:case_002.nii.gz",
    ]


def test_apply_bboxes_materializes_contiguous_crop():
    inferer = cascade_yolo.CascadeInfererYOLO.__new__(
        cascade_yolo.CascadeInfererYOLO
    )
    inferer.cropper_yolo = cascade_yolo.CropByYolo(
        keys=["image"],
        lm_key=None,
        bbox_key="bbox",
        min_shape=(0, 0, 0),
        margin=0,
    )
    image = MetaTensor(
        torch.arange(1 * 4 * 5 * 6, dtype=torch.float32).reshape(1, 4, 5, 6),
        meta={"spatial_shape": (4, 5, 6), "affine": torch.eye(4)},
    )
    bbox = {"width": (0.25, 1.0), "ap": (0.2, 1.0), "height": (0.0, 2.0 / 3.0)}
    data = [{"image": image, "bbox": bbox, "case_id": "case_001"}]

    out = inferer.apply_bboxes(data)
    cropped = out[0]["image"]

    assert cropped.shape == (1, 3, 4, 4)
    assert cropped.is_contiguous()
    assert out[0]["bounding_box"] == (
        slice(0, 1),
        slice(1, 4),
        slice(1, 5),
        slice(2, 6),
    )
    assert cropped.storage_offset() == 0
    assert out[0]["full_meta"]["spatial_shape"] == (4, 5, 6)
