from pathlib import Path

import torch
from localiser.inference import base as localiser_base
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


def test_apply_bboxes_materializes_contiguous_crop():
    inferer = cascade_yolo.CascadeInfererYOLO.__new__(
        cascade_yolo.CascadeInfererYOLO
    )
    image = MetaTensor(
        torch.arange(1 * 4 * 5 * 6, dtype=torch.float32).reshape(1, 4, 5, 6),
        meta={"spatial_shape": (4, 5, 6)},
    )
    bbox = (
        slice(0, 1),
        slice(1, 4),
        slice(1, 5),
        slice(2, 6),
    )
    data = [{"image": image, "bounding_box": bbox}]

    out = inferer.apply_bboxes(data)
    cropped = out[0]["image"]

    assert cropped.shape == (1, 3, 4, 4)
    assert cropped.is_contiguous()
    assert cropped.storage_offset() == 0
    assert out[0]["full_meta"]["spatial_shape"] == (4, 5, 6)
