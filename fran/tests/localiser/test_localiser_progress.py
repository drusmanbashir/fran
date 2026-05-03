from contextlib import nullcontext
from pathlib import Path

import torch

from fran.preprocessing import regionbounded
from localiser.inference import localiserinferer as localiserinferer_module
from localiser.inference.localiserinferer import LocaliserInferer


class FakeProgress:
    instances = []

    def __init__(self, *args, **kwargs):
        self.total = kwargs["total"]
        self.desc = kwargs["desc"]
        self.leave = kwargs["leave"]
        self.updates = []
        self.__class__.instances.append(self)

    def update(self, n):
        self.updates.append(n)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_preprocess_to_workspace_progress_tracks_cases(monkeypatch, tmp_path):
    FakeProgress.instances.clear()
    monkeypatch.setattr(localiserinferer_module, "tqdm", FakeProgress)

    inferer = LocaliserInferer.__new__(LocaliserInferer)
    inferer.bs = 2

    images = [Path(f"/tmp/case_{index:03d}.nii.gz") for index in range(1, 6)]

    monkeypatch.setattr(inferer, "write_workspace_args", lambda workspace, data: None)

    def prepare_data(data_chunk):
        inferer.pred_dl = [list(data_chunk)]

    monkeypatch.setattr(inferer, "prepare_data", prepare_data)
    monkeypatch.setattr(inferer, "preprocess_case", lambda case: {"case": case})
    monkeypatch.setattr(
        inferer,
        "export_case",
        lambda processed, workspace: processed["case"],
    )
    monkeypatch.setattr(
        inferer,
        "cleanup",
        lambda: setattr(inferer, "pred_dl", None),
    )

    case_records = inferer.preprocess_to_workspace(images, tmp_path, chunksize=2)

    progress = FakeProgress.instances[-1]
    assert case_records == images
    assert progress.total == len(images)
    assert progress.desc == "Stage 1: nifti -> jpg"
    assert progress.updates == [2, 2, 1]


def test_regionbounded_preprocess_to_workspace_progress_tracks_cases(
    monkeypatch, tmp_path
):
    FakeProgress.instances.clear()
    monkeypatch.setattr(localiserinferer_module, "tqdm", FakeProgress)

    inferer = regionbounded.LocaliserInfererPT_RB.__new__(
        regionbounded.LocaliserInfererPT_RB
    )
    inferer.bs = 2

    images = [Path(f"/tmp/case_{index:03d}.nii.gz") for index in range(1, 6)]

    monkeypatch.setattr(inferer, "write_workspace_args", lambda workspace, data: None)

    def prepare_data(data_chunk):
        inferer.pred_dl = [list(data_chunk)]

    monkeypatch.setattr(inferer, "prepare_data", prepare_data)
    monkeypatch.setattr(inferer, "preprocess_case", lambda case: {"case": case})
    monkeypatch.setattr(
        inferer,
        "export_case",
        lambda processed, workspace: processed["case"],
    )
    monkeypatch.setattr(
        inferer,
        "cleanup",
        lambda: setattr(inferer, "pred_dl", None),
    )

    case_records = inferer.preprocess_to_workspace(images, tmp_path, chunksize=2)

    progress = FakeProgress.instances[-1]
    assert case_records == images
    assert progress.total == len(images)
    assert progress.desc == "Stage 1: nifti -> jpg"
    assert progress.updates == [2, 2, 1]


def test_predict_from_workspace_progress_tracks_cases(monkeypatch):
    FakeProgress.instances.clear()
    monkeypatch.setattr(localiserinferer_module, "tqdm", FakeProgress)

    inferer = LocaliserInferer.__new__(LocaliserInferer)
    inferer.bs = 2
    inferer.save_jpg = False
    inferer.fabric_device = "cpu"
    inferer.mem_quota = 0.5

    class FabricStub:
        def autocast(self):
            return nullcontext()

    inferer.fabric = FabricStub()
    inferer.model = (
        lambda image_batch_device, verbose=False: [
            torch.tensor(index) for index in range(image_batch_device.shape[0])
        ]
    )

    monkeypatch.setattr(
        inferer,
        "load_case_original",
        lambda source: torch.zeros(1, 2, 2, 2),
    )
    monkeypatch.setattr(
        inferer,
        "load_projection_jpg",
        lambda jpg_path: torch.zeros(3, 4, 4),
    )
    monkeypatch.setattr(
        inferer,
        "package_preds",
        lambda batch: [
            {"source": batch["projection_meta"][case_index * 2]["filename_or_obj"]}
            for case_index in range(len(batch["image_orig"]))
        ],
    )
    monkeypatch.setattr(inferer, "combine_bboxes", lambda out: out)
    monkeypatch.setattr(inferer, "postprocess", lambda out: out)
    monkeypatch.setattr(inferer, "save_bboxes_final", lambda out: None)
    monkeypatch.setattr(inferer, "system_mem_remaining", lambda: 1.0)
    monkeypatch.setattr(inferer, "delete_image_orig", lambda outputs: None)

    case_records = []
    for index in range(1, 6):
        source = Path(f"/tmp/case_{index:03d}.nii.gz")
        case_records.append(
            {
                "source": source,
                "projections": [
                    {
                        "jpg_path": Path(f"/tmp/case_{index:03d}_lat.jpg"),
                        "meta": {"filename_or_obj": str(source)},
                    },
                    {
                        "jpg_path": Path(f"/tmp/case_{index:03d}_ap.jpg"),
                        "meta": {"filename_or_obj": str(source)},
                    },
                ],
            }
        )

    outputs = inferer.predict_from_workspace(case_records)

    progress = FakeProgress.instances[-1]
    assert outputs == [{"source": str(case["source"])} for case in case_records]
    assert progress.total == len(case_records)
    assert progress.desc == "Stage 2: jpg inference"
    assert progress.updates == [2, 2, 1]
