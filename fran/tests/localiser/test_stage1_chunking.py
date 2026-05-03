from pathlib import Path

from fran.preprocessing import regionbounded
from localiser.inference.localiserinferer import LocaliserInferer


def test_localiser_preprocess_to_workspace_rebuilds_loader_per_source_chunk(
    monkeypatch, tmp_path
):
    inferer = LocaliserInferer.__new__(LocaliserInferer)
    inferer.bs = 2

    images = [Path(f"/tmp/case_{index:03d}.nii.gz") for index in range(1, 6)]
    prepare_calls = []
    cleanup_calls = []

    monkeypatch.setattr(inferer, "write_workspace_args", lambda workspace, data: None)

    def prepare_data(data_chunk):
        chunk = list(data_chunk)
        prepare_calls.append(chunk)
        inferer.pred_dl = [chunk]

    def cleanup():
        cleanup_calls.append(True)
        inferer.pred_dl = None

    monkeypatch.setattr(inferer, "prepare_data", prepare_data)
    monkeypatch.setattr(inferer, "preprocess_case", lambda case: {"case": case})
    monkeypatch.setattr(
        inferer,
        "export_case",
        lambda processed, workspace: processed["case"],
    )
    monkeypatch.setattr(inferer, "cleanup", cleanup)

    case_records = inferer.preprocess_to_workspace(images, tmp_path, chunksize=2)

    assert prepare_calls == [
        images[0:2],
        images[2:4],
        images[4:5],
    ]
    assert len(cleanup_calls) == 3
    assert case_records == images
    assert inferer.pred_dl is None


def test_regionbounded_localiser_preprocess_to_workspace_rebuilds_loader_per_chunk(
    monkeypatch, tmp_path
):
    inferer = regionbounded.LocaliserInfererPT_RB.__new__(
        regionbounded.LocaliserInfererPT_RB
    )
    inferer.bs = 2

    images = [Path(f"/tmp/case_{index:03d}.nii.gz") for index in range(1, 6)]
    prepare_calls = []
    cleanup_calls = []

    monkeypatch.setattr(inferer, "write_workspace_args", lambda workspace, data: None)

    def prepare_data(data_chunk):
        chunk = list(data_chunk)
        prepare_calls.append(chunk)
        inferer.pred_dl = [chunk]

    def cleanup():
        cleanup_calls.append(True)
        inferer.pred_dl = None

    monkeypatch.setattr(inferer, "prepare_data", prepare_data)
    monkeypatch.setattr(inferer, "preprocess_case", lambda case: {"case": case})
    monkeypatch.setattr(
        inferer,
        "export_case",
        lambda processed, workspace: processed["case"],
    )
    monkeypatch.setattr(inferer, "cleanup", cleanup)

    case_records = inferer.preprocess_to_workspace(images, tmp_path, chunksize=2)

    assert prepare_calls == [
        images[0:2],
        images[2:4],
        images[4:5],
    ]
    assert len(cleanup_calls) == 3
    assert case_records == images
