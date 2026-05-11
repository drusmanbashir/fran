from fran.inference import base as base_module
from fran.inference import pt_inferers as pt_module


def test_process_data_sublist_returns_last_processed_batch(monkeypatch):
    inferer = base_module.BaseInferer.__new__(base_module.BaseInferer)
    inferer.safe_mode = False
    seen_losses = []

    monkeypatch.setattr(inferer, "load_images", lambda data_sublist: data_sublist)
    monkeypatch.setattr(inferer, "prepare_data", lambda data, collate_fn=None: None)
    monkeypatch.setattr(
        inferer, "create_and_set_postprocess_transforms", lambda: None
    )
    monkeypatch.setattr(
        inferer,
        "predict",
        lambda: iter([{"pred": "first"}, {"pred": "last"}]),
    )
    monkeypatch.setattr(inferer, "postprocess", lambda batch: batch["pred"])
    monkeypatch.setattr(inferer, "compute_loss", lambda batch: seen_losses.append(batch))

    output = inferer.process_data_sublist(["case_001"])

    assert output == "last"
    assert seen_losses == ["first", "last"]


def test_process_data_sublist_keeps_last_output_in_safe_mode(monkeypatch):
    inferer = base_module.BaseInferer.__new__(base_module.BaseInferer)
    inferer.safe_mode = True
    reset_calls = []

    monkeypatch.setattr(inferer, "load_images", lambda data_sublist: data_sublist)
    monkeypatch.setattr(inferer, "prepare_data", lambda data, collate_fn=None: None)
    monkeypatch.setattr(
        inferer, "create_and_set_postprocess_transforms", lambda: None
    )
    monkeypatch.setattr(inferer, "predict", lambda: iter([{"pred": "last"}]))
    monkeypatch.setattr(inferer, "postprocess", lambda batch: batch["pred"])
    monkeypatch.setattr(inferer, "compute_loss", lambda batch: None)
    monkeypatch.setattr(inferer, "reset", lambda: reset_calls.append(True))

    output = inferer.process_data_sublist(["case_001"])

    assert output == "last"
    assert reset_calls == [True]


def test_run_returns_last_processed_sublist(monkeypatch):
    inferer = base_module.BaseInferer.__new__(base_module.BaseInferer)

    monkeypatch.setattr(inferer, "setup", lambda: None)
    monkeypatch.setattr(inferer, "maybe_filter_images", lambda data, overwrite=False: data)
    monkeypatch.setattr(
        base_module,
        "chunks",
        lambda data, n_sized_chunks=None, n_chunks=None: [["case_001"], ["case_002"], ["case_003"]],
    )
    monkeypatch.setattr(
        inferer,
        "process_data_sublist",
        lambda imgs_sublist: imgs_sublist[0],
    )

    output = inferer.run(["case_001", "case_002", "case_003"], chunksize=1)

    assert output == "case_003"


def test_pt_process_imgs_sublist_returns_last_processed_batch(monkeypatch):
    inferer = pt_module.BaseInfererPT.__new__(pt_module.BaseInfererPT)
    inferer.safe_mode = False
    monkeypatch.setattr(inferer, "load_images_and_gts", lambda imgs_gt_sublist: imgs_gt_sublist)
    monkeypatch.setattr(inferer, "prepare_data", lambda data: None)
    monkeypatch.setattr(inferer, "create_and_set_postprocess_transforms", lambda: None)
    monkeypatch.setattr(inferer, "predict", lambda: iter([{"pred": "first"}, {"pred": "last"}]))
    monkeypatch.setattr(inferer, "compute_loss", lambda batch: batch)
    monkeypatch.setattr(inferer, "postprocess", lambda batch: batch["pred"])
    assert inferer.process_imgs_sublist(["case_001"], gt_fldr=None) == "last"


def test_pt_run_returns_last_processed_sublist_and_losses(monkeypatch):
    inferer = pt_module.BaseInfererPT.__new__(pt_module.BaseInfererPT)
    monkeypatch.setattr(inferer, "setup", lambda: None)
    monkeypatch.setattr(inferer, "maybe_filter_images", lambda imgs, overwrite=False: imgs)
    monkeypatch.setattr(
        pt_module,
        "chunks",
        lambda imgs, n_sized_chunks=None, n_chunks=None: [["case_001"], ["case_002"]],
    )
    def process_imgs_sublist(imgs_sublist, gt_fldr=None):
        inferer.losses.append({"case_id": imgs_sublist[0]})
        return imgs_sublist[0]
    monkeypatch.setattr(inferer, "process_imgs_sublist", process_imgs_sublist)
    assert inferer.run(["case_001", "case_002"], None, chunksize=1) == (
        "case_002",
        [{"case_id": "case_001"}, {"case_id": "case_002"}],
    )
