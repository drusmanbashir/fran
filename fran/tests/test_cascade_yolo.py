import json
from pathlib import Path

import pytest
from fran.inference import cascade_yolo


def _serialized_bbox(width, ap, height, cls=1):
    return {
        "lat": {
            "orig_shape": [256, 256],
            "xyxy": [[ap[0] * 256, height[0] * 256, ap[1] * 256, height[1] * 256]],
            "conf": [0.8],
            "cls": [cls],
            "track_id": None,
            "meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]},
            "classes": [cls],
        },
        "ap": {
            "orig_shape": [256, 256],
            "xyxy": [[width[0] * 256, height[0] * 256, width[1] * 256, height[1] * 256]],
            "conf": [0.9],
            "cls": [cls],
            "track_id": None,
            "meta": {"letterbox_padded": [[0, 0], [0, 0], [0, 0]]},
            "classes": [cls],
        },
    }


def _fake_output(image, bbox):
    image = Path(image)
    meta1 = {"filename_or_obj": image, "projection_key": "image1"}
    meta2 = {"filename_or_obj": image, "projection_key": "image2"}
    return {
        "pred": [None, None],
        "pred_image1": None,
        "pred_image2": None,
        "image": None,
        "image1": None,
        "image2": None,
        "image_orig": None,
        "projection_meta": [meta1, meta2],
        "projection_meta_image1": meta1,
        "projection_meta_image2": meta2,
        "bboxes_final": bbox,
        "bbox_messages": {"lat": "cached", "ap": "cached"},
    }


def _inferer(tmp_path):
    inferer = cascade_yolo.CachedLocaliserInfererPT.__new__(
        cascade_yolo.CachedLocaliserInfererPT
    )
    inferer.out_folder = tmp_path
    inferer.classes = []
    return inferer


def test_resolve_yolo_class_filter_prefers_localiser_labels_over_regions():
    yolo_specs = {"data": {"names": ["chest", "abdomen", "pelvis"]}}

    classes = cascade_yolo._resolve_yolo_class_filter(
        yolo_specs,
        localiser_labels=["abdomen", "pelvis"],
        localiser_regions="chest,abdomen,pelvis",
    )

    assert classes == [1, 2]


def test_region_classes_from_list_validates_and_matches_projection_names(monkeypatch):
    yolo_specs = {
        "data": {
            "names": [
                "abdomen_ap",
                "chest_ap",
                "neck_ap",
                "pelvis_ap",
                "abdomen_lat",
                "chest_lat",
                "neck_lat",
                "pelvis_lat",
            ]
        }
    }

    class FakeTSLRegions:
        def __init__(self):
            self.regions = ["abdomen", "chest", "neck", "pelvis"]

    monkeypatch.setattr(cascade_yolo, "TSLRegions", FakeTSLRegions)

    classes = cascade_yolo._region_classes(yolo_specs, ["abdomen", "pelvis"])

    assert classes == [0, 3, 4, 7]


def test_resolve_yolo_class_filter_accepts_numeric_label_selectors():
    yolo_specs = {"data": {"names": ["chest", "abdomen", "pelvis"]}}

    classes = cascade_yolo._resolve_yolo_class_filter(
        yolo_specs,
        localiser_labels=[1, "2"],
        localiser_regions="chest,abdomen,pelvis",
    )

    assert classes == [1, 2]


def test_cached_localiser_run_uses_cached_file_when_exists_and_overwrite_false(
    tmp_path, monkeypatch
):
    inferer = _inferer(tmp_path)
    images = [tmp_path / "case_001.pt", tmp_path / "case_002.pt"]
    bbox1 = _serialized_bbox((0.1, 0.5), (0.2, 0.6), (0.3, 0.7))
    bbox2 = _serialized_bbox((0.0, 1.0), (0.1, 0.9), (0.2, 0.8))
    (tmp_path / "case_001.json").write_text(json.dumps(bbox1))
    (tmp_path / "case_002.json").write_text(json.dumps(bbox2))
    seen = []

    def fake_run(self, images, chunksize=None, overwrite=False):
        seen.append((list(images), overwrite))
        return []

    monkeypatch.setattr(cascade_yolo.LocaliserInfererPT, "run", fake_run)

    outputs = inferer.run(images, overwrite=False)

    assert seen == [([], False)]
    assert outputs == [_fake_output(images[0], bbox1), _fake_output(images[1], bbox2)]


def test_cached_localiser_run_uses_fresh_yolo_when_cache_missing_and_overwrite_false(
    tmp_path, monkeypatch
):
    inferer = _inferer(tmp_path)
    images = [tmp_path / "case_002.pt"]
    bbox_fresh = _serialized_bbox((0.4, 0.8), (0.0, 0.4), (0.1, 0.9))
    seen = []

    def fake_run(self, images, chunksize=None, overwrite=False):
        seen.append((list(images), overwrite))
        return [_fake_output(images[0], bbox_fresh)]

    monkeypatch.setattr(cascade_yolo.LocaliserInfererPT, "run", fake_run)

    outputs = inferer.run(images, overwrite=False)

    assert seen == [([images[0]], False)]
    assert outputs == [_fake_output(images[0], bbox_fresh)]


def test_cached_localiser_run_uses_fresh_yolo_when_cache_exists_and_overwrite_true(
    tmp_path, monkeypatch
):
    inferer = _inferer(tmp_path)
    images = [tmp_path / "case_001.pt"]
    bbox_cached = _serialized_bbox((0.1, 0.5), (0.2, 0.6), (0.3, 0.7))
    (tmp_path / "case_001.json").write_text(json.dumps(bbox_cached))
    seen = []
    expected = [_fake_output(images[0], _serialized_bbox((0.2, 0.4), (0.1, 0.9), (0.3, 0.6)))]

    def fake_run(self, images, chunksize=None, overwrite=False):
        seen.append((list(images), overwrite))
        return expected

    monkeypatch.setattr(cascade_yolo.LocaliserInfererPT, "run", fake_run)

    outputs = inferer.run(images, overwrite=True)

    assert seen == [([images[0]], True)]
    assert outputs == expected


def test_build_bbox_lookup_standardizes_serialized_bboxes():
    image = Path("/tmp/case_001.pt")
    bbox = _serialized_bbox((0.1, 0.5), (0.2, 0.6), (0.3, 0.8))
    by_name = cascade_yolo._build_bbox_lookup([_fake_output(image, bbox)])
    out = cascade_yolo._resolve_bbox(by_name, image)

    assert out["width"] == pytest.approx((0.1, 0.5))
    assert out["ap"] == pytest.approx((0.2, 0.6))
    assert out["height"] == pytest.approx((0.3, 0.8))


def test_dummy_apply_bboxes_crops_images_and_stores_bbox():
    D = cascade_yolo.Dummy()
    bbox = {"width": (0.1, 0.5), "ap": (0.2, 0.6), "height": (0.3, 0.7)}
    data = [{"image": "orig"}]

    class FakeCropper:
        def __call__(self, data):
            return {"image": ("cropped", data["bbox"])}

    D.cropper_yolo = FakeCropper()

    out = D.apply_bboxes(data, [bbox])

    assert out[0]["image"] == ("cropped", bbox)
    assert out[0]["bounding_box"] == bbox
    assert D.data_cropped == out


def test_dummy_inherits_cascade_inferer():
    assert issubclass(cascade_yolo.Dummy, cascade_yolo.CascadeInferer)


def test_dummy_resolve_yolo_classes_uses_localiser_labels():
    D = cascade_yolo.Dummy(
        localiser_labels=["abdomen", "pelvis"],
        localiser_regions="chest,abdomen,pelvis",
    )
    D.yolo_specs = {"data": {"names": ["chest", "abdomen", "pelvis"]}}

    classes = D._resolve_yolo_classes()

    assert classes == [1, 2]


def test_dummy_patch_prediction_uses_patch_inferer_flow():
    D = cascade_yolo.Dummy(run_p="RUN-1")
    seen = {}

    class FakePatchInferer:
        run_name = "RUN-1"

        def setup(self):
            seen["setup"] = True

        def prepare_data(self, data, collate_fn=None):
            seen["prepare_data"] = data
            seen["collate_fn"] = collate_fn

        def create_and_set_postprocess_transforms(self):
            seen["postprocess_tfms"] = True

        def predict(self):
            yield {"pred": "raw", "bounding_box": [{"x": 1}]}

        def postprocess(self, batch):
            return {"pred": "done", "bounding_box": batch["bounding_box"]}

    D.P = FakePatchInferer()
    D.data_cropped = [{"image": "cropped", "bounding_box": {"x": 1}}]

    out = D.patch_prediction()

    assert seen["setup"] is True
    assert seen["prepare_data"] == D.data_cropped
    assert seen["collate_fn"] is cascade_yolo.img_bbox_collated
    assert seen["postprocess_tfms"] is True
    assert out == {"RUN-1": [{"pred": "done", "bounding_box": [{"x": 1}]}]}
    assert D.pred_patches == out


def test_dummy_run_follows_parent_style_flow(monkeypatch):
    D = cascade_yolo.Dummy(run_p="RUN-1")
    seen = []
    data = [{"image": "loaded"}]
    bboxes = [{"width": (0.1, 0.5), "ap": (0.2, 0.6), "height": (0.3, 0.7)}]
    cropped = [{"image": "cropped", "bounding_box": bboxes[0]}]
    preds = {"RUN-1": [{"pred": "done"}]}

    def fake_load_images(imagefiles):
        seen.append(("load_images", imagefiles))
        return data

    def fake_setup_yolo_inferer():
        seen.append(("setup_yolo_inferer", None))
        return "Y"

    def fake_extract_fg_bboxes(data_in, overwrite=None):
        seen.append(("extract_fg_bboxes", data_in, overwrite))
        D.bboxes = bboxes
        return bboxes

    def fake_apply_bboxes(data_in, bboxes_in):
        seen.append(("apply_bboxes", data_in, bboxes_in))
        D.data_cropped = cropped
        return cropped

    def fake_setup_patch_inferer():
        seen.append(("setup_patch_inferer", None))
        D.P = object()
        return D.P

    def fake_patch_prediction(data_in):
        seen.append(("patch_prediction", data_in))
        return preds

    monkeypatch.setattr(D, "load_images", fake_load_images)
    monkeypatch.setattr(D, "setup_yolo_inferer", fake_setup_yolo_inferer)
    monkeypatch.setattr(D, "extract_fg_bboxes", fake_extract_fg_bboxes)
    monkeypatch.setattr(D, "apply_bboxes", fake_apply_bboxes)
    monkeypatch.setattr(D, "setup_patch_inferer", fake_setup_patch_inferer)
    monkeypatch.setattr(D, "patch_prediction", fake_patch_prediction)

    out = D.run(["case.pt"], overwrite=False)

    assert out == preds
    assert seen == [
        ("load_images", ["case.pt"]),
        ("setup_yolo_inferer", None),
        ("extract_fg_bboxes", data, False),
        ("apply_bboxes", data, bboxes),
        ("setup_patch_inferer", None),
        ("patch_prediction", cropped),
    ]
