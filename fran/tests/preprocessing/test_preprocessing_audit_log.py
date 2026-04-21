from pathlib import Path

import pandas as pd

from fran.preprocessing.preprocessor import Preprocessor


def _make_preprocessor(tmp_path):
    pre = Preprocessor.__new__(Preprocessor)
    pre.output_folder = tmp_path
    pre.mini_dfs = [
        pd.DataFrame(
            [
                {"case_id": "case_ok", "image": Path("/tmp/case_ok_img.pt"), "lm": Path("/tmp/case_ok_lm.pt")},
                {"case_id": "case_warn", "image": Path("/tmp/case_warn_img.pt"), "lm": Path("/tmp/case_warn_lm.pt")},
                {"case_id": "case_err", "image": Path("/tmp/case_err_img.pt"), "lm": Path("/tmp/case_err_lm.pt")},
            ]
        )
    ]
    return pre


def test_preprocessing_audit_log_status_normalization(tmp_path):
    pre = _make_preprocessor(tmp_path)
    results = [
        [
            {"case_id": "case_ok", "ok": True, "shape": [1, 2, 3, 4], "_preprocess_events": []},
            {
                "case_id": "case_warn",
                "ok": True,
                "shape": [1, 2, 3, 4],
                "_preprocess_events": [
                    {"error_type": "CropByYolo", "error_message": "fg mismatch first"},
                    {"error_type": "CropByYolo", "error_message": "fg mismatch expanded"},
                ],
            },
            {
                "case_id": "case_err",
                "ok": False,
                "err": "RuntimeError('boom')",
                "_preprocess_error": {
                    "error_type": "RuntimeError",
                    "error_message": "boom",
                    "traceback": "traceback lines",
                },
            },
        ]
    ]

    rows = pre.build_preprocessing_log_rows(results)
    assert [row["status"] for row in rows] == ["OK", "WARNING", "ERROR"]
    assert rows[1]["error_type"] == "CropByYolo"
    assert rows[1]["error_message"] == "fg mismatch first; fg mismatch expanded"
    assert rows[2]["error_type"] == "RuntimeError"
    assert rows[2]["traceback"] == "traceback lines"

    pre.write_preprocessing_log(results)
    df = pd.read_csv(tmp_path / "preprocessing_log.csv")
    assert list(df.columns) == Preprocessor.PREPROCESS_LOG_COLUMNS
    assert df["status"].tolist() == ["OK", "WARNING", "ERROR"]


def test_flatten_results_strips_preprocess_audit_columns(tmp_path):
    pre = _make_preprocessor(tmp_path)
    df = pre.flatten_results(
        [
            [
                {"case_id": "case_ok", "ok": True, "_preprocess_events": []},
                {
                    "case_id": "case_err",
                    "ok": False,
                    "_preprocess_error": {"error_type": "RuntimeError"},
                },
            ]
        ]
    )
    assert "_preprocess_events" not in df.columns
    assert "_preprocess_error" not in df.columns
    assert "case_id" in df.columns
