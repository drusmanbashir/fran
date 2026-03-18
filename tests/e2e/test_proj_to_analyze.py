import os

import pytest
from fran.tests.e2e.proj_to_analyze import run_proj_to_analyze


@pytest.mark.e2e
def test_proj_to_analyze_setup_only():
    if os.environ.get("FRAN_E2E", "0") != "1":
        pytest.skip("Set FRAN_E2E=1 to run e2e pipeline test.")
    report = run_proj_to_analyze(run_analyze=False, num_processes=1, debug=False)
    assert report.n_fail == 0, f"Plan failures: {report.n_fail}"
    assert len(report.errors) == 0, f"Project/config errors: {report.errors}"
