from __future__ import annotations

import csv
from pathlib import Path

from fastapi.testclient import TestClient

from agent.webapp.api import main
from agent.webapp.api.routers import jobs_dashboard


def _write_registry_row(root: Path, job_id: str, state: str, input_method: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "job_registry.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                job_id,
                "2026-05-07T10:00:00+01:00",
                "sbatch.sh",
                "demo-job",
                "remote.sh",
                state,
                "-",
                "-",
                "2026-05-07T10:00:01+01:00",
                input_method,
                "-",
            ]
        )


def test_root_lands_on_home_page() -> None:
    client = TestClient(main.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "FRAN" in response.text
    assert 'href="/hpc/jobs?scope=hpc"' in response.text
    assert 'href="/docs"' in response.text
    assert 'href="/openapi.json"' in response.text


def test_jobs_page_proxies_hpc_and_local_scopes(tmp_path: Path, monkeypatch) -> None:
    hpc_root = tmp_path / "hpc_logs"
    local_root = tmp_path / "local_logs"
    _write_registry_row(hpc_root, "12345", "RUNNING", "hpc_submit_poll_fetch")
    _write_registry_row(local_root, "local-abc", "RUNNING", "local_train_retry")
    monkeypatch.setenv("FRAN_HPC_JOBS_ROOT", str(hpc_root))
    monkeypatch.setenv("FRAN_LOCAL_JOBS_ROOT", str(local_root))
    jobs_dashboard.stop_backends()

    client = TestClient(main.app)

    hpc_response = client.get("/hpc/jobs?scope=hpc")
    assert hpc_response.status_code == 200
    assert "HPC Jobs Dashboard" in hpc_response.text
    assert 'href="/"' in hpc_response.text
    assert 'href="/docs"' in hpc_response.text
    assert 'href="/openapi.json"' in hpc_response.text
    assert "12345" in hpc_response.text
    assert "local-abc" not in hpc_response.text

    local_response = client.get("/hpc/jobs?scope=local")
    assert local_response.status_code == 200
    assert 'href="/hpc/jobs?scope=local"' in local_response.text
    assert "local-abc" in local_response.text
    assert "12345" not in local_response.text

    jobs_dashboard.stop_backends()
