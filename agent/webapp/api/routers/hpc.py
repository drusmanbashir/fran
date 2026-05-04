"""HPC router — dataset upload and other agent/hpc/ shell tools."""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from agent.webapp.api import jobs
from agent.webapp.api.schemas import DatasetUploadRequest, JobInfo

router = APIRouter(prefix="/hpc", tags=["hpc"])

# Resolve agent/hpc/ relative to this file's location inside agent/webapp/api/routers/
_HPC_DIR = Path(__file__).parents[3] / "hpc"
_UPLOAD_SCRIPT = _HPC_DIR / "upload_dataset_rsync.sh"


@router.post("/upload", response_model=JobInfo)
async def upload_dataset(req: DatasetUploadRequest) -> JobInfo:
    """Upload a local dataset to HPC cold storage via rsync."""
    if not _UPLOAD_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Upload script not found: {_UPLOAD_SCRIPT}",
        )

    cmd: List[str] = [
        "bash", str(_UPLOAD_SCRIPT),
        "--source", req.source,
        "--dataset", req.dataset_rel,
        "--yes",  # non-interactive when called via API
    ]
    if req.remote:
        cmd += ["--remote", req.remote]
    if req.cold_storage:
        cmd += ["--cold-storage", req.cold_storage]
    if req.dry_run:
        cmd.append("--dry-run")

    job_id = await jobs.runner.submit(cmd)
    info = jobs.runner.get(job_id)
    assert info is not None
    return info


@router.get("/jobs", response_model=List[JobInfo])
async def list_jobs() -> List[JobInfo]:
    return jobs.runner.list_all()


@router.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: str) -> JobInfo:
    info = jobs.runner.get(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return info
