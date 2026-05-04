"""Preprocessing router — launch fran/run/preproc jobs."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from agent.webapp.api import jobs
from agent.webapp.api.schemas import JobInfo, PreprocessRequest

router = APIRouter(prefix="/preprocess", tags=["preprocess"])

# Entry-point script (relative to repo root, resolved at runtime).
_PREPROC_SH = Path(__file__).parents[5] / "fran" / "run" / "preproc" / "preproc.sh"


@router.post("/", response_model=JobInfo)
async def run_preprocess(req: PreprocessRequest) -> JobInfo:
    """Launch a preprocessing job for *project* and return a job handle."""
    script = _PREPROC_SH
    if not script.exists():
        # Fall back to PATH lookup
        found = shutil.which("preproc.sh")
        if found is None:
            raise HTTPException(status_code=500, detail=f"preproc.sh not found at {script}")
        script = Path(found)

    cmd = ["bash", str(script), "--project", req.project]
    if req.overwrite:
        cmd.append("--overwrite")
    if req.extra_args:
        cmd.extend(req.extra_args)

    job_id = await jobs.runner.submit(cmd)
    info = jobs.runner.get(job_id)
    assert info is not None
    return info


@router.get("/{job_id}", response_model=JobInfo)
async def get_job(job_id: str) -> JobInfo:
    info = jobs.runner.get(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return info
