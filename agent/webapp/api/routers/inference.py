"""Inference router — run fran sliding-window / cascade inference."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from agent.webapp.api import jobs
from agent.webapp.api.schemas import InferenceRequest, JobInfo

router = APIRouter(prefix="/inference", tags=["inference"])

_VALID_MODES = {"source", "whole", "lbd"}


@router.post("/", response_model=JobInfo)
async def run_inference(req: InferenceRequest) -> JobInfo:
    """Submit an inference job and return a job handle."""
    if req.mode not in _VALID_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"mode must be one of {sorted(_VALID_MODES)}, got '{req.mode}'",
        )

    # Use the same Python interpreter that is running the server so fran is on
    # sys.path without needing an extra activation step.
    python = sys.executable
    script = str(Path(__file__).parents[5] / "fran" / "run" / "inference" / "base.py")
    if not Path(script).exists():
        found = shutil.which("fran-infer")
        if found is None:
            raise HTTPException(status_code=500, detail="fran inference entry-point not found")
        script = found
        cmd = [script]
    else:
        cmd = [python, script]

    cmd += [
        "--project", req.project,
        "--run", req.run_id,
        "--input", req.input_folder,
        "--output", req.output_folder,
        "--mode", req.mode,
    ]
    if req.extra_args:
        cmd.extend(req.extra_args)

    job_id = await jobs.runner.submit(cmd)
    info = jobs.runner.get(job_id)
    assert info is not None
    return info


@router.get("/", response_model=List[JobInfo])
async def list_jobs() -> List[JobInfo]:
    return jobs.runner.list_all()


@router.get("/{job_id}", response_model=JobInfo)
async def get_job(job_id: str) -> JobInfo:
    info = jobs.runner.get(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return info
