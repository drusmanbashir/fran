"""Training router — launch and monitor fran training jobs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from agent.webapp.api import jobs
from agent.webapp.api.schemas import JobInfo, TrainingRequest

router = APIRouter(prefix="/training", tags=["training"])

_TRAIN_SH = Path(__file__).parents[5] / "fran" / "run" / "training" / "p_train.sh"


@router.post("/", response_model=JobInfo)
async def launch_training(req: TrainingRequest) -> JobInfo:
    """Launch a training run for *project* and return a job handle."""
    script = _TRAIN_SH
    if not script.exists():
        found = shutil.which("p_train.sh")
        if found is None:
            raise HTTPException(status_code=500, detail=f"p_train.sh not found at {script}")
        script = Path(found)

    cmd: List[str] = ["bash", str(script), "--project", req.project]
    if req.config_overrides:
        cmd += ["--overrides", json.dumps(req.config_overrides)]
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
