"""Pydantic request/response models for the fran webapp API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    pid: Optional[int] = None
    return_code: Optional[int] = None
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


class ProjectSummary(BaseModel):
    name: str
    path: str
    config: Optional[Dict[str, Any]] = None


class ProjectListResponse(BaseModel):
    projects: List[ProjectSummary]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class PreprocessRequest(BaseModel):
    project: str = Field(..., description="Project name (must exist under fran_storage)")
    overwrite: bool = False
    extra_args: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainingRequest(BaseModel):
    project: str
    config_overrides: Optional[Dict[str, Any]] = None
    extra_args: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class InferenceRequest(BaseModel):
    project: str
    run_id: str
    input_folder: str
    output_folder: str
    mode: str = Field("source", description="Inference mode: source | whole | lbd")
    extra_args: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# HPC
# ---------------------------------------------------------------------------


class DatasetUploadRequest(BaseModel):
    source: str = Field(..., description="Local path to upload")
    dataset_rel: str = Field(..., description="Relative path under datasets/ on remote")
    remote: Optional[str] = Field(None, description="user@host (defaults to script default)")
    cold_storage: Optional[str] = Field(None, description="Remote cold-storage root")
    dry_run: bool = False
