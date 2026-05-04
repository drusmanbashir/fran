"""Projects router — list configured fran projects and their status."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from fastapi import APIRouter, HTTPException

from agent.webapp.api.schemas import ProjectListResponse, ProjectSummary

router = APIRouter(prefix="/projects", tags=["projects"])

# Default location of the datasets config used across fran.
_DATASETS_CONF = Path("/s/fran_storage/conf/datasets.yaml")


def _load_datasets_conf(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


@router.get("/", response_model=ProjectListResponse)
async def list_projects(conf: str = str(_DATASETS_CONF)) -> ProjectListResponse:
    """Return all projects registered in *datasets.yaml*."""
    data = _load_datasets_conf(Path(conf))
    projects: List[ProjectSummary] = []
    for name, info in data.items():
        if isinstance(info, dict):
            projects.append(ProjectSummary(name=name, path=info.get("root", ""), config=info))
        else:
            projects.append(ProjectSummary(name=name, path=str(info)))
    return ProjectListResponse(projects=projects)


@router.get("/{project}", response_model=ProjectSummary)
async def get_project(project: str, conf: str = str(_DATASETS_CONF)) -> ProjectSummary:
    """Return details for a single project."""
    data = _load_datasets_conf(Path(conf))
    if project not in data:
        raise HTTPException(status_code=404, detail=f"Project '{project}' not found")
    info = data[project]
    if isinstance(info, dict):
        return ProjectSummary(name=project, path=info.get("root", ""), config=info)
    return ProjectSummary(name=project, path=str(info))
