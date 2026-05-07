"""FastAPI application entrypoint for the fran webapp service.

Start with:
    uvicorn agent.webapp.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from agent.webapp.api.routers import hpc, inference, jobs_dashboard, preprocess, projects, training

app = FastAPI(
    title="fran webapp API",
    description="REST wrapper around fran pipeline operations and HPC tooling.",
    version="0.1.0",
)

app.include_router(projects.router)
app.include_router(preprocess.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(jobs_dashboard.router)
app.include_router(hpc.router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}


@app.on_event("shutdown")
async def shutdown_jobs_dashboard() -> None:
    jobs_dashboard.stop_backends()


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/hpc/jobs", status_code=307)
