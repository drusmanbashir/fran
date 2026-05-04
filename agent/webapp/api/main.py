"""FastAPI application entrypoint for the fran webapp service.

Start with:
    uvicorn agent.webapp.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI

from agent.webapp.api.routers import hpc, inference, preprocess, projects, training

app = FastAPI(
    title="fran webapp API",
    description="REST wrapper around fran pipeline operations and HPC tooling.",
    version="0.1.0",
)

app.include_router(projects.router)
app.include_router(preprocess.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(hpc.router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}
