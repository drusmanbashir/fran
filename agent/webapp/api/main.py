"""FastAPI application entrypoint for the fran webapp service.

Start with:
    uvicorn agent.webapp.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>FRAN Webapp</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f4f7f4;
        --panel: #ffffff;
        --text: #163022;
        --muted: #56705f;
        --accent: #1d6b43;
        --border: #d7e3d9;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Segoe UI", sans-serif;
        background: linear-gradient(180deg, #eef5ef 0%, var(--bg) 100%);
        color: var(--text);
      }
      main {
        max-width: 1100px;
        margin: 0 auto;
        padding: 48px 24px 64px;
      }
      h1 {
        margin: 0 0 12px;
        font-size: clamp(2.4rem, 4vw, 4rem);
      }
      p {
        margin: 0;
        max-width: 720px;
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.6;
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 18px;
        margin-top: 32px;
      }
      .card {
        display: block;
        padding: 22px;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: var(--panel);
        color: inherit;
        text-decoration: none;
        box-shadow: 0 12px 30px rgba(20, 48, 34, 0.08);
      }
      .card span {
        display: inline-block;
        margin-bottom: 10px;
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .card strong {
        display: block;
        margin-bottom: 8px;
        font-size: 1.15rem;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>FRAN</h1>
      <p>Operational entrypoint for FRAN job control, API exploration, and service documentation.</p>
      <section class="grid">
        <a class="card" href="/hpc/jobs?scope=hpc">
          <span>Jobs</span>
          <strong>HPC Jobs Dashboard</strong>
          Inspect, poll, cancel, and resubmit tracked HPC jobs.
        </a>
        <a class="card" href="/docs">
          <span>Docs</span>
          <strong>Interactive API Docs</strong>
          Browse endpoints and execute requests through Swagger UI.
        </a>
        <a class="card" href="/openapi.json">
          <span>API</span>
          <strong>OpenAPI Schema</strong>
          Consume the raw schema for generated clients and integration checks.
        </a>
      </section>
    </main>
  </body>
</html>
"""
