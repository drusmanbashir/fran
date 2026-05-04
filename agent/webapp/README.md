# fran webapp

A lightweight FastAPI service that wraps `fran` pipeline operations and
`agent/hpc/` shell tools behind a REST API.

## Layout

```
agent/
├── hpc/
│   └── upload_dataset_rsync.sh   # HPC dataset-upload helper
└── webapp/
    ├── api/
    │   ├── main.py               # FastAPI app — mounts all routers
    │   ├── schemas.py            # Pydantic request/response models
    │   ├── routers/
    │   │   ├── projects.py       # list / inspect configured projects
    │   │   ├── preprocess.py     # launch preprocessing jobs
    │   │   ├── training.py       # launch / monitor training runs
    │   │   ├── inference.py      # run sliding-window / cascade inference
    │   │   └── hpc.py            # dataset upload via agent/hpc/ scripts
    │   └── jobs/
    │       └── runner.py         # async subprocess manager
    ├── frontend/                 # browser GUI (reserved)
    ├── requirements.txt
    └── README.md
```

## Quick start

```bash
# Install service dependencies (isolated from fran core)
pip install -r agent/webapp/requirements.txt

# Start the server (from the repo root so fran is importable)
uvicorn agent.webapp.api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs are available at <http://localhost:8000/docs>.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/projects/` | List all projects from `datasets.yaml` |
| GET | `/projects/{project}` | Single project details |
| POST | `/preprocess/` | Start a preprocessing job |
| GET | `/preprocess/{job_id}` | Check preprocessing job status |
| POST | `/training/` | Start a training run |
| GET | `/training/` | List all training jobs |
| GET | `/training/{job_id}` | Check training job status |
| POST | `/inference/` | Start an inference job |
| GET | `/inference/` | List all inference jobs |
| GET | `/inference/{job_id}` | Check inference job status |
| POST | `/hpc/upload` | Upload dataset to HPC cold storage |
| GET | `/hpc/jobs` | List all HPC jobs |
| GET | `/hpc/jobs/{job_id}` | Check HPC job status |

## Job lifecycle

Long-running processes (GPU training, rsync uploads) are spawned as background
asyncio subprocesses via `api/jobs/runner.py`.  Jobs are tracked in-process by
UUID.  Swap `runner.py` for Celery / RQ if you need persistence across restarts
or multi-worker deployments.

## Frontend

`frontend/` is a reserved slot.  Drop a React build or plain HTML files there
and configure uvicorn / nginx to serve them from `/`.
