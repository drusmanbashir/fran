"""Subprocess manager for long-running jobs (preprocessing, training, inference, rsync).

Jobs are tracked in an in-process dict keyed by a UUID.  Swap this out for
Celery / RQ when you need persistence across restarts or multi-worker deployments.
"""

from __future__ import annotations

import asyncio
import subprocess
import uuid
from collections import deque
from typing import Deque, Dict, List, Optional

from agent.webapp.api.schemas import JobInfo, JobStatus

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_TAIL_LINES = 50  # how many stdout/stderr lines to keep per job

_jobs: Dict[str, "_Job"] = {}


class _Job:
    def __init__(self, job_id: str, cmd: List[str]) -> None:
        self.job_id = job_id
        self.cmd = cmd
        self.status: JobStatus = JobStatus.pending
        self.pid: Optional[int] = None
        self.return_code: Optional[int] = None
        self._stdout: Deque[str] = deque(maxlen=_TAIL_LINES)
        self._stderr: Deque[str] = deque(maxlen=_TAIL_LINES)
        self._proc: Optional[asyncio.subprocess.Process] = None

    def to_info(self) -> JobInfo:
        return JobInfo(
            job_id=self.job_id,
            status=self.status,
            pid=self.pid,
            return_code=self.return_code,
            stdout_tail="\n".join(self._stdout) or None,
            stderr_tail="\n".join(self._stderr) or None,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def submit(cmd: List[str]) -> str:
    """Launch *cmd* as a background subprocess and return its job_id."""
    job_id = str(uuid.uuid4())
    job = _Job(job_id, cmd)
    _jobs[job_id] = job
    asyncio.create_task(_run(job))
    return job_id


def get(job_id: str) -> Optional[JobInfo]:
    """Return current status for *job_id*, or ``None`` if unknown."""
    job = _jobs.get(job_id)
    return job.to_info() if job else None


def list_all() -> List[JobInfo]:
    return [j.to_info() for j in _jobs.values()]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _run(job: _Job) -> None:
    job.status = JobStatus.running
    try:
        proc = await asyncio.create_subprocess_exec(
            *job.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        job._proc = proc
        job.pid = proc.pid

        stdout_bytes, stderr_bytes = await proc.communicate()

        for line in (stdout_bytes or b"").decode(errors="replace").splitlines():
            job._stdout.append(line)
        for line in (stderr_bytes or b"").decode(errors="replace").splitlines():
            job._stderr.append(line)

        job.return_code = proc.returncode
        job.status = JobStatus.done if proc.returncode == 0 else JobStatus.failed
    except Exception as exc:  # noqa: BLE001
        job._stderr.append(str(exc))
        job.status = JobStatus.failed
