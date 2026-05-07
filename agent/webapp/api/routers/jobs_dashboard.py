from __future__ import annotations

import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from http.server import ThreadingHTTPServer

AGENT_REPO = Path("/home/ub/code/agent")
if str(AGENT_REPO) not in sys.path:
    sys.path.insert(0, str(AGENT_REPO))

from agent.control_plane import local_registry as local_cp
from agent.hpc.cli import hpc_dashboard_web as hdash

router = APIRouter(tags=["jobs-dashboard"])

FRAN_JOBS_PREFIX = "/hpc/jobs"


def hpc_root() -> Path:
    override = os.environ.get("FRAN_HPC_JOBS_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    return hdash.JobRegistry().root


def local_root() -> Path:
    override = os.environ.get("FRAN_LOCAL_JOBS_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    return local_cp.logs_root()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass
class DashboardBackend:
    scope: str
    root: Path
    state_dir: Path
    httpd: ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None
    base_url: str = ""

    def start(self) -> None:
        if self.httpd is not None:
            return
        registry = hdash.JobRegistry(root=self.root)
        action_state = hdash.ActionState(self.state_dir)
        handler_cls = type(f"{self.scope.title()}DashboardHandler", (hdash.DashboardHandler,), {})
        handler_cls.registry = registry
        handler_cls.action_state = action_state
        handler_cls.state_dir = self.state_dir
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        host, port = self.httpd.server_address[:2]
        self.base_url = f"http://{host}:{port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.httpd is None:
            return
        self.httpd.shutdown()
        self.httpd.server_close()
        self.httpd = None
        self.thread = None
        self.base_url = ""


_BACKENDS: dict[str, DashboardBackend] = {}
_URL_OPENER = urllib.request.build_opener(_NoRedirect)


def start_backends() -> None:
    global _BACKENDS
    if _BACKENDS:
        return
    hpc_backend = DashboardBackend("hpc", hpc_root(), hpc_root() / "fran_jobs_dashboard_state")
    local_backend = DashboardBackend("local", local_root(), local_root() / "fran_jobs_dashboard_state")
    hpc_backend.start()
    local_backend.start()
    _BACKENDS = {"hpc": hpc_backend, "local": local_backend}


def stop_backends() -> None:
    global _BACKENDS
    for backend in _BACKENDS.values():
        backend.stop()
    _BACKENDS = {}


def backend_for_scope(scope: str) -> DashboardBackend:
    return _BACKENDS[hdash.normalized_source_scope(scope)]


def scope_from_request(request: Request, body: bytes | None = None) -> str:
    if request.method == "GET":
        return hdash.normalized_source_scope(request.query_params.get("scope", "hpc"))
    if body is None:
        return "hpc"
    payload = urllib.parse.parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return hdash.normalized_source_scope(payload.get("scope", ["hpc"])[0])


def proxied_path(path: str) -> str:
    if path == FRAN_JOBS_PREFIX or path == FRAN_JOBS_PREFIX + "/":
        return "/"
    suffix = path[len(FRAN_JOBS_PREFIX):]
    return suffix or "/"


def rewrite_location(location: str) -> str:
    if location == "/":
        return FRAN_JOBS_PREFIX
    if location.startswith("/?"):
        return f"{FRAN_JOBS_PREFIX}{location[1:]}"
    if location.startswith("/jobs/"):
        return f"{FRAN_JOBS_PREFIX}{location}"
    if location.startswith("/poll_") or location.startswith("/cancel_selected") or location.startswith("/resubmit_selected"):
        return f"{FRAN_JOBS_PREFIX}{location}"
    return location


def top_nav_html(scope: str) -> str:
    jobs_href = f"{FRAN_JOBS_PREFIX}?scope={scope}"
    return (
        '<nav style="margin:0 0 24px;padding:16px 20px;border:1px solid #d7e3d9;'
        'border-radius:16px;background:#f6faf6">'
        '<strong style="margin-right:18px">FRAN</strong>'
        f'<a href="/" style="margin-right:14px">Home</a>'
        f'<a href="{jobs_href}" style="margin-right:14px">Jobs</a>'
        '<a href="/docs" style="margin-right:14px">Docs</a>'
        '<a href="/openapi.json">API</a>'
        "</nav>"
    )


def rewrite_html(body: str, scope: str) -> str:
    replacements = {
        'href="/?': f'href="{FRAN_JOBS_PREFIX}?',
        'action="/"': f'action="{FRAN_JOBS_PREFIX}"',
        'href="/jobs/': f'href="{FRAN_JOBS_PREFIX}/jobs/',
        'action="/poll_selected"': f'action="{FRAN_JOBS_PREFIX}/poll_selected"',
        'action="/poll_all_active"': f'action="{FRAN_JOBS_PREFIX}/poll_all_active"',
        'action="/cancel_selected"': f'action="{FRAN_JOBS_PREFIX}/cancel_selected"',
        'action="/resubmit_selected"': f'action="{FRAN_JOBS_PREFIX}/resubmit_selected"',
        'formaction="/poll_all_active"': f'formaction="{FRAN_JOBS_PREFIX}/poll_all_active"',
        'formaction="/cancel_selected"': f'formaction="{FRAN_JOBS_PREFIX}/cancel_selected"',
        'formaction="/resubmit_selected"': f'formaction="{FRAN_JOBS_PREFIX}/resubmit_selected"',
    }
    for old, new in replacements.items():
        body = body.replace(old, new)
    nav = top_nav_html(scope)
    if "<body>" in body:
        body = body.replace("<body>", f"<body>{nav}", 1)
    else:
        body = nav + body
    return body


def proxy_response(method: str, scope: str, path: str, query: str, body: bytes | None, content_type: str | None) -> Response:
    backend = backend_for_scope(scope)
    url = f"{backend.base_url}{proxied_path(path)}"
    if query:
        url = f"{url}?{query}"
    req = urllib.request.Request(url, data=body, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with _URL_OPENER.open(req, timeout=30) as resp:
            payload = resp.read()
            ctype = resp.headers.get("Content-Type", "text/plain; charset=utf-8")
            if ctype.startswith("text/html"):
                return HTMLResponse(rewrite_html(payload.decode("utf-8", errors="replace"), scope), status_code=resp.status)
            if ctype.startswith("text/plain"):
                return PlainTextResponse(payload.decode("utf-8", errors="replace"), status_code=resp.status)
            return Response(content=payload, media_type=ctype, status_code=resp.status)
    except urllib.error.HTTPError as err:
        if err.code in {301, 302, 303, 307, 308}:
            return RedirectResponse(url=rewrite_location(err.headers["Location"]), status_code=err.code)
        payload = err.read()
        ctype = err.headers.get("Content-Type", "text/plain; charset=utf-8")
        if ctype.startswith("text/html"):
            return HTMLResponse(rewrite_html(payload.decode("utf-8", errors="replace"), scope), status_code=err.code)
        if ctype.startswith("text/plain"):
            return PlainTextResponse(payload.decode("utf-8", errors="replace"), status_code=err.code)
        return Response(content=payload, media_type=ctype, status_code=err.code)


@router.get("/hpc/jobs", response_class=HTMLResponse, include_in_schema=False)
async def jobs_root(request: Request) -> Response:
    start_backends()
    scope = scope_from_request(request)
    return proxy_response("GET", scope, request.url.path, request.url.query, None, None)


@router.get("/hpc/jobs/{subpath:path}", include_in_schema=False)
async def jobs_get(subpath: str, request: Request) -> Response:
    start_backends()
    scope = scope_from_request(request)
    return proxy_response("GET", scope, request.url.path, request.url.query, None, None)


@router.post("/hpc/jobs/{subpath:path}", include_in_schema=False)
async def jobs_post(subpath: str, request: Request) -> Response:
    start_backends()
    body = await request.body()
    scope = scope_from_request(request, body)
    return proxy_response("POST", scope, request.url.path, request.url.query, body, request.headers.get("content-type"))
