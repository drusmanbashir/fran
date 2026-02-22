#!/usr/bin/env python3
"""
Minimal natural-language launcher for fran/run/train.py.

Examples:
  python fran/run/train_agent.py "train bones, plan 1"
  python fran/run/train_agent.py "train bones plan 1 fold 0 on gpu 1" --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = (
    "Extract training CLI args from user text. "
    "Return strict JSON with keys: "
    "project_title (str|null), plan (int|null), fold (int|null), devices (str|null), "
    "epochs (int|null), batch_size (int|null), lr (float|null), incremental (bool|null). "
    "Do not include extra keys."
)


def parse_args():
    p = argparse.ArgumentParser(description="Natural-language launcher for train.py")
    p.add_argument("text", help='Natural language command, e.g. "train bones, plan 1"')
    p.add_argument("--dry-run", action="store_true", help="Print command, do not execute")
    return p.parse_args()


def _post_json(url: str, headers: dict, payload: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def llm_extract(user_text: str) -> dict | None:
    # Anthropic (Claude)
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        payload = {
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            "max_tokens": 300,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_text}],
        }
        out = _post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            payload,
        )
        text = out["content"][0]["text"]
        return json.loads(text)

    # OpenAI-compatible endpoint (OpenAI or OpenRouter)
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openai_key or openrouter_key:
        if openrouter_key:
            base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
            auth_key = openrouter_key
            extra_headers = {}
        else:
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            auth_key = openai_key
            extra_headers = {}

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        out = _post_json(
            f"{base.rstrip('/')}/chat/completions",
            {"Authorization": f"Bearer {auth_key}", **extra_headers},
            payload,
        )
        text = out["choices"][0]["message"]["content"]
        return json.loads(text)
    return None


def ask_missing(spec: dict, project_choices: list[str]):
    if not spec.get("project_title"):
        prompt = "Project title"
        if project_choices:
            prompt += f" {project_choices[:8]}"
        spec["project_title"] = input(prompt + ": ").strip()
    if spec.get("project_title") not in project_choices and project_choices:
        print(
            f"Warning: project '{spec.get('project_title')}' not found under /s/fran_storage/projects"
        )
    if spec.get("plan") is None:
        raw = input("Plan id (e.g. 1): ").strip()
        spec["plan"] = int(raw) if raw else 1


def list_projects() -> list[str]:
    base = Path("/s/fran_storage/projects")
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def build_cmd(spec: dict) -> list[str]:
    cmd = [sys.executable, "fran/run/train.py", "-t", spec["project_title"], "-p", str(spec["plan"])]
    if spec.get("fold") is not None:
        cmd += ["-f", str(spec["fold"])]
    if spec.get("devices") is not None:
        cmd += ["--devices", str(spec["devices"])]
    if spec.get("epochs") is not None:
        cmd += ["-e", str(spec["epochs"])]
    if spec.get("batch_size") is not None:
        cmd += ["--bs", str(spec["batch_size"])]
    if spec.get("lr") is not None:
        cmd += ["-lr", str(spec["lr"])]
    if spec.get("incremental") is True:
        cmd += ["--incremental", "true"]
    return cmd


def main():
    args = parse_args()
    projects = list_projects()

    spec = None
    try:
        spec = llm_extract(args.text)
    except Exception:
        spec = None
    if not spec:
        print(
            "No LLM extraction available. Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY."
        )
        raise SystemExit(2)

    ask_missing(spec, projects)
    cmd = build_cmd(spec)
    print("Running:", " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
