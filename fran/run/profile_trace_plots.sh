#!/usr/bin/env bash
set -euo pipefail

PROJECT_TITLE="${PROJECT_TITLE:-lidc}"
TRACE_DIR="${TRACE_DIR:-}"
STAMP="${STAMP:-}"
TOP_K="${TOP_K:-20}"

cmd=(python -m fran.run.profile_trace_plots --project-title "${PROJECT_TITLE}" --top-k "${TOP_K}")

if [[ -n "${TRACE_DIR}" ]]; then
  cmd+=(--trace-dir "${TRACE_DIR}")
fi
if [[ -n "${STAMP}" ]]; then
  cmd+=(--stamp "${STAMP}")
fi

"${cmd[@]}" "$@"
