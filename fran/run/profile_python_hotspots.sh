#!/usr/bin/env bash
set -euo pipefail

PROJECT_TITLE="${PROJECT_TITLE:-lidc}"
TRACE_DIR="${TRACE_DIR:-}"
STAMP="${STAMP:-}"
TOP_K="${TOP_K:-20}"
UNIT_LABEL="${UNIT_LABEL:-self_cpu_time_total}"

cmd=(
  python -m fran.run.profile_python_hotspots
  --project-title "${PROJECT_TITLE}"
  --top-k "${TOP_K}"
  --unit-label "${UNIT_LABEL}"
)

if [[ -n "${TRACE_DIR}" ]]; then
  cmd+=(--trace-dir "${TRACE_DIR}")
fi
if [[ -n "${STAMP}" ]]; then
  cmd+=(--stamp "${STAMP}")
fi

"${cmd[@]}" "$@"
