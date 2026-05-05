#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FRAN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HOME_DIR="$(cd "$FRAN_ROOT/../.." && pwd)"
PYTHON_BIN="$HOME_DIR/mambaforge/envs/dl/bin/python"
BLOCK_SUSPEND="$RUN_DIR/misc/block_suspend.py"
ANALYZE_RESAMPLE_PY="$SCRIPT_DIR/analyze_resample.py"
PROJECT="kits23"
PLAN_NUM="2"
NUM_PROCESSES="8"
OVERWRITE="${OVERWRITE:-False}"

if [[ $# -gt 0 ]]; then
  OVERWRITE="$1"
fi

OVERWRITE_ARGS=()
case "${OVERWRITE,,}" in
  true|1|yes|on)
    OVERWRITE_ARGS+=("-o")
    ;;
esac

# "$PYTHON_BIN" "$BLOCK_SUSPEND" "$ANALYZE_RESAMPLE_PY" -t litsmc -p 12 -n "$NUM_PROCESSES"
# "$PYTHON_BIN" "$BLOCK_SUSPEND" "$ANALYZE_RESAMPLE_PY" -t tmpa -p 2 -n "$NUM_PROCESSES"
# "$PYTHON_BIN" "$BLOCK_SUSPEND" "$ANALYZE_RESAMPLE_PY" -t lidc -p 1 -n "$NUM_PROCESSES"
# "$PYTHON_BIN" "$BLOCK_SUSPEND" --allow-suspend "$ANALYZE_RESAMPLE_PY" -t totalseg -p 2 -n 6
# exec "$PYTHON_BIN" "$BLOCK_SUSPEND" "$ANALYZE_RESAMPLE_PY" -t test -p 1 -n 6 "${OVERWRITE_ARGS[@]}"
# python "$BLOCK_SUSPEND" "$ANALYZE_RESAMPLE_PY" -t tmpts -p 3 -n "$NUM_PROCESSES"
"$PYTHON_BIN" "$BLOCK_SUSPEND" "$ANALYZE_RESAMPLE_PY" -t "$PROJECT" -p "$PLAN_NUM" -n "$NUM_PROCESSES" "${OVERWRITE_ARGS[@]}"
