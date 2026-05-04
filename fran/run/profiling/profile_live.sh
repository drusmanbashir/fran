#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_LIVE_PY="$SCRIPT_DIR/profile_live.py"
PROJECT="kits23"
GPU="0"
SECONDS="120"
INTERVAL_MS="500"

# python "$PROFILE_LIVE_PY" --project "$PROJECT" --gpu "$GPU" --seconds "$SECONDS" --interval-ms "$INTERVAL_MS"
python "$PROFILE_LIVE_PY" --project "$PROJECT" --gpu "$GPU" --seconds "$SECONDS" --interval-ms "$INTERVAL_MS"
