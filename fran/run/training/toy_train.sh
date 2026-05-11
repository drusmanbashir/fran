#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_RETRY_PY="$SCRIPT_DIR/train_retry.py"
PROJECT="${1:-kits2}"
PLAN_NUM="${2:-3}"
DEVICES="${3:-[1]}"
BS="${4:-3}"
EPOCHS="${5:-500}"
FOLD="${6:-1}"
VAL_DEVICE="${7:-cuda}"
COMPILED="${8:-false}"
PROFILER="${9:-false}"
WANDB="${10:-true}"
CACHE_RATE="${11:-0.0}"
LR="${12:-}"
RUN_NAME="${13:-}"
DESCRIPTION="${14:-}"
DS_TYPE="${15:-}"
ALL="${16:-false}"
VAL_EVERY_N_EPOCHS="${17:-1}"
TRAIN_INDICES="${18:-20}"
BSF="${19:-true}"
DUAL_SSD="${20:-false}"
MAX_RETRIES="${21:-3}"
STEP="${22:-1}"
MIN_BS="${23:-1}"
PYTHON_BIN="${24:-python}"

cmd=(
  "$PYTHON_BIN" "$TRAIN_RETRY_PY"
  --project "$PROJECT"
  --plan-num "$PLAN_NUM"
  --devices "$DEVICES"
  --bs "$BS"
  --fold "$FOLD"
  --epochs "$EPOCHS"
  --compiled "$COMPILED"
  --profiler "$PROFILER"
  --wandb "$WANDB"
  --cache-rate "$CACHE_RATE"
  --val-device "$VAL_DEVICE"
  --all "$ALL"
  --val-every-n-epochs "$VAL_EVERY_N_EPOCHS"
  --bsf "$BSF"
  --dual-ssd "$DUAL_SSD"
  --max-retries "$MAX_RETRIES"
  --step "$STEP"
  --min-bs "$MIN_BS"
)

if [[ -n "$LR" ]]; then
  cmd+=(--learning-rate "$LR")
fi
if [[ -n "$RUN_NAME" && "$RUN_NAME" != "none" && "$RUN_NAME" != "null" ]]; then
  cmd+=(--run-name "$RUN_NAME")
fi
if [[ -n "$DESCRIPTION" ]]; then
  cmd+=(--description "$DESCRIPTION")
fi
if [[ -n "$DS_TYPE" ]]; then
  cmd+=(--ds-type "$DS_TYPE")
fi
if [[ -n "$TRAIN_INDICES" && "$TRAIN_INDICES" != "none" && "$TRAIN_INDICES" != "null" ]]; then
  cmd+=(--train-indices "$TRAIN_INDICES")
fi

exec "${cmd[@]}"
