#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FRAN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HOME_DIR="$(cd "$FRAN_ROOT/../.." && pwd)"
PYTHON_BIN="$HOME_DIR/mambaforge/envs/dl/bin/python"
BLOCK_SUSPEND="$RUN_DIR/misc/block_suspend.py"
TRAIN_PY="$SCRIPT_DIR/train.py"
SUSPEND_ONLY="--suspend-only"
PROJECT="${1:-totalseg}"
PLAN_NUM="${2:-8}"
DEVICES="${3:-[0]}"
BS="${4:-2}"
EPOCHS="${5:-8}"
FOLD="${6:-1}"
VAL_DEVICE="${7:-cuda}"
COMPILED="${8:-false}"
PROFILER="${9:-false}"
WANDB="${10:-true}"
CACHE_RATE="${11:-0.0}"
LR="${12:-0.001}"
RUN_NAME="${13:-}"
DESCRIPTION="${14:-}"
DS_TYPE="${15:-}"
ALL="${16:-false}"
VAL_EVERY_N_EPOCHS="${17:-2}"
TRAIN_INDICES="${18:-50}"
BSF="${19:-true}"
DUAL_SSD="${20:-false}"
BATCH_TFMS="${21:-false}"

cmd=(
  "$PYTHON_BIN" "$BLOCK_SUSPEND" "$SUSPEND_ONLY" "$TRAIN_PY"
  --project "$PROJECT"
  --plan-num "$PLAN_NUM"
  --devices "$DEVICES"
  --bs "$BS"
  --epochs "$EPOCHS"
  --fold "$FOLD"
  --compiled "$COMPILED"
  --profiler "$PROFILER"
  --wandb "$WANDB"
  --cache-rate "$CACHE_RATE"
  --val-device "$VAL_DEVICE"
  --all "$ALL"
  --val-every-n-epochs "$VAL_EVERY_N_EPOCHS"
  --bsf "$BSF"
  --dual-ssd "$DUAL_SSD"
  --batch-tfms "$BATCH_TFMS"
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
