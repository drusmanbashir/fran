#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="$SCRIPT_DIR/train.py"
PROJECT="${1:-nodes}"
PLAN_NUM="${2:-7}"
DEVICES="${3:-1}"
BS="${4:-1}"
EPOCHS="${5:-600}"
FOLD="${6:-0}"
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
VAL_EVERY_N_EPOCHS="${17:-5}"
TRAIN_INDICES="${18:-}"
BSF="${19:-true}"
DUAL_SSD="${20:-false}"
BATCH_TFMS="${21:-false}"

cmd=(
  python -m ipdb "$TRAIN_PY"
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
