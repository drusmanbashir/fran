#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FRAN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HOME_DIR="$(cd "$FRAN_ROOT/../.." && pwd)"
PYTHON_BIN="$HOME_DIR/mambaforge/envs/dl/bin/python"
BLOCK_SUSPEND="$RUN_DIR/misc/block_suspend.py"
TRAIN_PY="$SCRIPT_DIR/train.py"
SUSPEND_ONLY="--suspend-only"
PROJECT="kits23"
PLAN_NUM="2"
FOLD="1"
EPOCHS="8"
DEVICES='[0]'
WANDB="true"
VAL_EVERY_N_EPOCHS="2"
RUN_NAME=""
# EXAMPLE_RUN_NAME="KITS23-SIRIG"

# exec "$PYTHON_BIN" "$BLOCK_SUSPEND" --suspend-only "$TRAIN_PY" --project kits23 --plan-num 2 --fold 1 --epochs 800 --devices [0] --wandb true --val-every-n-epochs 2 --run-name "$EXAMPLE_RUN_NAME"

cmd=(
  "$PYTHON_BIN" "$BLOCK_SUSPEND" "$SUSPEND_ONLY" "$TRAIN_PY"
  --project "$PROJECT"
  --plan-num "$PLAN_NUM"
  --fold "$FOLD"
  --epochs "$EPOCHS"
  --devices "$DEVICES"
  --wandb "$WANDB"
  --val-every-n-epochs "$VAL_EVERY_N_EPOCHS"
)

if [[ -n "${RUN_NAME}" && "${RUN_NAME}" != "none" && "${RUN_NAME}" != "null" ]]; then
  cmd+=(--run-name "$RUN_NAME")
fi

exec "${cmd[@]}"
