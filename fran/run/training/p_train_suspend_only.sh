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
EPOCHS="800"
DEVICES='[0]'
WANDB="true"
VAL_EVERY_N_EPOCHS="2"
RUN_NAME="KITS23-SIRIG"

exec "$PYTHON_BIN" "$BLOCK_SUSPEND" "$SUSPEND_ONLY" "$TRAIN_PY" --project "$PROJECT" --plan-num "$PLAN_NUM" --fold "$FOLD" --epochs "$EPOCHS" --devices "$DEVICES" --wandb "$WANDB" --val-every-n-epochs "$VAL_EVERY_N_EPOCHS" --run-name "$RUN_NAME"
