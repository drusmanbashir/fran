#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_RETRY_PY="$SCRIPT_DIR/train_retry.py"
PROJECT="kits2"
PLAN_NUM="3"
FOLD="1"
EPOCHS="500"
BS="3"
BSF="true"
DEVICES='[1]'
WANDB="true"
TRAIN_INDICES="20"
VAL_EVERY_N_EPOCHS="1"

# python "$TRAIN_RETRY_PY" --project "$PROJECT" --plan-num "$PLAN_NUM" --fold "$FOLD" --epochs "$EPOCHS" --bs "$BS" --bsf "$BSF" --devices "$DEVICES" --wandb "$WANDB" --train-indices "$TRAIN_INDICES" --val-every-n-epochs "$VAL_EVERY_N_EPOCHS"
# python -m ipdb "$TRAIN_RETRY_PY" --project "$PROJECT" --plan-num "$PLAN_NUM" --fold "$FOLD" --epochs "$EPOCHS" --bs "$BS" --bsf "$BSF" --devices "$DEVICES" --wandb "$WANDB" --train-indices "$TRAIN_INDICES" --val-every-n-epochs "$VAL_EVERY_N_EPOCHS"
python "$TRAIN_RETRY_PY" --project "$PROJECT" --plan-num "$PLAN_NUM" --fold "$FOLD" --epochs "$EPOCHS" --bs "$BS" --bsf "$BSF" --devices "$DEVICES" --wandb "$WANDB" --train-indices "$TRAIN_INDICES" --val-every-n-epochs "$VAL_EVERY_N_EPOCHS"
