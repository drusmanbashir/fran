#!/bin/bash
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
# python "$SCRIPT_DIR/train.py" --project lits32 --devices [0] --bs 8 --fold 2
# python "$SCRIPT_DIR/train.py" --project litsmc --devices [0] --bs 10 --fold 1 --bsf true --epochs 500
# python "$SCRIPT_DIR/train.py" --project kits2 --plan-num 3 --fold 1 --epochs 500 --bs 18 --bsf false --devices [1] --wandb true --train-indices 40 --val-every-n-epochs 5
python "$TRAIN_RETRY_PY" --project "$PROJECT" --plan-num "$PLAN_NUM" --fold "$FOLD" --epochs "$EPOCHS" --bs "$BS" --bsf "$BSF" --devices "$DEVICES" --wandb "$WANDB" --train-indices "$TRAIN_INDICES" --val-every-n-epochs "$VAL_EVERY_N_EPOCHS"
