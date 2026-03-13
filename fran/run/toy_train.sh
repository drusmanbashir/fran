#!/bin/bash
python train_retry.py \
  --project kits \
  --plan-num 3 \
  --fold 1 \
  --epochs 500 \
  --bs 3 \
  --bsf true \
  --devices "[1]" \
  --wandb true \
  --train-indices 20 \
  --val-every-n-epochs 1

# python train.py -t lits32 -d [0] --bs 8 -f 2
# python  train.py -t litsmc -d [0]  --bs 10 -f 1 -bsf  -e 500
# python  train.py -t litsmc -r LITS-811 -e 500 --lr 11e-4 -b 8
# python  train.py -t litsmc -r LITS-940 -e 500  -d [1]  
# python train.py \
#   --project kits \
#   --plan-num 3 \
#   --fold 1 \
#   --epochs 500 \
#   --bsf false \
#   --bs 18 \
#   --devices [1] \
#   --wandb true \
#   --train-indices 40 \
#   --val-every-n-epochs 5
# python  train.py -t lungs  -e 500 --lr 11e-3 …-b 2

