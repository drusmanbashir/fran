#!/bin/bash
# python train_retry.py --project kits2 --plan-num 3 --fold 1 --epochs 500 --bs 3 --bsf true --devices [1] --wandb true --train-indices 20 --val-every-n-epochs 1
# python train.py --project lits32 --devices [0] --bs 8 --fold 2
# python train.py --project litsmc --devices [0] --bs 10 --fold 1 --bsf true --epochs 500
# python train.py --project kits2 --plan-num 3 --fold 1 --epochs 500 --bs 18 --bsf false --devices [1] --wandb true --train-indices 40 --val-every-n-epochs 5
python train_retry.py --project kits2 --plan-num 3 --fold 1 --epochs 500 --bs 3 --bsf true --devices [1] --wandb true --train-indices 20 --val-every-n-epochs 1
