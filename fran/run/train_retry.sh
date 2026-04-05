#!/usr/bin/env bash
# python train_retry.py --project kits2 --plan-num 3 --fold 1 --epochs 500 --bs 3 --bsf true --devices [1] --wandb true --train-indices 20 --val-every-n-epochs 1
# python -m ipdb train_retry.py --project kits2 --plan-num 3 --fold 1 --epochs 500 --bs 3 --bsf true --devices [1] --wandb true --train-indices 20 --val-every-n-epochs 1
python train_retry.py --project kits2 --plan-num 3 --fold 1 --epochs 500 --bs 3 --bsf true --devices [1] --wandb true --train-indices 20 --val-every-n-epochs 1
