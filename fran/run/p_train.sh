#!/bin/bash
# python train.py --project lits32 --devices [0] --bs 8 --fold 2
# python train.py --project litsmc --devices [0] --bs 10 --fold 1 --bsf true --epochs 500
# python train.py --project litsmc --run-name LITS-811 --epochs 500 --lr 11e-4 --bs 8
# python train.py --project kits2 --plan-num 1 --fold 1 --epochs 600 --bs 4 --bsf true --devices [1] --wandb true --val-every-n-epochs 5 --run-name KITS2-bah
python train.py --project kits2 --plan-num 1 --fold 1 --epochs 600 --bs 4 --bsf true --devices [1] --wandb true --val-every-n-epochs 5 --run-name KITS2-bah
