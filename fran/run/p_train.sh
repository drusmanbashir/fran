#!/bin/bash
# /home/ub/mambaforge/envs/dl/bin/python /home/ub/code/fran/fran/run/block_suspend.py /home/ub/code/fran/fran/run/train.py --project lits32 --devices [0] --bs 8 --fold 2
# /home/ub/mambaforge/envs/dl/bin/python /home/ub/code/fran/fran/run/block_suspend.py /home/ub/code/fran/fran/run/train.py --project litsmc --devices [0] --bs 10 --fold 1 --bsf true --epochs 500
# /home/ub/mambaforge/envs/dl/bin/python /home/ub/code/fran/fran/run/block_suspend.py --allow-suspend /home/ub/code/fran/fran/run/train.py --project kits2 --plan-num 1 --fold 1 --epochs 600 --bs 4 --bsf true --devices [1] --wandb true --val-every-n-epochs 5 --run-name KITS2-bah
exec /home/ub/mambaforge/envs/dl/bin/python /home/ub/code/fran/fran/run/block_suspend.py /home/ub/code/fran/fran/run/train.py --project kits23 --plan-num 2 --fold 1 --epochs 800 --devices [0] --wandb true --val-every-n-epochs 2 --run-name KITS23-SIRIG
