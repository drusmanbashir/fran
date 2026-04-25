#!/bin/bash
exec /home/ub/mambaforge/envs/dl/bin/python /home/ub/code/fran/fran/run/block_suspend.py --suspend-only /home/ub/code/fran/fran/run/train.py --project kits23 --plan-num 2 --fold 1 --epochs 800 --devices [0] --wandb true --val-every-n-epochs 2 --run-name KITS23-SIRIG
