#!/usr/bin/env bash
# python train.py --project lidc --plan-num 1 --devices 0 --bs 4 --epochs 20 --compiled false --profiler false --wandb true --cache-rate 0.0 --fold 0
# python -m ipdb train.py --project lidc --plan-num 1 --devices 0 --bs 4 --epochs 20 --compiled false --profiler false --wandb true --cache-rate 0.0 --fold 0
# python -m ipdb train.py --project lidc --plan-num 1 --devices 0 --bs 4 --epochs 20 --compiled false --profiler false --wandb true --cache-rate 0.0 --fold 0 --run-name LITS-1290
python -m ipdb train.py --project lidc --plan-num 1 --devices 0 --bs 4 --epochs 20 --compiled false --profiler false --wandb true --cache-rate 0.0 --fold 0
