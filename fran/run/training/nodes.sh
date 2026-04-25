#!/usr/bin/env bash
# python train.py --project nodes --plan-num 7 --devices 1 --bs 1 --batch-finder true --epochs 600 --compiled false --profiler false --wandb true --cache-rate 0.0 --fold 0
# python -m ipdb train.py --project nodes --plan-num 7 --devices 1 --bs 1 --batch-finder true --epochs 600 --compiled false --profiler false --wandb true --cache-rate 0.0 --fold 0
# python -m ipdb train.py --project nodes --plan-num 7 --devices 1 --bs 1 --batch-finder true --epochs 600 --compiled false --profiler false --wandb true --cache-rate 0.0 --fold 0 --run-name LITS-1290
python -m ipdb train.py --project nodes --plan-num 7 --devices 1 --bs 1 --batch-finder true --epochs 600 --compiled false --profiler false --wandb true --cache-rate 0.0 --fold 0
