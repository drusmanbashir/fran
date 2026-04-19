#!/bin/bash
# python p_train_perf.py --project kits23 --plan-num 2 --fold 1 --devices '[0]' --bs 6 --train-indices 96 --limit-train-batches 12 --repeat 1 --num-workers 8,12,16,24 --prefetch-factor 2
# python p_train_perf.py --project kits23 --plan-num 2 --fold 1 --devices '[0]' --bs 6 --train-indices 192 --limit-train-batches 24 --repeat 2 --num-workers 12,24 --prefetch-factor 2,4
python /home/ub/code/fran/fran/run/p_train_perf.py --project kits23 --plan-num 2 --fold 1 --devices '[0]' --bs 6 --train-indices 192 --val-indices 1 --limit-train-batches 24 --repeat 1 --num-workers 12,24 --prefetch-factor 2
