#!/usr/bin/env bash
# nodes.sh — run training.py with nodes.py defaults

python  -m ipdb train.py \
  --project lidc\
  --plan-num 1 \
  # --devices 0 \
  --bs 4 \
  --epochs 20 \
  --compiled false\
  --profiler false \
  --neptune true \
  --cache-rate 0.0\
  --fold 0\
  # --run-name LITS-1290 \
