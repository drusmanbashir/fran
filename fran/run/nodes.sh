#!/usr/bin/env bash
# nodes.sh — run training.py with nodes.py defaults

python -m ipdb train.py \
  --project nodes \
  --plan-num 7 \
  --devices 0 \
  --bs 4 \
  --epochs 600 \
  --compiled false\
  --profiler false \
  --neptune true \
  --cache-rate 0.0\
  --fold 0\
  # --run-name LITS-1290 \
