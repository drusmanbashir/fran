#!/usr/bin/env bash
# Default is summary-only: no Chrome trace, no stacks, CUDA ops table plus optional CPU table.
# python -m fran.run.profile_train_stacks -t kits23 -p 2 --devices '[0]' --batch-size 1 --epochs 1 --n-samples 2 --skip-val true --num-workers 0 --cache-rate 0 --limit-train-batches 5
# Only request a Perfetto timeline deliberately, and keep CPU/stacks off to avoid huge JSON.
# python -m fran.run.profile_train_stacks -t kits23 -p 2 --devices '[0]' --batch-size 1 --epochs 1 --n-samples 2 --skip-val true --num-workers 0 --cache-rate 0 --limit-train-batches 5 --export-chrome-trace true --cpu-profiling false --profile-with-stack false --export-stacks false
python -m fran.run.profile_train_stacks -t kits23 -p 2 --devices '[0]' --batch-size 1 --epochs 1 --n-samples 2 --skip-val true --num-workers 0 --cache-rate 0 --limit-train-batches 5
