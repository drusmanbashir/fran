#!/usr/bin/env bash
# CPU affine path.
# python -m fran.run.profile_transform_times -t kits23 -p 2 --devices '[0]' --bs 6 --train-indices 192 --limit-batches 24 --num-workers 24 --prefetch-factor 4 --batch-affine false
# After-batch affine path; CPU Affine is skipped in the dataloader.
python -m fran.run.profile_transform_times -t kits23 -p 2 --devices '[0]' --bs 6 --train-indices 192 --limit-batches 24 --num-workers 24 --prefetch-factor 4 --batch-affine true
