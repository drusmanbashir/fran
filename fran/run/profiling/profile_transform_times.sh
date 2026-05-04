#!/usr/bin/env bash
MODULE="fran.run.profile_transform_times"
PROJECT="kits23"
PLAN_NUM="2"
DEVICES='[0]'
BS="6"
TRAIN_INDICES="192"
LIMIT_BATCHES="24"
NUM_WORKERS="24"
PREFETCH_FACTOR="4"
BATCH_AFFINE="true"
CPU_BATCH_AFFINE="false"

# CPU affine path.
# python -m "$MODULE" -t "$PROJECT" -p "$PLAN_NUM" --devices "$DEVICES" --bs "$BS" --train-indices "$TRAIN_INDICES" --limit-batches "$LIMIT_BATCHES" --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" --batch-affine "$CPU_BATCH_AFFINE"
# After-batch affine path; CPU Affine is skipped in the dataloader.
python -m "$MODULE" -t "$PROJECT" -p "$PLAN_NUM" --devices "$DEVICES" --bs "$BS" --train-indices "$TRAIN_INDICES" --limit-batches "$LIMIT_BATCHES" --num-workers "$NUM_WORKERS" --prefetch-factor "$PREFETCH_FACTOR" --batch-affine "$BATCH_AFFINE"
