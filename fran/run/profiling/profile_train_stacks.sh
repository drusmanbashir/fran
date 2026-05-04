#!/usr/bin/env bash
MODULE="fran.run.profile_train_stacks"
PROJECT="kits23"
PLAN_NUM="2"
DEVICES='[0]'
BATCH_SIZE="1"
EPOCHS="1"
N_SAMPLES="2"
SKIP_VAL="true"
NUM_WORKERS="0"
CACHE_RATE="0"
LIMIT_TRAIN_BATCHES="5"
TRACE_EXPORT_CHROME="true"
TRACE_CPU_PROFILING="false"
TRACE_PROFILE_WITH_STACK="false"
TRACE_EXPORT_STACKS="false"

# Default is summary-only: no Chrome trace, no stacks, CUDA ops table plus optional CPU table.
# python -m "$MODULE" -t "$PROJECT" -p "$PLAN_NUM" --devices "$DEVICES" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --n-samples "$N_SAMPLES" --skip-val "$SKIP_VAL" --num-workers "$NUM_WORKERS" --cache-rate "$CACHE_RATE" --limit-train-batches "$LIMIT_TRAIN_BATCHES"
# Only request a Perfetto timeline deliberately, and keep CPU/stacks off to avoid huge JSON.
# python -m "$MODULE" -t "$PROJECT" -p "$PLAN_NUM" --devices "$DEVICES" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --n-samples "$N_SAMPLES" --skip-val "$SKIP_VAL" --num-workers "$NUM_WORKERS" --cache-rate "$CACHE_RATE" --limit-train-batches "$LIMIT_TRAIN_BATCHES" --export-chrome-trace "$TRACE_EXPORT_CHROME" --cpu-profiling "$TRACE_CPU_PROFILING" --profile-with-stack "$TRACE_PROFILE_WITH_STACK" --export-stacks "$TRACE_EXPORT_STACKS"
python -m "$MODULE" -t "$PROJECT" -p "$PLAN_NUM" --devices "$DEVICES" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --n-samples "$N_SAMPLES" --skip-val "$SKIP_VAL" --num-workers "$NUM_WORKERS" --cache-rate "$CACHE_RATE" --limit-train-batches "$LIMIT_TRAIN_BATCHES"
