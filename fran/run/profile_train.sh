#!/usr/bin/env bash
set -euo pipefail

PROJECT_TITLE="${PROJECT_TITLE:-lidc}"
PLAN="${PLAN:-3}"
DEVICES="${DEVICES:-[0]}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-2}"
COMPILED="${COMPILED:-false}"
TEST_EVERY_N_EPOCHS="${TEST_EVERY_N_EPOCHS:-0}"

N_SAMPLES="${N_SAMPLES:-16}"
PROFILE_MODE="${PROFILE_MODE:-verbose}"
CPU_PROFILING="true"
PROFILE_PLOTTING="true"
PROFILE_RECORD_SHAPES="false"
PROFILE_WITH_STACK="true"

python -m fran.run.profile_train \
  --project-title "${PROJECT_TITLE}" \
  --plan "${PLAN}" \
  --devices "${DEVICES}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --compiled "${COMPILED}" \
  --test-every-n-epochs "${TEST_EVERY_N_EPOCHS}" \
  --n-samples "${N_SAMPLES}" \
  --profile-mode "${PROFILE_MODE}" \
  --cpu-profiling "${CPU_PROFILING}" \
  --profile-plotting "${PROFILE_PLOTTING}" \
  --profile-record-shapes "${PROFILE_RECORD_SHAPES}" \
  --profile-with-stack "${PROFILE_WITH_STACK}" \
  "$@"
