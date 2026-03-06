#!/usr/bin/env bash
set -euo pipefail

PROJECT_TITLE="${PROJECT_TITLE:-lidc}"
PLAN="${PLAN:-3}"
DEVICES="${DEVICES:-[0]}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-1}"
COMPILED="${COMPILED:-false}"
TEST_EVERY_N_EPOCHS="${TEST_EVERY_N_EPOCHS:-0}"

N_SAMPLES="${N_SAMPLES:-4}"
CPU_PROFILING="${CPU_PROFILING:-true}"
PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-false}"
PROFILE_WITH_STACK="${PROFILE_WITH_STACK:-true}"
PROFILE_EXPERIMENTAL_VERBOSE="${PROFILE_EXPERIMENTAL_VERBOSE:-true}"
STACK_DEPTH="${STACK_DEPTH:-4}"
EXPORT_STACKS="${EXPORT_STACKS:-true}"

python -m fran.run.profile_train_stacks \
  --project-title "${PROJECT_TITLE}" \
  --plan "${PLAN}" \
  --devices "${DEVICES}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --compiled "${COMPILED}" \
  --test-every-n-epochs "${TEST_EVERY_N_EPOCHS}" \
  --n-samples "${N_SAMPLES}" \
  --cpu-profiling "${CPU_PROFILING}" \
  --profile-record-shapes "${PROFILE_RECORD_SHAPES}" \
  --profile-with-stack "${PROFILE_WITH_STACK}" \
  --profile-experimental-verbose "${PROFILE_EXPERIMENTAL_VERBOSE}" \
  --stack-depth "${STACK_DEPTH}" \
  --export-stacks "${EXPORT_STACKS}" \
  "$@"
