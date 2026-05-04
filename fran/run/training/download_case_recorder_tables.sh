#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODULE="fran.wandb.download_case_recorder_tables"
PROJECT="kits23"
RUN_NAME="KITS23-SIRIG"

cd "$FRAN_ROOT"
python -m "$MODULE" --project "$PROJECT" --run-name "$RUN_NAME"
