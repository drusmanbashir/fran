#!/usr/bin/env bash
set -euo pipefail

cd /home/ub/code
python -m fran.wandb.download_case_recorder_tables --project kits23 --run-name KITS23-SIRIG
