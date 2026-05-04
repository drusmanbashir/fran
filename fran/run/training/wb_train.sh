#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="$SCRIPT_DIR/train.py"
PROJECT="lits32"
DEVICES="2"
BS="8"
FOLD="2"

# python "$TRAIN_PY" -t "$PROJECT" -d "$DEVICES" --bs "$BS" -f "$FOLD"
# python -m ipdb "$TRAIN_PY" -t "$PROJECT" -d "$DEVICES" --bs "$BS" -f "$FOLD"
python "$TRAIN_PY" -t "$PROJECT" -d "$DEVICES" --bs "$BS" -f "$FOLD"
