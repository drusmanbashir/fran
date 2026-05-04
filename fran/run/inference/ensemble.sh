#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENSEMBLE_PY="$SCRIPT_DIR/ensemble_singlegpu.py"
PROJECT="lits"
EXPERIMENTS=(LITS-499 LITS-500 LITS-501 LITS-502 LITS-503)
INPUT_DIR="/s/insync/datasets/crc_project/images_ub/done"

# python "$ENSEMBLE_PY" -t "$PROJECT" -e "${EXPERIMENTS[@]}" -i "$INPUT_DIR"
# python -m ipdb "$ENSEMBLE_PY" -t "$PROJECT" -e "${EXPERIMENTS[@]}" -i "$INPUT_DIR"
python -m ipdb "$ENSEMBLE_PY" -t "$PROJECT" -e "${EXPERIMENTS[@]}" -i "$INPUT_DIR"
