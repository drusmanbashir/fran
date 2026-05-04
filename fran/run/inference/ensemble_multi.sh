#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENSEMBLE_PY="$SCRIPT_DIR/ensemble.py"
PROJECT="litsmc"
EXPERIMENTS=(LITS-787 LITS-810 LITS-811)
INPUT_DIR="/s/xnat_shadow/crc/wxh/images/"
EXAMPLE_INPUT_DIR="/s/xnat_shadow/crc/srn/cases_with_findings/images"
EXAMPLE_ALT_PROJECT="lits"
EXAMPLE_ALT_EXPERIMENTS=(LITS-499 LITS-500 LITS-501 LITS-502 LITS-503)
EXAMPLE_ALT_INPUT_DIR="/s/xnat_shadow/litq/test/images_few/"
EXAMPLE_OVERWRITE="-o"

# python "$ENSEMBLE_PY" -t "$PROJECT" -e "${EXPERIMENTS[@]}" -i "$INPUT_DIR"
# python "$ENSEMBLE_PY" -t "$PROJECT" -e "${EXPERIMENTS[@]}" -i "$EXAMPLE_INPUT_DIR"
# python "$ENSEMBLE_PY" -t "$EXAMPLE_ALT_PROJECT" -e "${EXAMPLE_ALT_EXPERIMENTS[@]}" -i "$EXAMPLE_ALT_INPUT_DIR" "$EXAMPLE_OVERWRITE"
python "$ENSEMBLE_PY" -t "$PROJECT" -e "${EXPERIMENTS[@]}" -i "$INPUT_DIR"
