#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAN_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HOME_DIR="$(cd "$FRAN_ROOT/../.." && pwd)"
PYTHON_BIN="$HOME_DIR/mambaforge/envs/dl/bin/python"
VIEW_IMAGE_PY="$SCRIPT_DIR/view_image.py"

# "$SCRIPT_DIR/imageviewer.sh" /tmp/image.pt
# "$SCRIPT_DIR/imageviewer.sh" /tmp/image.nii.gz /tmp/label.nii.gz
exec "$PYTHON_BIN" "$VIEW_IMAGE_PY" "$@"
