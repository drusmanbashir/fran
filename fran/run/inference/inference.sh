#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_PY="$SCRIPT_DIR/by_mnemonic.py"

MNEMONIC="${MNEMONIC:-kidneys}"
LOCALISER_TYPE="${LOCALISER_TYPE:-}"
FOLDER="${FOLDER:-}"
DATASET="${DATASET:-totalseg}"
GPUS=(0)
CHUNKSIZE="${CHUNKSIZE:-4}"
PATCH_OVERLAP="${PATCH_OVERLAP:-0.2}"
OVERWRITE="${OVERWRITE:-false}"
SOURCE_FLAG="${FOLDER:+--folder}"
SOURCE_FLAG="${SOURCE_FLAG:---dataset}"
SOURCE_VALUE="${FOLDER:-$DATASET}"

cmd=(python "$SCRIPT_DIR/by_mnemonic.py" "$MNEMONIC" "$SOURCE_FLAG" "$SOURCE_VALUE" --gpus "${GPUS[@]}" --chunksize "$CHUNKSIZE" --patch-overlap "$PATCH_OVERLAP")

if [[ -n "$LOCALISER_TYPE" ]]; then
  cmd+=(--localiser-type "$LOCALISER_TYPE")
fi

if [[ "$OVERWRITE" == true ]]; then
  cmd+=(-o)
fi

"${cmd[@]}"
