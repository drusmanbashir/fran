#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_PY="$SCRIPT_DIR/by_mnemonic.py"

LOCALISER_TYPE="${LOCALISER_TYPE:-}"
FOLDER="${FOLDER:-}"
DATASET="${DATASET:-totalseg}"
GPUS=(0)
CHUNKSIZE="${CHUNKSIZE:-4}"
PATCH_OVERLAP="${PATCH_OVERLAP:-0.2}"
OVERWRITE="${OVERWRITE:-false}"
MNEMONIC="${MNEMONIC:-kidneys}"
POSITIONAL_SOURCES=()
PASSTHROUGH_ARGS=()
POSITIONAL_LOCALISER_TYPE=""
POSITIONAL_SOURCE_MODE=""

if [[ "$#" -gt 0 ]]; then
  MNEMONIC="$1"
  shift
fi

if [[ "$#" -gt 0 && ( "$1" == "yolo" || "$1" == "TSL" ) ]]; then
  POSITIONAL_LOCALISER_TYPE="$1"
  shift
fi

if [[ "$#" -gt 0 && ( "$1" == "dataset" || "$1" == "datasource" || "$1" == "folder" ) ]]; then
  POSITIONAL_SOURCE_MODE="$1"
  shift
fi

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --localiser-type)
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --chunksize|--patch-overlap|--folder|--dataset)
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --gpus)
      PASSTHROUGH_ARGS+=("$1")
      shift
      while [[ "$#" -gt 0 && "$1" != --* ]]; do
        PASSTHROUGH_ARGS+=("$1")
        shift
      done
      ;;
    -o|--overwrite)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    *)
      POSITIONAL_SOURCES+=("$1")
      shift
      ;;
  esac
done

cmd=(python "$INFER_PY" "$MNEMONIC")

if [[ "$POSITIONAL_SOURCE_MODE" == "folder" ]]; then
  cmd+=(--folder "${POSITIONAL_SOURCES[0]}")
elif [[ "${#POSITIONAL_SOURCES[@]}" -gt 0 ]]; then
  cmd+=(--dataset "${POSITIONAL_SOURCES[@]}")
elif [[ -n "$FOLDER" ]]; then
  cmd+=(--folder "$FOLDER")
else
  cmd+=(--dataset "$DATASET")
fi

cmd+=(--gpus "${GPUS[@]}" --chunksize "$CHUNKSIZE" --patch-overlap "$PATCH_OVERLAP")
cmd+=("${PASSTHROUGH_ARGS[@]}")

if [[ -n "$LOCALISER_TYPE" ]]; then
  cmd+=(--localiser-type "$LOCALISER_TYPE")
fi

if [[ -n "$POSITIONAL_LOCALISER_TYPE" ]]; then
  cmd+=(--localiser-type "$POSITIONAL_LOCALISER_TYPE")
fi

if [[ "$OVERWRITE" == true ]]; then
  cmd+=(-o)
fi

"${cmd[@]}"
