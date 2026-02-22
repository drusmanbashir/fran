#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REMOTE="mpx588@login.hpc.qmul.ac.uk"
DEFAULT_COLD_STORAGE="/data/EECS-LITQ/fran_storage"
DEFAULT_DATASETS_SUBDIR="datasets"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--source PATH] [--dataset REL_PATH] [--remote USER@HOST] [--cold-storage PATH] [--dry-run] [--yes]

Examples:
  $(basename "$0")
  $(basename "$0") --source /s/xnat_shadow/nodes/images --dataset xnat_shadow/nodes

Notes:
  - Remote target defaults to: \$COLD_STORAGE/datasets/<dataset>
  - If source is a directory and you want its contents only, pass a trailing slash.
USAGE
}

SOURCE=""
DATASET_REL=""
REMOTE="$DEFAULT_REMOTE"
COLD_STORAGE="$DEFAULT_COLD_STORAGE"
DRY_RUN=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="${2:-}"
      shift 2
      ;;
    --dataset)
      DATASET_REL="${2:-}"
      shift 2
      ;;
    --remote)
      REMOTE="${2:-}"
      shift 2
      ;;
    --cold-storage)
      COLD_STORAGE="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --yes|-y)
      ASSUME_YES=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

prompt_if_empty() {
  local var_name="$1"
  local prompt="$2"
  local default_value="${3:-}"
  local current_value
  current_value="${!var_name}"

  if [[ -z "$current_value" ]]; then
    if [[ -n "$default_value" ]]; then
      read -r -p "$prompt [$default_value]: " current_value
      current_value="${current_value:-$default_value}"
    else
      read -r -p "$prompt: " current_value
    fi
    printf -v "$var_name" '%s' "$current_value"
  fi
}

prompt_if_empty SOURCE "Local source path"
prompt_if_empty DATASET_REL "Dataset path under datasets/ (e.g. xnat_shadow/nodes)"
prompt_if_empty REMOTE "Remote user@host" "$DEFAULT_REMOTE"
prompt_if_empty COLD_STORAGE "Remote COLD_STORAGE root" "$DEFAULT_COLD_STORAGE"

if [[ ! -e "$SOURCE" ]]; then
  echo "Source path does not exist: $SOURCE" >&2
  exit 1
fi

REMOTE_TARGET="${COLD_STORAGE%/}/${DEFAULT_DATASETS_SUBDIR}/${DATASET_REL#/}/"

CMD=(rsync -avz --partial)
if [[ $DRY_RUN -eq 1 ]]; then
  CMD+=(--dry-run)
fi
CMD+=("$SOURCE" "${REMOTE}:${REMOTE_TARGET}")

echo
echo "Upload summary"
echo "  Source:        $SOURCE"
echo "  Remote:        $REMOTE"
echo "  COLD_STORAGE:  $COLD_STORAGE"
echo "  Dataset path:  ${DATASET_REL#/}"
echo "  Full target:   ${REMOTE}:${REMOTE_TARGET}"

echo
if [[ $ASSUME_YES -eq 0 ]]; then
  read -r -p "Proceed with rsync? [y/N]: " ok
  if [[ ! "$ok" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
  fi
fi

printf 'Running: '
printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
