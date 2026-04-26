#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <project_title> [project_title ...]" >&2
  exit 2
fi

cd "${SCRIPT_DIR}"
python project_delete.py "$@"
