#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [ "$#" -eq 0 ]; then
  cat <<'EOF'
Usage: ./project.sh -t <project_title> -m <mnemonic> -ds <datasource> [<datasource> ...]

Examples:
  ./project.sh -t tmpts -m test -ds totalseg_short
  ./project.sh -t nodes -m nodes -ds nodes nodesthick
EOF
  exit 2
fi

python project_init.py "$@"
