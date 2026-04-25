#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${FRAN_CONF:-}" ]]; then
  echo "FRAN_CONF is not set" >&2
  exit 2
fi

CONFIG_FILE="${FRAN_CONF}/config.yaml"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config not found: ${CONFIG_FILE}" >&2
  exit 2
fi

python - "${CONFIG_FILE}" <<'PY'
import sys
from pathlib import Path
import yaml

config_file = Path(sys.argv[1])
config = yaml.safe_load(config_file.read_text())
projects_root = Path(config["projects_folder"]).expanduser()

if not projects_root.exists():
    raise FileNotFoundError(f"projects_folder does not exist: {projects_root}")
if not projects_root.is_dir():
    raise NotADirectoryError(f"projects_folder is not a directory: {projects_root}")

projects = []
for entry in sorted(projects_root.iterdir()):
    if entry.is_dir():
        projects.append(entry.name)

for name in projects:
    print(name)
PY
