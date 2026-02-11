#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $(basename "$0") <command...>" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH. Ensure conda is installed and on PATH." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate a1

exec "$@"
