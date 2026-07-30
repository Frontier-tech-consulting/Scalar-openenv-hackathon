#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found at $PYTHON_BIN"
  exit 1
fi

echo "[1/2] Validating worker-level pipeline"
"$PYTHON_BIN" "$REPO_DIR/_validate_pipeline.py"

echo "[2/2] Building demo object"
"$PYTHON_BIN" -c 'from egocentric_dataset_test.competition.demo import create_demo; demo=create_demo(); print("demo OK" if demo is not None else "demo unavailable")'

echo "Demo e2e checks passed."
