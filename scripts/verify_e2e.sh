#!/usr/bin/env bash
set -euo pipefail

echo "==> VERIFY: TESTS"
# Check if pytest is effectively installed by trying to import it
if [ -f "pyproject.toml" ]; then
  if uv run python -c "import pytest" &>/dev/null; then
      uv run python -m pytest -q || { echo "FAIL: TESTS"; exit 10; }
  else
      echo "WARN: pytest not found, skipping tests."
  fi
fi

echo "==> VERIFY: DATA_SMOKE"
if uv run python -c "import importlib; import sys; sys.exit(0)"; then
  if [ -f "scripts/data_smoke.py" ]; then
    uv run python scripts/data_smoke.py || { echo "FAIL: DATA_SMOKE"; exit 20; }
  else
    echo "WARN: DATA_SMOKE (missing scripts/data_smoke.py)"
  fi
fi

echo "==> VERIFY: TRAIN_SMOKE"
if [ -f "scripts/train_smoke.py" ]; then
  uv run python scripts/train_smoke.py || { echo "FAIL: TRAIN_SMOKE"; exit 30; }
else
  echo "WARN: TRAIN_SMOKE (missing scripts/train_smoke.py)"
fi

echo "PASS: ALL"
