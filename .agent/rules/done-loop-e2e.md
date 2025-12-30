---
trigger: always_on
---

# Done-Loop (E2E) Rule — housing-data + training-framework

Goal:
- No human-in-the-loop. Keep running until DONE.
- DONE is ONLY when scripts/verify_e2e.sh exits 0.

Hard Loop:
1) If scripts/verify_e2e.sh is missing, create it from the template embedded in this rule (see below).
2) Run: bash scripts/verify_e2e.sh
3) If FAIL:
   - Read the failing section name printed by the script (DATA_SMOKE / TRAIN_SMOKE / TESTS).
   - Diagnose root cause from logs.
   - Apply smallest fix.
   - (Allowed) add TEMPLOG for diagnosis, but:
       - Must be behind env var DEBUG_E2E=1 or a clear "TEMPLOG:" prefix.
       - Before final DONE, remove TEMPLOG or gate it off by default.
   - Go back to step (2).
4) If PASS:
   - Enforce cleanup checks:
       - grep -R "TEMPLOG:" -n .  => must find nothing
       - grep -R "DEBUG_E2E" -n .  => allowed only if default-off and documented
   - Re-run verify once more (to ensure no flake).
   - Then STOP and report DONE with:
       - command output summary (1 screen max)
       - artifacts produced paths (e.g., _artifacts/e2e/*)

Safety + Autonomy:
- Prefer terminal execution policy Auto/Turbo; never use destructive commands unless explicitly required.
- If Turbo is enabled, respect a denylist (rm/rmdir/sudo/curl/wget) and use safe alternatives when possible.

--- verify script template (create if missing) ---
Create file: scripts/verify_e2e.sh

#!/usr/bin/env bash
set -euo pipefail

echo "==> VERIFY: TESTS"
if [ -f "pyproject.toml" ]; then
  python -m pytest -q || { echo "FAIL: TESTS"; exit 10; }
fi

echo "==> VERIFY: DATA_SMOKE"
# Must be fast (<60s). Use tiny fixture, produce deterministic artifact.
# Implement one of the following commands in your repo:
#   python -m app.data.smoke --in tests/fixtures/housing.parquet --out _artifacts/e2e/data.parquet
# OR
#   python scripts/data_smoke.py
if python -c "import importlib; import sys; sys.exit(0)"; then
  if [ -f "scripts/data_smoke.py" ]; then
    python scripts/data_smoke.py || { echo "FAIL: DATA_SMOKE"; exit 20; }
  else
    echo "FAIL: DATA_SMOKE (missing scripts/data_smoke.py)"; exit 21
  fi
fi

echo "==> VERIFY: TRAIN_SMOKE"
# Must be fast (<120s). Tiny config, few steps, CPU ok.
# Implement one of:
#   python -m app.train.smoke --max_steps 10 --out _artifacts/e2e/train/
# OR
#   python scripts/train_smoke.py
if [ -f "scripts/train_smoke.py" ]; then
  python scripts/train_smoke.py || { echo "FAIL: TRAIN_SMOKE"; exit 30; }
else
  echo "FAIL: TRAIN_SMOKE (missing scripts/train_smoke.py)"; exit 31
fi

echo "PASS: ALL"