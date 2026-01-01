---
trigger: always_on
---

# Done-Loop (E2E) Rule — housing-data + training-framework

Goal:
- No human-in-the-loop. Keep running until DONE.
- DONE is ONLY when ./run_pipeline.sh --limit 1000 --output-base outputs/quick_test exits 0.

Hard Loop:
1) Run: ./run_pipeline.sh --limit 1000 --output-base outputs/quick_test
2) If FAIL:
   - Read the failing section name printed by the script (DATA_SMOKE / TRAIN_SMOKE / TESTS).
   - Diagnose root cause from logs.
   - Apply smallest fix.
   - (Allowed) add TEMPLOG for diagnosis, but:
       - Must be behind env var DEBUG_E2E=1 or a clear "TEMPLOG:" prefix.
       - Before final DONE, remove TEMPLOG or gate it off by default.
   - Go back to step (2).
3) If PASS:
   - Enforce cleanup checks:
       - grep -R "TEMPLOG:" -n ./src  => must find nothing
       - grep -R "DEBUG_E2E" -n ./src  => allowed only if default-off and documented
   - Re-run verify once more (to ensure no flake).
   - Then STOP and report DONE with:
       - command output summary (1 screen max)
       - artifacts produced paths (e.g., _artifacts/e2e/*)

Safety + Autonomy:
- Prefer terminal execution policy Auto/Turbo; never use destructive commands unless explicitly required.
- If Turbo is enabled, respect a denylist (rm/rmdir/sudo/curl/wget) and use safe alternatives when possible.