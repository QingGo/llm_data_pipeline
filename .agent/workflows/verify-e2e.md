---
description: verify e2e result
---

Run E2E done-loop:
- Ensure scripts/verify_e2e.sh exists (create from rule template if missing).
- Execute it.
- If it fails: follow the rule’s loop strictly until it passes.
- When it passes twice in a row: summarize outputs + artifact paths and declare DONE.
