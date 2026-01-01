---
description: verify e2e result
---

Run E2E done-loop:
- Execute "./run_pipeline.sh --limit 1000 --output-base outputs/quick_test".
- If it fails: follow the rule’s loop strictly until it passes.
- When it passes twice in a row: summarize outputs + artifact paths and declare DONE.