---
trigger: always_on
---

# Fail-fast imports & exceptions

- 禁止用 try/except 来“探测依赖是否存在”或“吞掉严重异常”。
  - 禁止：except ImportError: pass
  - 禁止：except Exception: logger.warning(...); continue
- 任何影响正确性的依赖缺失/初始化失败：
  - 必须立刻 raise（带可执行的修复提示），不得静默降级。
- 允许捕获异常的唯一理由：补充上下文后立刻 re-raise（不得继续执行）。
  - 允许：catch specific -> raise RuntimeError(...) from e
  - 允许：logger.exception(...); raise
- “可选功能”必须显式 gated：
  - 只有在 feature_flag/config 启用时才检查依赖；若启用但依赖缺失 -> raise
  - 若未启用 -> 直接 return（不做 try/except 探测）
