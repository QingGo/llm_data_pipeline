"""清洗规则与度量：判定文本是否保留并给出原因与指标"""

import re
from dataclasses import dataclass
from typing import Dict, Tuple

_NON_WS_RE = re.compile(r"\S")
_ALPHA_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_PUNCT_RE = re.compile(r"[^\w\s\u4e00-\u9fff]", re.UNICODE)


@dataclass(frozen=True)
class CleanRules:
    """判定文本质量的阈值集合"""
    min_chars: int = 200
    max_chars: int = 200_000
    min_non_ws_ratio: float = 0.7
    min_alpha_cjk_ratio: float = 0.4
    max_punct_ratio: float = 0.25
    max_dup_line_ratio: float = 0.35


def basic_clean(text: str) -> str:
    """统一换行并压缩多余空行"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def _ratios(text: str) -> Dict[str, float]:
    """计算非空白、字母+CJK、标点所占比例"""
    n = len(text)
    if n <= 0:
        return {"non_ws": 0.0, "alpha_cjk": 0.0, "punct": 0.0}

    non_ws = len(_NON_WS_RE.findall(text)) / n
    alpha_cjk = (len(_ALPHA_RE.findall(text)) + len(_CJK_RE.findall(text))) / n
    punct = len(_PUNCT_RE.findall(text)) / n
    return {"non_ws": non_ws, "alpha_cjk": alpha_cjk, "punct": punct}


def _dup_line_ratio(text: str) -> float:
    """估计重复行在有效行中的比例"""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 5:
        return 0.0
    freq: Dict[str, int] = {}
    for ln in lines:
        freq[ln] = freq.get(ln, 0) + 1
    dup = sum(v for v in freq.values() if v > 1)
    return dup / max(1, len(lines))


def judge(text: str, rules: CleanRules) -> Tuple[bool, str, Dict[str, float]]:
    """应用规则判断文本是否保留，并返回原因与度量

    返回：
      - 是否保留
      - 原因字符串
      - 指标字典（含非空白/语言信号/标点/重复行比例）
    """
    if not text or len(text) < rules.min_chars:
        return False, "too_short", {"len": float(len(text or ""))}
    if len(text) > rules.max_chars:
        return False, "too_long", {"len": float(len(text))}

    r = _ratios(text)
    if r["non_ws"] < rules.min_non_ws_ratio:
        return False, "too_sparse", r
    if r["alpha_cjk"] < rules.min_alpha_cjk_ratio:
        return False, "low_language_signal", r
    if r["punct"] > rules.max_punct_ratio:
        return False, "too_much_punct", r

    d = _dup_line_ratio(text)
    if d > rules.max_dup_line_ratio:
        r2 = dict(r)
        r2["dup_line"] = d
        return False, "dup_lines", r2

    r2 = dict(r)
    r2["dup_line"] = d
    return True, "ok", r2
