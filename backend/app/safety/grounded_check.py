"""板斧一：生成句与召回证据的一致性校验。

策略：把答案按句切分，凡含数字/金额/时限的"事实句"，其全部数字必须能在
证据文本中找到，否则记为不可溯源句。电商场景的幻觉高发区正是
价格、库存、尺寸、退换时限——数字可核对，比语义相似度更硬。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SENT_SPLIT = re.compile(r"[。！？；!?;\n]+")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class GroundCheckResult:
    """一致性校验结果：score = 可溯源句 / 总句数。"""

    score: float
    total_claims: int
    supported_claims: int
    unsupported: list[str] = field(default_factory=list)

    def passed(self, threshold: float) -> bool:
        return self.score >= threshold


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _evidence_digits(evidence: list[dict[str, Any]] | list[str]) -> set[str]:
    joined = []
    for item in evidence:
        joined.append(item["text"] if isinstance(item, dict) else str(item))
    return set(_NUM_RE.findall(" ".join(joined)))


def check(answer: str, evidence: list[dict[str, Any]] | list[str]) -> GroundCheckResult:
    """校验 answer 中每个数字事实是否被 evidence 支撑。"""
    sentences = _sentences(answer)
    if not sentences:
        return GroundCheckResult(score=1.0, total_claims=0, supported_claims=0)
    ev_digits = _evidence_digits(evidence)
    supported = 0
    unsupported: list[str] = []
    for sent in sentences:
        digits = set(_NUM_RE.findall(sent))
        if not digits:
            supported += 1  # 无数字的客套/引导句视为可溯源
            continue
        if digits <= ev_digits:
            supported += 1
        else:
            unsupported.append(sent)
    return GroundCheckResult(
        score=round(supported / len(sentences), 4),
        total_claims=len(sentences),
        supported_claims=supported,
        unsupported=unsupported,
    )
