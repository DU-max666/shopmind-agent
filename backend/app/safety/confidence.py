"""板斧三：置信度阈值 → 回答 / 澄清 / 转人工 三态决策。

阈值来自配置（ROUTE_ANSWER_THRESHOLD / ROUTE_CLARIFY_THRESHOLD），
低于澄清线不硬答，直接生成追问；证据一致性低于 grounded 下限转人工。
"""
from __future__ import annotations

from dataclasses import dataclass

ANSWER = "answer"
CLARIFY = "clarify"
HUMAN = "human"


@dataclass
class ConfidenceDecision:
    action: str
    confidence: float
    reason: str


def decide(confidence: float, *, answer_threshold: float = 0.75,
           clarify_threshold: float = 0.55, grounded_score: float = 1.0,
           grounded_min: float = 0.8) -> ConfidenceDecision:
    """综合路由置信度与证据一致性，输出下一步动作。"""
    if grounded_score < grounded_min:
        return ConfidenceDecision(HUMAN, confidence,
                                  f"证据一致性 {grounded_score:.2f} 低于下限 {grounded_min}")
    if confidence >= answer_threshold:
        return ConfidenceDecision(ANSWER, confidence, "高置信，直接作答")
    if confidence >= clarify_threshold:
        return ConfidenceDecision(CLARIFY, confidence, "中置信，先向用户澄清")
    return ConfidenceDecision(HUMAN, confidence, "低置信，转人工避免误答")
