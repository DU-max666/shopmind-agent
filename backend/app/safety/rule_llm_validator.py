"""板斧四：规则 + LLM 协同校验。

分工：规则层负责"零成本、可解释"的硬约束（违禁话术、承诺性表述、缺引用）；
LLM 层负责规则写不全的语义一致性（答案是否曲解证据）。规则不过直接拦截，
LLM 仅在规则通过后抽样复核，控制时延与成本。离线环境 LLM 为 MockLLM，
其"一致性复核"退化为再跑一遍数字回查，行为可测。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from backend.app.llm import BaseLLM, ChatMessage
from backend.app.safety import citation, grounded_check

# 电商客服违禁/风险话术：绝对化承诺与编造库存价格的高危信号
BANNED_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("绝对化承诺", re.compile(r"绝对|百分百|一定没问题|保证最低价|随时都能退")),
    ("无据库存承诺", re.compile(r"库存充足可以随便买|想要多少有多少")),
    ("越权承诺", re.compile(r"我已经给您退了|款项肯定已到账|马上打款")),
]


@dataclass
class ValidationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    llm_checked: bool = False
    llm_verdict: dict[str, Any] | None = None


def rule_check(answer: str, evidence: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for label, pattern in BANNED_PATTERNS:
        if pattern.search(answer):
            issues.append(f"规则违禁：{label}")
    ground = grounded_check.check(answer, evidence)
    if ground.score < 0.8:
        issues.append(f"证据一致性不足：{ground.score:.2f}")
    cite = citation.verify(answer, evidence)
    if cite.unknown:
        issues.append(f"编造引用：{', '.join(cite.unknown)}")
    if cite.missing:
        issues.append("含事实数字但无引用编号")
    return issues


def llm_check(answer: str, evidence: list[dict[str, Any]], llm: BaseLLM) -> dict[str, Any]:
    prompt = (
        "你是电商客服答案质检员。请判断【答案】是否与【证据】一致，"
        "是否存在编造价格/库存/时限。只输出 JSON：{\"consistent\": true/false, \"reason\": \"...\"}\n"
        "EVIDENCE_BEGIN\n" + "\n".join(f"[{e.get('doc_id')}] {e.get('text')}" for e in evidence)
        + "\nEVIDENCE_END\n答案：" + answer
    )
    raw = llm.complete([ChatMessage("user", prompt)])
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"consistent": False, "reason": "LLM 校验输出不可解析"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"consistent": False, "reason": "LLM 校验 JSON 解析失败"}


def validate(answer: str, evidence: list[dict[str, Any]], llm: BaseLLM | None = None,
             use_llm: bool = True) -> ValidationResult:
    """先规则后 LLM 的协同校验入口。"""
    issues = rule_check(answer, evidence)
    if issues:
        return ValidationResult(passed=False, issues=issues)
    if use_llm and llm is not None and evidence:
        verdict = llm_check(answer, evidence, llm)
        if not verdict.get("consistent", False):
            return ValidationResult(passed=False,
                                    issues=[f"LLM 复核不一致：{verdict.get('reason', '')}"],
                                    llm_checked=True, llm_verdict=verdict)
        return ValidationResult(passed=True, llm_checked=True, llm_verdict=verdict)
    return ValidationResult(passed=True)
