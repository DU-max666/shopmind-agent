"""板斧二：引用溯源与回查。

约定答案中的引用格式为 `【doc_id】`（doc_id 如 M-ZP-SF-001 / POLICY-RMA / FAQ-004）。
回查逻辑：引用编号必须存在于本轮召回证据中，且该句数字须命中对应证据块，
防止"引用了一个真实编号但答非所据"的伪溯源。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CITE_RE = re.compile(r"【([A-Za-z0-9][A-Za-z0-9\-]*)】")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class CitationResult:
    ok: bool
    used: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)   # 不在证据集中的编号（编造引用）
    unverified: list[str] = field(default_factory=list)  # 编号存在但数字对不上（答非所据）
    missing: bool = False  # 含事实句但完全无引用


def extract_citations(answer: str) -> list[str]:
    return CITE_RE.findall(answer)


def verify(answer: str, evidence: list[dict[str, Any]]) -> CitationResult:
    """回查答案引用。evidence 为 [{doc_id, text}] 列表。"""
    evidence_map: dict[str, set[str]] = {}
    for item in evidence:
        doc_id = str(item.get("doc_id", ""))
        evidence_map.setdefault(doc_id, set()).update(_NUM_RE.findall(str(item.get("text", ""))))
    citations = extract_citations(answer)
    used = list(dict.fromkeys(citations))
    unknown = [c for c in used if c not in evidence_map]
    unverified: list[str] = []
    # 数字核对：答案中每个引用句的数字，须在该引用对应证据块内出现
    for sent in re.split(r"[。！？；!?;\n]+", answer):
        cites = CITE_RE.findall(sent)
        if not cites:
            continue
        digits = set(_NUM_RE.findall(sent))
        if not digits:
            continue
        pool: set[str] = set()
        for cite in cites:
            pool |= evidence_map.get(cite, set())
        if not digits <= pool:
            for cite in cites:
                if cite in evidence_map and cite not in unverified:
                    unverified.append(cite)
    has_facts = bool(_NUM_RE.search(answer))
    missing = has_facts and not citations
    ok = not unknown and not unverified and not missing
    return CitationResult(ok=ok, used=used, unknown=unknown, unverified=unverified, missing=missing)
