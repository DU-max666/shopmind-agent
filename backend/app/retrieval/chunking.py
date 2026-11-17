"""三种面向电商知识形态的切分策略。

1. 商品说明书 → 父子块：父块=章节（保留完整语境），子块=段落（供召回），
   命中子块后回填父块文本进入提示词，缓解"规格参数被切断"导致的幻觉；
2. 售后/价保政策 → 条款级：按"第X条"切分，引用可精确到条款号；
3. FAQ → 问答对：一问一答为一个块，答案天然完整。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_CLAUSE_RE = re.compile(r"(第[一二三四五六七八九十百]+条)")
_FAQ_Q_RE = re.compile(r"问[：:]")


@dataclass
class Chunk:
    """检索块：id 全局唯一；parent_id 用于父子块回填。"""

    chunk_id: str
    doc_id: str
    doc_type: str            # manual / policy / faq
    text: str
    parent_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def chunk_manual_parent_child(doc_id: str, content: str) -> list[Chunk]:
    """说明书按 `# 标题` 分父块，父块内按空行分段落子块。"""
    chunks: list[Chunk] = []
    sections = re.split(r"^#\s+", content, flags=re.M)
    for si, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        lines = section.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        parent_id = f"{doc_id}#S{si}"
        chunks.append(Chunk(chunk_id=parent_id, doc_id=doc_id, doc_type="manual",
                            text=f"{title}\n{body}", meta={"title": title, "is_parent": True}))
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for pi, para in enumerate(paragraphs):
            chunks.append(Chunk(chunk_id=f"{parent_id}#P{pi}", doc_id=doc_id, doc_type="manual",
                                text=para, parent_id=parent_id,
                                meta={"title": title, "is_parent": False}))
    return chunks


def chunk_policy_clauses(doc_id: str, content: str) -> list[Chunk]:
    """政策按"第X条"切分为条款块，条款号写入 meta 便于引用溯源。"""
    chunks: list[Chunk] = []
    parts = _CLAUSE_RE.split(content)
    header = parts[0].strip()
    if header:
        chunks.append(Chunk(chunk_id=f"{doc_id}#H", doc_id=doc_id, doc_type="policy",
                            text=header, meta={"clause": "总则"}))
    for i in range(1, len(parts) - 1, 2):
        clause_no, body = parts[i], parts[i + 1].strip()
        chunks.append(Chunk(chunk_id=f"{doc_id}#{clause_no}", doc_id=doc_id, doc_type="policy",
                            text=f"{clause_no} {body}", meta={"clause": clause_no}))
    return chunks


def chunk_faq_pairs(doc_id: str, content: str) -> list[Chunk]:
    """FAQ 按"问："切分为问答对块。"""
    chunks: list[Chunk] = []
    blocks = _FAQ_Q_RE.split(content)
    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        parts = re.split(r"答[：:]", block, maxsplit=1)
        question = parts[0].strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        chunks.append(Chunk(chunk_id=f"{doc_id}#Q{i}", doc_id=doc_id, doc_type="faq",
                            text=f"问：{question}\n答：{answer}",
                            meta={"question": question}))
    return chunks


def chunk_document(doc_id: str, doc_type: str, content: str) -> list[Chunk]:
    """统一入口：按文档类型路由到对应切分器。"""
    if doc_type == "manual":
        return chunk_manual_parent_child(doc_id, content)
    if doc_type == "policy":
        return chunk_policy_clauses(doc_id, content)
    if doc_type == "faq":
        return chunk_faq_pairs(doc_id, content)
    return [Chunk(chunk_id=f"{doc_id}#0", doc_id=doc_id, doc_type=doc_type, text=content)]
