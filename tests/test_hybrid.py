"""混合检索链单测：RRF 融合、双路召回、父子块回填、证据区拼装。"""
from __future__ import annotations

from backend.app.retrieval.chunking import chunk_document
from backend.app.retrieval.embedding import HashEmbedder, tokenize
from backend.app.retrieval.hybrid import HybridRetriever, rrf_fuse
from backend.app.retrieval.milvus_store import InMemoryVectorStore
from backend.app.retrieval.rerank import LexicalOverlapReranker

MANUAL = """\
# 产品概述
型号 ZP-ZC-001，天然乳胶枕，乳胶含量 93%。

ZP-ZC-001 售价 249 元，提供 10cm 与 12cm 两种枕高。
"""
POLICY = """\
退货政策

第一条 签收后 7 天内可无理由退货。
"""
FAQ = "问：发票怎么开？\n答：确认收货后在订单页申请，1 个工作日开出（FAQ-004）。"


def make_retriever() -> HybridRetriever:
    chunks = (
        chunk_document("M-ZP-ZC-001", "manual", MANUAL)
        + chunk_document("POLICY-RMA", "policy", POLICY)
        + chunk_document("FAQ-004", "faq", FAQ)
    )
    retriever = HybridRetriever(
        HashEmbedder(dim=64), InMemoryVectorStore(), LexicalOverlapReranker()
    )
    retriever.add_documents(chunks)
    return retriever


def test_tokenize_cjk_bigram():
    tokens = tokenize("乳胶枕")
    assert "乳胶" in tokens and "胶枕" in tokens


def test_rrf_fuse_prefers_consensus():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    top = dict(fused)
    assert top["b"] > top["c"]  # 两路共识优于单路第一


def test_search_finds_manual_and_backfills_parent():
    retriever = make_retriever()
    hits = retriever.search("乳胶枕 售价 249", top_k=3)
    doc_ids = {h.doc_id for h in hits}
    assert "M-ZP-ZC-001" in doc_ids
    hit = next(h for h in hits if h.doc_id == "M-ZP-ZC-001")
    # 子块命中后回填父块：章节标题应出现在返回文本里
    assert "产品概述" in hit.text
    # meta.snippet 保留子块原文，供证据引用回查
    assert "ZP-ZC-001" in hit.meta["snippet"]


def test_search_policy_clause():
    retriever = make_retriever()
    hits = retriever.search("退货 签收 几天", top_k=2)
    assert any(h.doc_id == "POLICY-RMA" for h in hits)


def test_evidence_block_format():
    retriever = make_retriever()
    hits = retriever.search("乳胶枕 售价", top_k=3)
    block = retriever.evidence_block(hits)
    lines = [ln for ln in block.splitlines() if ln.strip()]
    assert lines and all(ln.startswith("[") for ln in lines)
    # 关键句优先取含数字的行
    assert any(any(ch.isdigit() for ch in ln) for ln in lines)
