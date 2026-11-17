"""混合检索编排：稠密(Milvus) + 稀疏(BM25) → RRF 融合 → 精排 → 父子块回填。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.chunking import Chunk
from backend.app.retrieval.embedding import BaseEmbedder
from backend.app.retrieval.milvus_store import BaseVectorStore
from backend.app.retrieval.rerank import BaseReranker


@dataclass
class SearchHit:
    """融合与精排后的最终命中：doc_id 用于引用溯源，text 进入提示词。"""

    chunk_id: str
    doc_id: str
    doc_type: str
    text: str
    score: float
    meta: dict[str, Any] = field(default_factory=dict)


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion：score(d)=Σ 1/(k+rank)。

    对两路召回的分数尺度完全免疫（向量余弦与 BM25 分数不可比），
    是本项目在电商说明书场景替代加权求和的默认融合方式。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    """把切分、双路索引、融合、精排、父块回填串成一条可离线运行的检索链。"""

    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore,
                 reranker: BaseReranker | None = None, rrf_k: int = 60) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.bm25 = BM25Index()
        self._chunks: dict[str, Chunk] = {}

    def add_documents(self, chunks: list[Chunk]) -> int:
        leaf = [c for c in chunks if not c.meta.get("is_parent")]
        parents = [c for c in chunks if c.meta.get("is_parent")]
        for c in chunks:
            self._chunks[c.chunk_id] = c
        for c in leaf:
            self.bm25.add(c.chunk_id, c.text)
        vectors = self.embedder.embed([c.text for c in leaf]) if leaf else []
        payloads = [{"doc_id": c.doc_id, "doc_type": c.doc_type, "text": c.text} for c in leaf]
        self.store.upsert([c.chunk_id for c in leaf], vectors, payloads)
        # 父块只进 BM25 之外的回填字典，避免段落与章节重复召回
        _ = parents
        return len(leaf)

    def _resolve_text(self, chunk: Chunk) -> str:
        """子块命中后回填父块全文，保证规格参数语境完整。"""
        if chunk.parent_id and chunk.parent_id in self._chunks:
            return self._chunks[chunk.parent_id].text
        return chunk.text

    def search(self, query: str, top_k: int = 5, candidate_k: int = 10) -> list[SearchHit]:
        dense_ids = [h.chunk_id for h in self.store.search(self.embedder.embed([query])[0], candidate_k)]
        sparse_ids = [cid for cid, _ in self.bm25.search(query, candidate_k)]
        fused = rrf_fuse([dense_ids, sparse_ids], k=self.rrf_k)[: candidate_k * 2]
        hits: list[SearchHit] = []
        for chunk_id, score in fused:
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            meta = dict(chunk.meta)
            meta["snippet"] = chunk.text  # 保留子块原文，供证据区与引用回查
            hits.append(SearchHit(chunk_id=chunk_id, doc_id=chunk.doc_id, doc_type=chunk.doc_type,
                                  text=self._resolve_text(chunk), score=score, meta=meta))
        if self.reranker is not None and hits:
            scores = self.reranker.rerank(query, [h.text for h in hits])
            for h, s in zip(hits, scores):
                h.score = s
            hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[:top_k]

    def evidence_block(self, hits: list[SearchHit], max_chars: int = 1200) -> str:
        """把命中块拼成提示词证据区：每行 `[doc_id] 关键句`，供生成端引用。

        关键句优先取子块中含数字的行（规格/价格/时限），保证引用回查与
        幻觉校验层能拿"可核对的事实"而不是标题。
        """
        lines = []
        budget = max_chars
        for hit in hits:
            snippet = hit.meta.get("snippet") or hit.text
            candidate_lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
            key_line = next((ln for ln in candidate_lines if any(ch.isdigit() for ch in ln)),
                            candidate_lines[0] if candidate_lines else "")
            line = f"[{hit.doc_id}] {key_line}"
            if len(line) > budget:
                break
            lines.append(line)
            budget -= len(line)
        return "\n".join(lines)
