"""混合检索层：BGE 语义向量（Milvus）+ BM25 词面 + RRF 融合 + BGE-Reranker 精排。

离线模式使用确定性 HashEmbedder + 内存向量库 + 词面重合精排，保证无 GPU/无模型也能评测。
"""
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.chunking import (
    Chunk, chunk_faq_pairs, chunk_manual_parent_child, chunk_policy_clauses,
)
from backend.app.retrieval.embedding import BaseEmbedder, HashEmbedder, get_embedder
from backend.app.retrieval.hybrid import HybridRetriever, SearchHit, rrf_fuse
from backend.app.retrieval.milvus_store import InMemoryVectorStore, BaseVectorStore, Hit
from backend.app.retrieval.rerank import BaseReranker, LexicalOverlapReranker

__all__ = [
    "BM25Index", "Chunk", "chunk_faq_pairs", "chunk_manual_parent_child",
    "chunk_policy_clauses", "BaseEmbedder", "HashEmbedder", "get_embedder",
    "HybridRetriever", "SearchHit", "rrf_fuse", "InMemoryVectorStore",
    "BaseVectorStore", "Hit", "BaseReranker", "LexicalOverlapReranker",
]
