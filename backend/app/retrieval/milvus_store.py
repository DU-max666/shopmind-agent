"""向量库接口层：Milvus（生产）与内存余弦检索（离线）。"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from backend.app.config import Settings, get_settings


@dataclass
class Hit:
    """一条向量召回结果。"""

    chunk_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, chunk_ids: list[str], vectors: list[list[float]],
               payloads: list[dict[str, Any]]) -> int: ...

    @abstractmethod
    def search(self, vector: list[float], top_k: int = 10) -> list[Hit]: ...


class InMemoryVectorStore(BaseVectorStore):
    """内存向量库：余弦相似度暴力检索，供离线评测/单测使用。"""

    def __init__(self) -> None:
        self._items: list[tuple[str, list[float], dict[str, Any]]] = []

    def upsert(self, chunk_ids: list[str], vectors: list[list[float]],
               payloads: list[dict[str, Any]]) -> int:
        existing = {cid for cid, _, _ in self._items}
        for cid, vec, payload in zip(chunk_ids, vectors, payloads):
            if cid in existing:
                self._items = [it for it in self._items if it[0] != cid]
            self._items.append((cid, vec, payload))
        return len(chunk_ids)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def search(self, vector: list[float], top_k: int = 10) -> list[Hit]:
        scored = [Hit(cid, self._cosine(vector, vec), payload) for cid, vec, payload in self._items]
        scored.sort(key=lambda h: (-h.score, h.chunk_id))
        return scored[:top_k]


class MilvusVectorStore(BaseVectorStore):  # pragma: no cover
    """Milvus 集合封装：私有化部署时的生产后端。"""

    def __init__(self, uri: str, collection: str, dim: int) -> None:
        from pymilvus import (CollectionSchema, DataType, FieldSchema, MilvusClient)

        self._client = MilvusClient(uri=uri)
        self._collection = collection
        if not self._client.has_collection(collection):
            schema = CollectionSchema([
                FieldSchema("pk", DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema("doc_id", DataType.VARCHAR, max_length=128),
                FieldSchema("doc_type", DataType.VARCHAR, max_length=32),
                FieldSchema("text", DataType.VARCHAR, max_length=8192),
            ])
            self._client.create_collection(collection_name=collection, schema=schema)

    def upsert(self, chunk_ids: list[str], vectors: list[list[float]],
               payloads: list[dict[str, Any]]) -> int:
        rows = [{"pk": cid, "vector": vec,
                 "doc_id": p.get("doc_id", ""), "doc_type": p.get("doc_type", ""),
                 "text": p.get("text", "")[:8000]} for cid, vec, p in zip(chunk_ids, vectors, payloads)]
        self._client.upsert(collection_name=self._collection, data=rows)
        return len(rows)

    def search(self, vector: list[float], top_k: int = 10) -> list[Hit]:
        res = self._client.search(collection_name=self._collection, data=[vector],
                                  limit=top_k, output_fields=["doc_id", "doc_type", "text"])
        hits = []
        for row in res[0]:
            hits.append(Hit(row["id"], row["distance"], {
                "doc_id": row["entity"].get("doc_id"),
                "doc_type": row["entity"].get("doc_type"),
                "text": row["entity"].get("text"),
            }))
        return hits


def get_vector_store(settings: Settings | None = None) -> BaseVectorStore:
    """配置了 MILVUS_URI 则连 Milvus，否则回退内存实现。"""
    s = settings or get_settings()
    if s.milvus_uri:
        try:
            return MilvusVectorStore(s.milvus_uri, s.milvus_collection, s.embedding_dim)
        except Exception:
            return InMemoryVectorStore()
    return InMemoryVectorStore()
