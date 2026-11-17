"""精排层：BGE-Reranker（生产可选）与词面重合精排（离线替身）。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.retrieval.embedding import tokenize


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """返回与 texts 对齐的相关性分数（越大越相关）。"""


class LexicalOverlapReranker(BaseReranker):
    """词面重合精排：query 与候选文本的 token 覆盖率 × 长度平滑。

    对"规格/条款号"这类强词面信号（如 120cm、第七条）效果接近轻量交叉编码器，
    作为 BGE-Reranker 的离线替身，保证精排链路在无模型环境同样可测。
    """

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return [0.0] * len(texts)
        from collections import Counter

        q_counter = Counter(q_tokens)
        scores: list[float] = []
        for text in texts:
            d_counter = Counter(tokenize(text))
            overlap = sum(min(q_counter[t], d_counter.get(t, 0)) for t in q_counter)
            length_norm = 1.0 + 0.004 * max(0, len(text) - 60)
            scores.append(overlap / (len(q_tokens) * length_norm))
        return scores


class BGEReranker(BaseReranker):  # pragma: no cover
    """BGE-Reranker 本地交叉编码器，私有化部署时启用。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        from FlagEmbedding import FlagReranker

        self._model = FlagReranker(model_name, use_fp16=True)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.compute_score([[query, t] for t in texts])]


def get_reranker(prefer_local: bool = False) -> BaseReranker:
    if prefer_local:
        try:
            return BGEReranker()
        except Exception:
            pass
    return LexicalOverlapReranker()
