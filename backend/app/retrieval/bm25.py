"""BM25 稀疏检索：纯 Python 实现（Okapi BM25），零第三方依赖。"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from backend.app.retrieval.embedding import tokenize


class BM25Index:
    """可增量的 BM25 索引：doc_id → 词频表 + 全局统计。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_tokens: dict[str, Counter] = {}
        self._doc_len: dict[str, int] = {}
        self._df: Counter = Counter()
        self._total_len = 0

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        tf = Counter(tokens)
        self._doc_tokens[doc_id] = tf
        self._doc_len[doc_id] = len(tokens)
        for term in tf:
            self._df[term] += 1
        self._total_len += len(tokens)

    @property
    def size(self) -> int:
        return len(self._doc_tokens)

    def _idf(self, term: str) -> float:
        n = self.size
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """返回按 BM25 分数降序的 (doc_id, score) 列表。"""
        if not self._doc_tokens:
            return []
        avg_len = self._total_len / self.size
        scores: dict[str, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for doc_id, tf in self._doc_tokens.items():
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                norm = freq * (self.k1 + 1) / (
                    freq + self.k1 * (1 - self.b + self.b * self._doc_len[doc_id] / avg_len)
                )
                scores[doc_id] += idf * norm
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]
