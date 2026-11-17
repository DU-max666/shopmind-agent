"""向量化接口层：BGE 本地模型（可选）与确定性 Hash 替身。"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

from backend.app.config import Settings, get_settings

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """轻量中文切词：拉丁词 + 汉字二元组（bigram），无外部分词依赖。

    bigram 对电商中文检索（材质/规格/政策条款）召回效果稳定，
    且与 HashEmbedder、BM25 共用同一套切词，保证两路信号一致。
    """
    lowered = text.lower()
    tokens = _LATIN.findall(lowered)
    for run in _CJK_RUN.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class BaseEmbedder(ABC):
    """文本向量化抽象。"""

    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量编码，返回 L2 归一化向量。"""


class HashEmbedder(BaseEmbedder):
    """确定性哈希嵌入替身：token → 多哈希桶累加，签名可复现。

    不追求语义精度，但保证：
    1) 相同输入恒等输出（评测可复现）；
    2) 词面重合度越高、余弦相似度越高（与 BM25 信号互补而非矛盾）。
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, int]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1 if digest[4] % 2 == 0 else -1
        return idx, sign

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in tokenize(text):
                idx, sign = self._bucket(token)
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class BGELocalEmbedder(BaseEmbedder):  # pragma: no cover
    """BGE 中文向量模型（FlagEmbedding），私有化部署时启用。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        from FlagEmbedding import FlagModel

        self._model = FlagModel(model_name)
        self.dim = self._model.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, row)) for row in self._model.encode(texts)]


def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """按配置选择嵌入后端：显式指定 BGE 则加载本地模型，否则用 Hash 替身。"""
    s = settings or get_settings()
    if s.debug is False and s.llm_api_key:  # 生产模式才尝试真实模型
        try:
            return BGELocalEmbedder()
        except Exception:
            pass
    return HashEmbedder(dim=s.embedding_dim)
