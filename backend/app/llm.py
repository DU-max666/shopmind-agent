"""LLM 接口层：统一抽象 + OpenAI 兼容客户端（仅标准库）+ 离线 MockLLM。

- OpenAICompatLLM：通过 urllib 调用任意 OpenAI 兼容网关，支持私有化部署；
- MockLLM：确定性、可复现的本地替身，用于单测/评测/演示端，无密钥可跑通全链路。
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

from backend.app.config import Settings, get_settings


class LLMUnavailable(RuntimeError):
    """LLM 网关不可用（网络错误、鉴权失败、超时等）。"""


@dataclass
class ChatMessage:
    """一条对话消息。"""

    role: str  # system / user / assistant
    content: str


class BaseLLM(ABC):
    """LLM 提供方抽象基类。"""

    @abstractmethod
    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.2,
                 max_tokens: int = 1024) -> str:
        """输入消息列表，返回补全文本。"""


class OpenAICompatLLM(BaseLLM):
    """OpenAI Chat Completions 兼容客户端，仅依赖标准库 urllib。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.2,
                 max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:  # pragma: no cover
            raise LLMUnavailable(f"LLM 网关调用失败: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover
            raise LLMUnavailable(f"LLM 返回结构异常: {body}") from exc


# MockLLM 意图关键词表：仅作离线兜底分类器，刻意保持与规则层不同的朴素实现，
# 以便评测时体现"规则前置 + LLM 兜底"两层互补。
_MOCK_INTENT_KEYWORDS: dict[str, list[str]] = {
    "logistics": ["物流", "快递", "运单", "派送", "签收", "发货", "到哪", "什么时候到"],
    "after_sale": ["退货", "换货", "退款", "质量", "破损", "坏了", "投诉", "工单", "维修", "保修"],
    "promo": ["价保", "保价", "差价", "降价", "优惠券", "满减", "活动", "大促", "双十一", "双11", "618"],
    "guide": ["推荐", "想买", "有没有", "多少钱", "价格", "对比", "选", "尺寸", "材质", "款式"],
}


class MockLLM(BaseLLM):
    """确定性离线 LLM 替身。

    识别两类提示词标记：
    1. ``意图分类``：按朴素关键词计分，输出 JSON ``{"intent", "confidence"}``；
    2. ``EVIDENCE_BEGIN``：从证据块首行抽取事实，拼装带引用编号的受约束答案，
       用于离线跑通"生成 + 引用溯源"链路（答案中的数字全部来自证据，天然防幻觉）。
    """

    def complete(self, messages: list[ChatMessage], *, temperature: float = 0.2,
                 max_tokens: int = 1024) -> str:
        content = messages[-1].content if messages else ""
        if "意图分类" in content:
            return self._classify(content)
        if '"consistent"' in content:
            return self._quality_check(content)
        if "EVIDENCE_BEGIN" in content:
            return self._compose_from_evidence(content)
        return "好的，我来帮您处理。如需进一步信息，请补充订单号或商品名称。"

    @staticmethod
    def _classify(content: str) -> str:
        m = re.search(r"用户问题[：:]\s*(.+)", content)
        text = m.group(1) if m else content
        scores: dict[str, float] = {}
        for intent, words in _MOCK_INTENT_KEYWORDS.items():
            scores[intent] = sum(1.0 for w in words if w in text)
        best_intent = max(scores, key=lambda k: scores[k]) if scores else "clarify"
        best_score = scores.get(best_intent, 0.0)
        if best_score == 0:
            return json.dumps({"intent": "clarify", "confidence": 0.3}, ensure_ascii=False)
        total = sum(scores.values()) or 1.0
        confidence = min(0.9, 0.45 + 0.45 * best_score / total)
        return json.dumps({"intent": best_intent, "confidence": round(confidence, 2)},
                          ensure_ascii=False)

    @staticmethod
    def _compose_from_evidence(content: str) -> str:
        m = re.search(r"EVIDENCE_BEGIN\n(.*?)\nEVIDENCE_END", content, re.S)
        block = m.group(1).strip() if m else ""
        first = block.splitlines()[0] if block else ""
        mm = re.match(r"\[([A-Za-z0-9\-]+)\]\s*(.*)", first)
        if mm:
            doc_id, body = mm.group(1), mm.group(2)
            return f"根据资料，{body}【{doc_id}】"
        return "抱歉，当前资料不足，无法给出可靠答案，已为您转接人工客服。"

    @staticmethod
    def _quality_check(content: str) -> str:
        """离线质检替身：答案中的数字必须全部出现在证据块中，否则判不一致。"""
        m = re.search(r"EVIDENCE_BEGIN\n(.*?)\nEVIDENCE_END", content, re.S)
        evidence = m.group(1) if m else ""
        answer = content.split("答案：", 1)[-1] if "答案：" in content else ""
        q_nums = set(re.findall(r"\d+(?:\.\d+)?", answer))
        e_nums = set(re.findall(r"\d+(?:\.\d+)?", evidence))
        if q_nums and not q_nums <= e_nums:
            return json.dumps({"consistent": False, "reason": "答案数字未在证据中出现"},
                              ensure_ascii=False)
        return json.dumps({"consistent": True, "reason": "离线 Mock 复核：数字回查通过"},
                          ensure_ascii=False)


def get_llm(settings: Settings | None = None) -> BaseLLM:
    """按配置返回 LLM 实例：有密钥走真实网关，否则返回 MockLLM。"""
    s = settings or get_settings()
    if s.llm_base_url and s.llm_api_key:
        return OpenAICompatLLM(s.llm_base_url, s.llm_api_key, s.llm_model, s.llm_timeout_s)
    return MockLLM()
