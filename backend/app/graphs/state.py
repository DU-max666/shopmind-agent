"""共享状态定义：Agent 状态字典、意图/工单枚举、运行上下文。

约定：
- ShopMindState 为 TypedDict（LangGraph 兼容），所有节点函数以"增量 dict"返回更新；
- AgentContext 聚合一次会话所需的 LLM、检索器、工具集与配置，方便离线注入 Mock。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

from backend.app.config import Settings, get_settings
from backend.app.llm import BaseLLM, get_llm


class Intent(str, Enum):
    """顶层意图：四个业务子图 + 澄清。"""

    GUIDE = "guide"            # 售前导购
    AFTER_SALE = "after_sale"  # 售后工单
    LOGISTICS = "logistics"    # 物流查询
    PROMO = "promo"            # 活动/价保
    CLARIFY = "clarify"        # 低置信：向用户追问


class TicketStatus(str, Enum):
    """售后工单状态机节点。"""

    INTAKE = "intake"                        # 识别诉求
    POLICY_CHECK = "policy_check"            # 检索政策依据
    SUGGESTION = "suggestion"                # 生成处理建议
    CREATED = "created"                      # 已创建工单
    AWAITING_CONFIRM = "awaiting_confirm"    # 高风险动作待人工确认
    ESCALATED = "escalated"                  # 已升级人工
    RESOLVED = "resolved"
    REJECTED = "rejected"


class ShopMindState(TypedDict, total=False):
    """一次对话轮次在 Supervisor 与子图之间流转的全量状态。"""

    session_id: str
    user_id: str
    user_text: str
    messages: list[dict[str, str]]
    intent: str
    intent_confidence: float
    route_source: str          # rule / llm / clarify
    slots: dict[str, Any]      # 槽位：订单号、SKU、金额、预算等
    evidence: list[dict[str, Any]]   # 召回证据 [{doc_id, text, score}]
    tool_results: list[dict[str, Any]]
    draft_answer: str
    final_answer: str
    grounded_score: float
    needs_human: bool
    human_reason: str
    ticket_id: str
    ticket_status: str


@dataclass
class AgentContext:
    """子图运行上下文：LLM + 检索器 + 工具集 + 配置。"""

    llm: BaseLLM
    settings: Settings
    retriever: Any = None  # HybridRetriever 或 None（离线降级）
    tools: dict[str, Any] = field(default_factory=dict)


def make_default_context(settings: Settings | None = None, llm: BaseLLM | None = None,
                         retriever: Any = None) -> AgentContext:
    """构建默认上下文：Mock 工具集，LLM/检索器按配置自动选择。"""
    from backend.app.tools.coupon import CouponTool
    from backend.app.tools.inventory import InventoryTool
    from backend.app.tools.logistics_api import LogisticsTool
    from backend.app.tools.order_query import OrderQueryTool
    from backend.app.tools.ticket import TicketTool

    s = settings or get_settings()
    return AgentContext(
        llm=llm or get_llm(s),
        settings=s,
        retriever=retriever,
        tools={
            "order_query": OrderQueryTool(s),
            "inventory": InventoryTool(s),
            "logistics_api": LogisticsTool(s),
            "coupon": CouponTool(s),
            "ticket": TicketTool(s),
        },
    )
