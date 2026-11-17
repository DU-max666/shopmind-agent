"""Supervisor：意图路由 + 子图分发 + 统一安全收尾。

路由策略（三层）：
1. 规则前置：高置信关键词/正则直接命中（可解释、零成本、毫秒级）；
2. LLM 兜底：规则未命中或置信不足时，调用 LLM 分类器；
3. 低置信澄清：两层都不到澄清线时，不硬答，向用户追问。

收尾链路对应"意图识别—工具调用—结果校验—生成—人工复核"中的校验与复核：
子图产出 draft 后统一过 规则+LLM 协同校验 与 证据一致性，不过则转人工。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from backend.app.config import Settings, get_settings
from backend.app.graphs.state import AgentContext, Intent, ShopMindState, make_default_context
from backend.app.llm import BaseLLM, ChatMessage
from backend.app.safety import confidence, grounded_check, rule_llm_validator

ORDER_ID_RE = re.compile(r"(ORD\d{8,})", re.IGNORECASE)

CLARIFY_REPLY = (
    "为了更快帮您处理，请补充一下：您是想 1) 选购推荐 2) 退换货/质量问题 "
    "3) 查物流 4) 咨询活动/价保？如有订单号（ORD 开头）请一并发我。"
)
FALLBACK_HUMAN_REPLY = "该回答未通过可靠性校验，已为您转接人工客服复核，请稍候。"


@dataclass
class RouteResult:
    """一次路由的完整决策记录（进入审计日志）。"""

    intent: str
    confidence: float
    source: str  # rule / llm / clarify
    matched: str = ""


# 规则表：按业务优先级排序（售后 > 活动 > 物流 > 导购），置信同分时取先者
_RULE_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    (Intent.AFTER_SALE.value,
     re.compile(r"退货|换货|退款|质量|破损|坏了|投诉|工单|维修|保修|三包"), 0.90),
    (Intent.PROMO.value,
     re.compile(r"价保|保价|差价|降价|优惠券|满减|活动|大促|双十一|双11|618|折扣"), 0.85),
    (Intent.LOGISTICS.value,
     re.compile(r"物流|快递|运单|派送|签收|发货|到哪|什么时候到|配送"), 0.90),
    (Intent.GUIDE.value,
     re.compile(r"推荐|想买|有没有|多少钱|价格|对比|怎么选|尺寸|材质|款式|搭配|预算"), 0.70),
]


def rule_route(text: str) -> RouteResult | None:
    """规则层：返回最高置信命中（同分按规则表优先级），未命中返回 None。"""
    best: RouteResult | None = None
    for intent, pattern, conf in _RULE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        candidate = RouteResult(intent, conf, "rule", matched=m.group(0))
        if best is None or candidate.confidence > best.confidence:
            best = candidate
    return best


def llm_route(text: str, llm: BaseLLM) -> RouteResult:
    """LLM 兜底分类层：解析失败按最低置信处理。"""
    prompt = (
        "你是电商客服意图分类器（意图分类）。仅输出 JSON："
        '{"intent": "guide|after_sale|logistics|promo|clarify", "confidence": 0.0~1.0}\n'
        f"用户问题：{text}"
    )
    raw = llm.complete([ChatMessage("user", prompt)], temperature=0.0)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return RouteResult(Intent.CLARIFY.value, 0.0, "llm")
    try:
        data = json.loads(m.group(0))
        intent = str(data.get("intent", "clarify"))
        valid = {i.value for i in Intent}
        if intent not in valid:
            return RouteResult(Intent.CLARIFY.value, 0.0, "llm")
        return RouteResult(intent, float(data.get("confidence", 0.0)), "llm")
    except (json.JSONDecodeError, TypeError, ValueError):
        return RouteResult(Intent.CLARIFY.value, 0.0, "llm")


def route_intent(text: str, llm: BaseLLM | None = None,
                 settings: Settings | None = None) -> RouteResult:
    """三层路由主入口：规则前置 → LLM 兜底 → 低置信澄清。"""
    s = settings or get_settings()
    rule = rule_route(text)
    if rule and rule.confidence >= s.route_answer_threshold:
        return rule
    llm_res = llm_route(text, llm) if llm is not None else None
    candidates = [c for c in (rule, llm_res) if c is not None]
    if not candidates:
        return RouteResult(Intent.CLARIFY.value, 0.3, "clarify")
    best = max(candidates, key=lambda c: c.confidence)
    if best.confidence < s.route_clarify_threshold:
        return RouteResult(Intent.CLARIFY.value, best.confidence, "clarify")
    return best


def _subgraph_handlers() -> dict[str, Any]:
    """延迟导入子图，避免模块级循环依赖。"""
    from backend.app.graphs import after_sale, guide, logistics, promo

    return {
        Intent.GUIDE.value: guide.handle,
        Intent.AFTER_SALE.value: after_sale.handle,
        Intent.LOGISTICS.value: logistics.handle,
        Intent.PROMO.value: promo.handle,
    }


def iter_turn_events(user_text: str, state: ShopMindState | None = None,
                     ctx: AgentContext | None = None) -> Iterator[dict[str, Any]]:
    """以事件流方式执行一轮对话（SSE 端点与评测共用同一执行链）。"""
    ctx = ctx or make_default_context()
    st: ShopMindState = state if state is not None else {}  # type: ignore[assignment]
    st["user_text"] = user_text
    st.setdefault("messages", []).append({"role": "user", "content": user_text})

    route = route_intent(user_text, ctx.llm, ctx.settings)
    st.update(intent=route.intent, intent_confidence=route.confidence, route_source=route.source)
    yield {"event": "route", "data": asdict(route)}

    if route.intent == Intent.CLARIFY.value:
        st["final_answer"] = CLARIFY_REPLY
        yield {"event": "answer", "data": {"text": st["final_answer"]}}
    else:
        handler = _subgraph_handlers()[route.intent]
        updates = handler(st, ctx)
        st.update(updates)  # type: ignore[arg-type]
        yield {"event": "subgraph", "data": {
            "intent": route.intent,
            "slots": st.get("slots", {}),
            "tool_results": st.get("tool_results", []),
            "ticket_id": st.get("ticket_id", ""),
            "ticket_status": st.get("ticket_status", ""),
        }}
        answer = st.get("draft_answer", "")
        evidence = st.get("evidence", [])
        ground = grounded_check.check(answer, evidence) if answer else None
        if answer and ground is not None:
            st["grounded_score"] = ground.score
            verdict = rule_llm_validator.validate(answer, evidence, ctx.llm)
            if not verdict.passed:
                st["needs_human"] = True
                st["human_reason"] = "; ".join(verdict.issues)
                st["final_answer"] = FALLBACK_HUMAN_REPLY
            else:
                st["final_answer"] = answer
        else:
            st["final_answer"] = answer or FALLBACK_HUMAN_REPLY
        yield {"event": "safety", "data": {
            "grounded_score": st.get("grounded_score", 0.0),
            "needs_human": st.get("needs_human", False),
            "human_reason": st.get("human_reason", ""),
        }}
        yield {"event": "answer", "data": {
            "text": st.get("final_answer", ""),
            "ticket_id": st.get("ticket_id", ""),
            "ticket_status": st.get("ticket_status", ""),
            "needs_human": st.get("needs_human", False),
        }}
    st["messages"].append({"role": "assistant", "content": st.get("final_answer", "")})
    yield {"event": "done", "data": {"session_id": st.get("session_id", "")}}


def run_turn(user_text: str, state: ShopMindState | None = None,
             ctx: AgentContext | None = None) -> ShopMindState:
    """同步执行一轮完整链路，返回最终状态。"""
    st: ShopMindState = dict(state or {})  # type: ignore[assignment]
    for _ in iter_turn_events(user_text, st, ctx):
        pass
    return st


def build_graph():
    """构建 LangGraph Supervisor 图（生产编排入口，需安装 langgraph）。

    拓扑：classify → (条件边) guide/after_sale/logistics/promo/clarify → finalize → END
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要 langgraph：pip install langgraph") from exc

    ctx = make_default_context()

    def classify_node(state: ShopMindState) -> dict[str, Any]:
        route = route_intent(state.get("user_text", ""), ctx.llm, ctx.settings)
        return {"intent": route.intent, "intent_confidence": route.confidence,
                "route_source": route.source}

    def make_subgraph_node(name: str):
        handler = _subgraph_handlers()[name]

        def node(state: ShopMindState) -> dict[str, Any]:
            return dict(handler(state, ctx))

        node.__name__ = f"{name}_node"
        return node

    def clarify_node(state: ShopMindState) -> dict[str, Any]:
        return {"final_answer": CLARIFY_REPLY}

    def finalize_node(state: ShopMindState) -> dict[str, Any]:
        answer = state.get("draft_answer", "")
        evidence = state.get("evidence", [])
        if not answer:
            return {"final_answer": FALLBACK_HUMAN_REPLY, "needs_human": True}
        verdict = rule_llm_validator.validate(answer, evidence, ctx.llm)
        if not verdict.passed:
            return {"needs_human": True, "human_reason": "; ".join(verdict.issues),
                    "final_answer": FALLBACK_HUMAN_REPLY}
        return {"final_answer": answer}

    def dispatch(state: ShopMindState) -> str:
        intent = state.get("intent", Intent.CLARIFY.value)
        if intent == Intent.CLARIFY.value:
            return "clarify"
        return intent

    graph = StateGraph(ShopMindState)
    graph.add_node("classify", classify_node)
    for name in ("guide", "after_sale", "logistics", "promo"):
        graph.add_node(name, make_subgraph_node(name))
    graph.add_node("clarify", clarify_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", dispatch,
                                {name: name for name in ("guide", "after_sale", "logistics",
                                                          "promo", "clarify")})
    for name in ("guide", "after_sale", "logistics", "promo"):
        graph.add_edge(name, "finalize")
    graph.add_edge("clarify", END)
    graph.add_edge("finalize", END)
    return graph.compile()
