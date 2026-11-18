"""Supervisor 路由与全链路单测（纯 stdlib + MockLLM，离线可跑）。"""
from __future__ import annotations

import pytest

from backend.app.graphs.state import make_default_context
from backend.app.graphs.supervisor import (
    CLARIFY_REPLY,
    rule_route,
    route_intent,
    run_turn,
)
from backend.app.llm import MockLLM
from backend.app.tools import ticket as ticket_mod


@pytest.fixture(autouse=True)
def _reset_ticket_store():
    """每个用例重置模块级工单账本，保证 TK/RH 自增 id 可断言。"""
    ticket_mod.reset_default_store()
    yield


def test_rule_route_after_sale_high_confidence():
    r = rule_route("收到的沙发有破损要退货")
    assert r is not None and r.intent == "after_sale" and r.confidence >= 0.75


def test_rule_route_promo():
    r = rule_route("双十一降价了能申请价保吗")
    assert r is not None and r.intent == "promo"


def test_rule_route_logistics_beats_guide_on_priority():
    # 同分时按规则表优先级：物流(0.90) 高于 导购(0.70)
    r = rule_route("物流到哪了，顺便推荐个台灯")
    assert r is not None and r.intent == "logistics"


def test_llm_fallback_routes_guide():
    # 导购规则置信 0.70 < 0.75，应由 MockLLM 兜底升到 guide
    r = route_intent("乳胶枕多少钱一个", MockLLM())
    assert r.intent == "guide" and r.source == "llm"


def test_clarify_on_greeting():
    r = route_intent("在吗", MockLLM())
    assert r.intent == "clarify"


def test_run_turn_logistics_happy_path():
    ctx = make_default_context(llm=MockLLM())
    st = run_turn("查一下 ORD20261105001 到哪了", ctx=ctx)
    assert st["intent"] == "logistics"
    assert "智铺速运" in st["final_answer"]
    assert st["needs_human"] is False
    assert st["grounded_score"] == 1.0


def test_run_turn_logistics_abnormal_escalates():
    ctx = make_default_context(llm=MockLLM())
    st = run_turn("ORD20261114005 物流一直没更新", ctx=ctx)
    assert st["needs_human"] is True
    assert st["ticket_status"] == "escalated"
    assert st["ticket_id"].startswith("TK")
    assert "48" in st["final_answer"]


def test_run_turn_after_sale_missing_order_asks_slot():
    ctx = make_default_context(llm=MockLLM())
    st = run_turn("我要退货", ctx=ctx)
    assert st["intent"] == "after_sale"
    assert "订单号" in st["final_answer"]
    assert st.get("needs_human", False) is False


def test_run_turn_after_sale_high_amount_manual():
    ctx = make_default_context(llm=MockLLM())
    st = run_turn("ORD20261105001 沙发有破损要退货退款", ctx=ctx)
    assert st["intent"] == "after_sale"
    assert st["needs_human"] is True  # 1299 元 > 人工审批线
    assert st["ticket_status"] == "escalated"
    assert "1299.0" in st["final_answer"]


def test_run_turn_promo_price_guard_eligible():
    ctx = make_default_context(llm=MockLLM())
    st = run_turn("台灯降价了，申请价保，订单 ORD20261020003", ctx=ctx)
    assert st["intent"] == "promo"
    assert st["slots"]["price_gap"]["eligible"] is True
    assert st["slots"]["price_gap"]["gap"] == 30.0
    assert "30.0" in st["final_answer"]


def test_run_turn_clarify_reply():
    ctx = make_default_context(llm=MockLLM())
    st = run_turn("你好", ctx=ctx)
    assert st["final_answer"] == CLARIFY_REPLY
    assert st["messages"][-1]["role"] == "assistant"
