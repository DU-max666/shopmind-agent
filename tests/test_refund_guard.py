"""退款防线单测：三级分档 + 沙箱预演 + 人工令牌 + 回滚。"""
from __future__ import annotations

from backend.app.config import Settings
from backend.app.safety import refund_guard
from backend.app.tools.ticket import InMemoryTicketStore, TicketTool

S = Settings(auto_refund_max=50.0, sandbox_refund_max=200.0)


def make_tool() -> TicketTool:
    return TicketTool(store=InMemoryTicketStore())


def test_tier_auto():
    d = refund_guard.decide_refund(30.0, settings=S)
    assert d.tier == refund_guard.TIER_AUTO and not d.requires_confirm


def test_tier_sandbox_requires_confirm():
    d = refund_guard.decide_refund(150.0, settings=S)
    assert d.tier == refund_guard.TIER_SANDBOX and d.requires_confirm


def test_tier_manual():
    d = refund_guard.decide_refund(500.0, settings=S)
    assert d.tier == refund_guard.TIER_MANUAL


def test_tier_reject_preconditions():
    assert refund_guard.decide_refund(10.0, order_found=False, settings=S).tier == refund_guard.TIER_REJECT
    assert refund_guard.decide_refund(10.0, policy_allowed=False, settings=S).tier == refund_guard.TIER_REJECT
    assert refund_guard.decide_refund(10.0, within_window=False, settings=S).tier == refund_guard.TIER_REJECT


def test_auto_commit_flow():
    tool = make_tool()
    d = refund_guard.decide_refund(30.0, settings=S)
    res = refund_guard.commit_refund(tool, d, order_id="ORD20261112004")
    assert res["status"] == "committed"


def test_sandbox_pending_without_token():
    tool = make_tool()
    d = refund_guard.decide_refund(150.0, settings=S)
    pending = refund_guard.commit_refund(tool, d, order_id="ORD20261112004")
    assert pending["status"] == "pending_confirm"


def test_sandbox_commit_requires_hitl_token():
    tool = make_tool()
    rh = tool.run(action="rehearse_refund", order_id="ORD20261112004", amount=150.0)
    assert rh.ok and rh.data["status"] == "rehearsed"
    rehearsal_id = rh.data["rehearsal_id"]
    d = refund_guard.decide_refund(150.0, settings=S)
    bad = refund_guard.commit_refund(tool, d, order_id="X", rehearsal_id=rehearsal_id,
                                     confirm_token="FAKE-1")
    assert bad["status"] == "failed"  # 非 HITL- 前缀令牌被拒绝
    good = refund_guard.commit_refund(tool, d, order_id="X", rehearsal_id=rehearsal_id,
                                      confirm_token="HITL-OPS-001")
    assert good["status"] == "committed"


def test_manual_tier_never_auto_commits():
    tool = make_tool()
    d = refund_guard.decide_refund(9999.0, settings=S)
    res = refund_guard.commit_refund(tool, d, order_id="X", rehearsal_id="RH00001",
                                    confirm_token="HITL-OPS-002")
    assert res["status"] == "manual_only"  # Agent 侧永远不直接提交大额退款


def test_rollback_after_commit():
    tool = make_tool()
    rh = tool.run(action="rehearse_refund", order_id="X", amount=80.0)
    rehearsal_id = rh.data["rehearsal_id"]
    tool.run(action="commit_refund", rehearsal_id=rehearsal_id, confirm_token="HITL-OPS-003")
    back = refund_guard.rollback_refund(tool, rehearsal_id)
    assert back["status"] == "rolled_back"
    again = tool.run(action="commit_refund", rehearsal_id=rehearsal_id, confirm_token="HITL-OPS-004")
    assert again.ok  # 回滚后允许重新提交
