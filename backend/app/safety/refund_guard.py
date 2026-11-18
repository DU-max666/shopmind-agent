"""高风险动作防线：退款金额三级分档 + 沙箱预演 + 人工确认 + 可回滚。

分档（阈值可配置，默认 50 / 200 元）：
- auto     ≤ 50 元：策略允许即可自动退款；
- sandbox  50~200 元：先在工单系统做沙箱预演（只写快照不动资金），
           提交正式退款必须携带人工确认令牌 HITL-xxxx；
- manual   > 200 元：一律转人工审批，Agent 只能创建待审工单。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.config import Settings, get_settings
from backend.app.tools.base import ToolResult

TIER_AUTO = "auto"
TIER_SANDBOX = "sandbox"
TIER_MANUAL = "manual"
TIER_REJECT = "reject"


@dataclass
class RefundDecision:
    """退款分级决策结果。"""

    tier: str
    amount: float
    reason: str
    requires_confirm: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def decide_refund(amount: float, *, order_found: bool = True, policy_allowed: bool = True,
                  within_window: bool = True, settings: Settings | None = None) -> RefundDecision:
    """先做业务前置校验，再按金额分档。"""
    s = settings or get_settings()
    if not order_found:
        return RefundDecision(TIER_REJECT, amount, "订单不存在，拒绝退款")
    if not policy_allowed:
        return RefundDecision(TIER_REJECT, amount, "不符合退换货政策条件")
    if not within_window:
        return RefundDecision(TIER_REJECT, amount, "超出售后时限，需人工特批")
    if amount <= 0:
        return RefundDecision(TIER_REJECT, amount, "退款金额非法")
    if amount <= s.auto_refund_max:
        return RefundDecision(TIER_AUTO, amount, f"≤{s.auto_refund_max:.0f} 元自动档")
    if amount <= s.sandbox_refund_max:
        return RefundDecision(TIER_SANDBOX, amount,
                              f"{s.auto_refund_max:.0f}~{s.sandbox_refund_max:.0f} 元沙箱预演档",
                              requires_confirm=True)
    return RefundDecision(TIER_MANUAL, amount, f">{s.sandbox_refund_max:.0f} 元人工审批档",
                          requires_confirm=True)


def rehearse_refund(ticket_tool: Any, order_id: str, amount: float) -> ToolResult:
    """沙箱预演：写入退款快照，不产生资金动作。"""
    return ticket_tool.run(action="rehearse_refund", order_id=order_id, amount=amount)


def commit_refund(ticket_tool: Any, decision: RefundDecision, *, order_id: str = "",
                  rehearsal_id: str = "", confirm_token: str = "") -> dict[str, Any]:
    """按分档执行正式退款；sandbox/manual 档缺令牌时挂起等待人工。"""
    if decision.tier == TIER_AUTO:
        result = ticket_tool.run(action="rehearse_refund", order_id=order_id,
                                 amount=decision.amount)
        if not result.ok:
            return {"status": "failed", "error": result.error}
        auto_id = str(result.data["rehearsal_id"])
        committed = ticket_tool.run(action="commit_refund", rehearsal_id=auto_id,
                                    confirm_token="HITL-AUTO")
        return {"status": "committed" if committed.ok else "failed",
                "rehearsal_id": auto_id, "error": committed.error}
    if decision.tier == TIER_SANDBOX:
        if not confirm_token:
            return {"status": "pending_confirm", "rehearsal_id": rehearsal_id,
                    "reason": decision.reason}
        result = ticket_tool.run(action="commit_refund", rehearsal_id=rehearsal_id,
                                 confirm_token=confirm_token)
        return {"status": "committed" if result.ok else "failed",
                "rehearsal_id": rehearsal_id, "error": result.error}
    # manual / reject：只允许人工通道，Agent 侧永远不直接提交
    return {"status": "manual_only", "reason": decision.reason}


def rollback_refund(ticket_tool: Any, rehearsal_id: str) -> dict[str, Any]:
    """依据预演快照回滚（committed 后人工发现问题时的撤销入口）。"""
    result = ticket_tool.run(action="rollback_refund", rehearsal_id=rehearsal_id)
    return {"status": "rolled_back" if result.ok else "failed", "error": result.error}
