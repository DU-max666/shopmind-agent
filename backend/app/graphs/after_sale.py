"""售后子图：显式工单状态机。

识别 → 政策检索 → 处理建议 → 创建工单 → 高风险动作人工确认。
状态迁移必须走 TRANSITIONS 白名单，任何非法迁移抛异常并记审计，
保证"Agent 不能凭空跳到已解决"这类越权路径在结构上不存在。
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from backend.app.graphs.state import AgentContext, ShopMindState, TicketStatus
from backend.app.graphs.supervisor import ORDER_ID_RE
from backend.app.llm import ChatMessage
from backend.app.safety import refund_guard
from backend.prompts.loader import load_prompt

# 状态机合法迁移表
TRANSITIONS: dict[TicketStatus, set[TicketStatus]] = {
    TicketStatus.INTAKE: {TicketStatus.POLICY_CHECK, TicketStatus.REJECTED},
    TicketStatus.POLICY_CHECK: {TicketStatus.SUGGESTION, TicketStatus.REJECTED},
    TicketStatus.SUGGESTION: {TicketStatus.CREATED, TicketStatus.REJECTED},
    TicketStatus.CREATED: {TicketStatus.AWAITING_CONFIRM, TicketStatus.ESCALATED,
                           TicketStatus.RESOLVED},
    TicketStatus.AWAITING_CONFIRM: {TicketStatus.RESOLVED, TicketStatus.ESCALATED},
    TicketStatus.ESCALATED: {TicketStatus.RESOLVED},
    TicketStatus.RESOLVED: set(),
    TicketStatus.REJECTED: set(),
}

_KIND_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("exchange", re.compile(r"换货|换新|换一件")),
    ("quality", re.compile(r"质量|破损|坏了|维修|保修")),
    ("return", re.compile(r"退货|退款|不想要了")),
]

# 售后时限（天）：与知识库 POLICY-RMA 条款保持一致
RETURN_WINDOW_DAYS = 7
EXCHANGE_WINDOW_DAYS = 15
QUALITY_WINDOW_DAYS = 30


class IllegalTransition(RuntimeError):
    """非法状态迁移。"""


def next_status(current: TicketStatus, target: TicketStatus) -> TicketStatus:
    """受控状态迁移：仅允许 TRANSITIONS 白名单内的边。"""
    if target not in TRANSITIONS.get(current, set()):
        raise IllegalTransition(f"非法迁移: {current.value} -> {target.value}")
    return target


def detect_kind(text: str) -> str | None:
    for kind, pattern in _KIND_PATTERNS:
        if pattern.search(text):
            return kind
    return None


def _within_window(sign_date_str: str | None, kind: str, today: date) -> bool:
    if not sign_date_str:
        return True  # 未签收（待发货/运输中）允许取消/拦截类售后
    try:
        signed = date.fromisoformat(sign_date_str)
    except ValueError:
        return True
    window = {"return": RETURN_WINDOW_DAYS, "exchange": EXCHANGE_WINDOW_DAYS,
              "quality": QUALITY_WINDOW_DAYS}.get(kind, RETURN_WINDOW_DAYS)
    return today - signed <= timedelta(days=window)


def handle(state: ShopMindState, ctx: AgentContext) -> dict[str, Any]:
    text = state.get("user_text", "")
    today = date.today()
    kind = detect_kind(text)
    updates: dict[str, Any] = {"slots": {"after_sale_kind": kind}}

    # INTAKE → POLICY_CHECK：检索售后政策
    status = next_status(TicketStatus.INTAKE, TicketStatus.POLICY_CHECK)
    hits = ctx.retriever.search(text, top_k=3) if ctx.retriever is not None else []
    evidence: list[dict[str, Any]] = [
        {"doc_id": h.doc_id, "text": h.meta.get("snippet", h.text)} for h in hits
    ]
    tool_results: list[dict[str, Any]] = []

    order_id = None
    m = ORDER_ID_RE.search(text)
    if m:
        order_id = m.group(1).upper()

    # POLICY_CHECK → SUGGESTION：给出处理建议
    status = next_status(status, TicketStatus.SUGGESTION)
    if not order_id:
        ask = "请提供订单号（ORD 开头），我先核对您的售后资格再给出处理方案。"
        updates.update({"ticket_status": status.value, "evidence": evidence,
                        "draft_answer": ask})
        return updates

    order_res = ctx.tools["order_query"].run(order_id=order_id)
    tool_results.append(order_res.to_dict())
    if not order_res.ok:
        miss = f"未查询到订单 {order_id}，请核对订单号。"
        updates.update({"ticket_status": TicketStatus.REJECTED.value, "evidence": evidence,
                        "tool_results": tool_results, "draft_answer": miss})
        return updates

    order = order_res.data["order"]
    paid = float(order["paid_amount"])
    evidence.append({"doc_id": f"ORDER-{order_id}",
                     "text": f"订单 {order_id} 实付 {paid} 元，状态 {order['status']}，"
                             f"签收日期 {order.get('sign_date') or '未签收'}"})
    in_window = _within_window(order.get("sign_date"), kind or "return", today)
    decision = refund_guard.decide_refund(
        paid, order_found=True, policy_allowed=kind is not None,
        within_window=in_window, settings=ctx.settings)

    # SUGGESTION → CREATED：创建工单
    status = next_status(status, TicketStatus.CREATED)
    ticket_res = ctx.tools["ticket"].run(
        action="create", order_id=order_id, user_id=order["user_id"],
        kind=kind or "other",
        summary=f"{kind or 'other'} 售后诉求：{text[:50]}；退款判定 {decision.tier} 档")
    tool_results.append(ticket_res.to_dict())
    ticket_id = ticket_res.data["ticket_id"] if ticket_res.ok else ""
    if ticket_id:
        evidence.append({"doc_id": ticket_id,
                         "text": f"工单 {ticket_id} 已创建，类型 {kind or 'other'}，"
                                 f"关联订单 {order_id}，金额 {paid} 元"})
    evidence.append({"doc_id": "POLICY-SLA",
                     "text": "工单创建后 48 小时内响应，退款审核 72 小时内完成"})

    # 高风险动作：按退款分档执行（自动 / 沙箱预演待确认 / 人工）
    needs_human = False
    human_reason = ""
    if decision.tier == refund_guard.TIER_AUTO and in_window and kind:
        commit = refund_guard.commit_refund(ctx.tools["ticket"], decision, order_id=order_id)
        tool_results.append(commit)
        if commit.get("status") == "committed":
            status = next_status(status, TicketStatus.RESOLVED)
            answer = (f"您的订单 {order_id} 符合{kind}政策，退款 {paid} 元已自动受理，"
                      f"工单 {ticket_id} 将在 72 小时内完成审核【{ticket_id}】"
                      f"【POLICY-SLA】。")
        else:
            needs_human, human_reason = True, f"自动退款执行失败：{commit.get('error')}"
            answer = f"已为您创建工单 {ticket_id}，退款处理遇到异常，已转人工跟进【{ticket_id}】。"
    elif decision.tier == refund_guard.TIER_SANDBOX:
        rehearse = refund_guard.rehearse_refund(ctx.tools["ticket"], order_id, paid)
        tool_results.append(rehearse.to_dict())
        status = next_status(status, TicketStatus.AWAITING_CONFIRM)
        needs_human, human_reason = True, "退款金额处于沙箱预演档，等待人工确认令牌"
        rehearsal_id = rehearse.data.get("rehearsal_id", "") if rehearse.ok else ""
        evidence.append({"doc_id": f"RH-{rehearsal_id}",
                         "text": f"沙箱预演快照 {rehearsal_id}，订单 {order_id}，金额 {paid} 元，未提交"})
        answer = (f"订单 {order_id} 退款金额 {paid} 元超过自动档上限，"
                  f"已完成沙箱预演（快照 {rehearsal_id}）【RH-{rehearsal_id}】，"
                  f"待人工确认后原路退回【{ticket_id}】。")
    elif decision.tier == refund_guard.TIER_MANUAL:
        status = next_status(status, TicketStatus.ESCALATED)
        needs_human, human_reason = True, "退款金额超过人工审批线"
        answer = f"订单 {order_id} 退款金额 {paid} 元需人工审批，工单 {ticket_id} 已升级【{ticket_id}】。"
    else:
        status = TicketStatus.ESCALATED
        needs_human, human_reason = True, decision.reason
        answer = (f"订单 {order_id} 的诉求（{decision.reason}）已登记工单 {ticket_id}，"
                  f"由人工客服复核特批【{ticket_id}】。")

    # 政策依据句：优先用 LLM 基于证据生成，离线 Mock 时回退到政策原文引用
    policy_line = evidence[0]["text"] if evidence else ""
    if policy_line:
        answer += f"\n政策依据：{policy_line.splitlines()[0]}【{evidence[0]['doc_id']}】"
    else:
        prompt = (f"{load_prompt('after_sale')}\n用户诉求：{text}\n"
                  "EVIDENCE_BEGIN\n暂无证据\nEVIDENCE_END")
        answer += "\n" + ctx.llm.complete([ChatMessage("user", prompt)])

    updates.update({
        "ticket_id": ticket_id,
        "ticket_status": status.value,
        "evidence": evidence,
        "tool_results": tool_results,
        "needs_human": needs_human,
        "human_reason": human_reason,
        "draft_answer": answer,
    })
    return updates
