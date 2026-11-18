"""物流子图：订单号槽位填充 → 物流 API → 异常件升级。

- 缺订单号：追问补齐（槽位填充），不猜测；
- 查到轨迹：话术中的运单号/时间/状态全部来自工具结果并进入证据集；
- 异常件（滞留/丢件/破损退回）：自动创建工单升级人工，标记 needs_human。
"""
from __future__ import annotations

from typing import Any

from backend.app.graphs.state import AgentContext, ShopMindState
from backend.app.graphs.supervisor import ORDER_ID_RE

ASK_ORDER_ID = "请提供您的订单号（ORD 开头），我马上为您查询物流轨迹。"
NOT_FOUND_REPLY_TMPL = "未查询到订单 {order_id} 的物流信息，请核对订单号，或转人工协助核实。"


def handle(state: ShopMindState, ctx: AgentContext) -> dict[str, Any]:
    text = state.get("user_text", "")
    m = ORDER_ID_RE.search(text)
    if not m:
        return {"slots": {"order_id": None}, "draft_answer": ASK_ORDER_ID, "evidence": []}
    order_id = m.group(1).upper()

    track_res = ctx.tools["logistics_api"].run(order_id=order_id)
    if not track_res.ok:
        return {"slots": {"order_id": order_id},
                "tool_results": [track_res.to_dict()],
                "draft_answer": NOT_FOUND_REPLY_TMPL.format(order_id=order_id),
                "evidence": []}

    data = track_res.data
    events = data.get("events", [])
    last_event = events[-1] if events else {"time": "-", "desc": "暂无轨迹"}
    evidence = [{"doc_id": f"LOG-{order_id}",
                 "text": (f"订单 {order_id}，运单 {data['waybill']}，承运商 {data['carrier']}，"
                          f"当前状态 {data['status']}。最新轨迹：{last_event['time']} "
                          f"{last_event['desc']}")} ]
    answer = (f"您的订单 {order_id} 由 {data['carrier']} 承运（运单号 {data['waybill']}），"
              f"当前状态：{data['status']}，最新轨迹：{last_event['desc']}"
              f"（{last_event['time']}）【LOG-{order_id}】。")
    needs_human = False
    human_reason = ""
    ticket_id = ""
    tool_results = [track_res.to_dict()]

    if data.get("abnormal"):
        ticket_res = ctx.tools["ticket"].run(
            action="create", order_id=order_id, user_id=state.get("user_id", "U-GUEST"),
            kind="other", summary=f"物流异常件（{data['status']}）跟进")
        tool_results.append(ticket_res.to_dict())
        ticket_id = ticket_res.data["ticket_id"] if ticket_res.ok else ""
        if ticket_id:
            evidence.append({"doc_id": ticket_id,
                             "text": f"异常工单 {ticket_id} 已创建，订单 {order_id}"})
            evidence.append({"doc_id": "POLICY-SLA",
                             "text": "异常件 48 小时内联系承运商核查并给出处理方案"})
            answer += (f"该件状态异常，已自动创建工单 {ticket_id} 升级人工，"
                       f"我们将在 48 小时内联系承运商核查【{ticket_id}】【POLICY-SLA】。")
        needs_human = True
        human_reason = f"物流异常件：{data['status']}"

    return {"slots": {"order_id": order_id}, "evidence": evidence,
            "tool_results": tool_results, "needs_human": needs_human,
            "human_reason": human_reason, "ticket_id": ticket_id,
            "ticket_status": "escalated" if needs_human else "",
            "draft_answer": answer}
