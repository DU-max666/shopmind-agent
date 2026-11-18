"""活动/价保子图：活动规则检索 → 差价计算工具 → 价保申请。

差价计算是纯函数（确定性、可单测）：
- 价保窗口：下单后 30 天内（POLICY-PRICE 条款一）；
- 对比口径：下单单价 vs 当前到手价（活动价/券后价取低者）；
- 差价 > 0 且在窗口内 → 可补；创建 price_guard 工单走人工复核打款。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from backend.app.graphs.state import AgentContext, ShopMindState
from backend.app.graphs.supervisor import ORDER_ID_RE
from backend.app.llm import ChatMessage
from backend.prompts.loader import load_prompt

PRICE_GUARD_DAYS = 30

ASK_ORDER_ID = "申请价保需要核对下单记录，请提供订单号（ORD 开头）。"


def calc_price_gap(paid_unit: float, current_price: float, *,
                   order_date: str, today: date | None = None,
                   protection_days: int = PRICE_GUARD_DAYS) -> dict[str, Any]:
    """差价计算工具：返回 eligible/gap/剩余天数，全部确定性。"""
    t = today or date.today()
    try:
        placed = date.fromisoformat(order_date)
    except ValueError:
        return {"eligible": False, "gap": 0.0, "remain_days": 0, "reason": "下单日期无法解析"}
    elapsed = (t - placed).days
    remain = protection_days - elapsed
    gap = round(max(0.0, paid_unit - current_price), 2)
    if remain < 0:
        return {"eligible": False, "gap": gap, "remain_days": 0,
                "reason": f"超出 {protection_days} 天价保期"}
    if gap <= 0:
        return {"eligible": False, "gap": gap, "remain_days": remain, "reason": "当前价不低于下单价"}
    return {"eligible": True, "gap": gap, "remain_days": remain,
            "reason": f"价保期内降价，可补差价 {gap} 元"}


def handle(state: ShopMindState, ctx: AgentContext) -> dict[str, Any]:
    text = state.get("user_text", "")
    hits = ctx.retriever.search("价保 政策 差价", top_k=2) if ctx.retriever is not None else []
    kb_evidence = [{"doc_id": h.doc_id, "text": h.meta.get("snippet", h.text)} for h in hits]

    m = ORDER_ID_RE.search(text)
    if not m:
        return {"draft_answer": ASK_ORDER_ID, "evidence": kb_evidence}
    order_id = m.group(1).upper()

    order_res = ctx.tools["order_query"].run(order_id=order_id)
    if not order_res.ok:
        return {"tool_results": [order_res.to_dict()],
                "draft_answer": f"未查询到订单 {order_id}，请核对订单号。", "evidence": kb_evidence}
    order = order_res.data["order"]
    item = order["items"][0]
    sku = item["sku"]

    inv_res = ctx.tools["inventory"].run(skus=[sku])
    current_price = float(inv_res.data["items"][0]["price"]) if inv_res.ok else float(item["paid"])

    coupon_note = ""
    if order.get("coupon_code") and "coupon" in ctx.tools:
        c_res = ctx.tools["coupon"].run(coupon_code=order["coupon_code"])
        if c_res.ok:
            c = c_res.data["coupon"]
            coupon_note = f"（下单使用券 {c['code']}，面额 {c['amount']} 元，状态 {c['status']}）"

    calc = calc_price_gap(float(item["paid"]), current_price, order_date=order["order_date"])
    calc_line = (f"订单 {order_id} 下单单价 {item['paid']} 元，当前到手价 {current_price} 元，"
                 f"差价 {calc['gap']} 元，价保期 {PRICE_GUARD_DAYS} 天，剩余 {calc['remain_days']} 天"
                 f"{coupon_note}")
    evidence = ([{"doc_id": "PROMO-CALC", "text": calc_line}] + kb_evidence +
                [{"doc_id": f"ORDER-{order_id}",
                  "text": f"订单 {order_id} 商品 {item['name']}，下单日期 {order['order_date']}"}])
    tool_results = [order_res.to_dict(), inv_res.to_dict()]

    if calc["eligible"]:
        ticket_res = ctx.tools["ticket"].run(
            action="create", order_id=order_id, user_id=order["user_id"],
            kind="price_guard", summary=f"价保申请：补差价 {calc['gap']} 元")
        tool_results.append(ticket_res.to_dict())
        ticket_id = ticket_res.data["ticket_id"] if ticket_res.ok else ""
        if ticket_id:
            evidence.append({"doc_id": ticket_id,
                             "text": f"价保工单 {ticket_id} 已创建，订单 {order_id}，金额 {calc['gap']} 元"})
        answer = (f"经核算，{calc_line}，符合价保条件【PROMO-CALC】。"
                  f"已创建价保工单 {ticket_id}，人工复核后按原支付渠道返还差价【{ticket_id}】。")
        needs_human, human_reason = True, "价保打款需人工复核"
    else:
        prompt = (f"{load_prompt('promo')}\n用户诉求：{text}\n"
                  f"EVIDENCE_BEGIN\n" + "\n".join(f"[{e['doc_id']}] {e['text']}" for e in evidence)
                  + "\nEVIDENCE_END\n请说明价保结论与依据。")
        answer = ctx.llm.complete([ChatMessage("user", prompt)])
        answer = f"经核算，{calc_line}，{calc['reason']}【PROMO-CALC】。"
        needs_human, human_reason = False, ""

    return {"slots": {"order_id": order_id, "price_gap": calc},
            "evidence": evidence, "tool_results": tool_results,
            "needs_human": needs_human, "human_reason": human_reason,
            "ticket_status": "awaiting_confirm" if needs_human else "",
            "draft_answer": answer}
