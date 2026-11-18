"""Mock 工具层单测：订单/库存/物流/优惠券/工单的统一结果契约。"""
from __future__ import annotations

from backend.app.tools.coupon import CouponTool
from backend.app.tools.inventory import InventoryTool
from backend.app.tools.logistics_api import LogisticsTool
from backend.app.tools.order_query import OrderQueryTool
from backend.app.tools.ticket import InMemoryTicketStore, TicketTool


def test_order_found_and_missing():
    tool = OrderQueryTool()
    res = tool.run(order_id="ord20261105001")  # 大小写不敏感
    assert res.ok and res.data["order"]["paid_amount"] == 1299.0
    bad = tool.run(order_id="ORD99999999999")
    assert not bad.ok and "ORDER_NOT_FOUND" in bad.error


def test_order_requires_params():
    res = OrderQueryTool().run()
    assert not res.ok


def test_order_list_by_user():
    res = OrderQueryTool().run(user_id="U1001")
    assert res.ok and len(res.data["orders"]) == 2


def test_inventory_by_sku_and_keyword():
    tool = InventoryTool()
    res = tool.run(skus=["ZP-DG-001"])
    assert res.ok and res.data["items"][0]["price"] == 129.0
    kw = tool.run(keyword="台灯")
    assert kw.ok and len(kw.data["items"]) == 1
    bad = tool.run(skus=["ZP-NOPE"])
    assert not bad.ok and "SKU_NOT_FOUND" in bad.error


def test_logistics_abnormal_flag():
    tool = LogisticsTool()
    normal = tool.run(order_id="ORD20261105001")
    assert normal.ok and normal.data["abnormal"] is False
    abnormal = tool.run(order_id="ORD20261114005")
    assert abnormal.ok and abnormal.data["abnormal"] is True
    missing = tool.run(order_id="ORD20261112004")
    assert not missing.ok and "TRACKING_NOT_FOUND" in missing.error


def test_coupon_user_and_code():
    tool = CouponTool()
    by_user = tool.run(user_id="U1001")
    assert by_user.ok and by_user.data["usable_count"] == 2
    by_code = tool.run(coupon_code="CP2026B")
    assert by_code.ok and by_code.data["coupon"]["status"] == "已过期"
    assert not tool.run(coupon_code="CPXXXX").ok


def test_ticket_lifecycle():
    tool = TicketTool(store=InMemoryTicketStore())
    created = tool.run(action="create", order_id="ORD20261112004", user_id="U1001",
                       kind="return", summary="测试退货")
    assert created.ok and created.data["ticket_id"] == "TK00001"
    updated = tool.run(action="update", ticket_id="TK00001", status="resolved")
    assert updated.ok and updated.data["status"] == "resolved"
    assert tool.run(action="update", ticket_id="TK99999", status="resolved").ok is False


def test_ticket_unknown_action_and_missing_params():
    tool = TicketTool(store=InMemoryTicketStore())
    assert not tool.run(action="delete", order_id="X").ok
    assert not tool.run(action="create", order_id="X").ok  # 缺 user_id/kind/summary


def test_tool_result_to_dict_contract():
    res = InventoryTool().run(skus=["ZP-SF-001"])
    d = res.to_dict()
    assert set(d) == {"ok", "tool", "data", "error", "latency_ms", "meta"}
    assert d["tool"] == "inventory"
