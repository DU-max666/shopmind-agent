"""订单查询工具：Provider 抽象 + 内置 Mock 数据 + 真实接口占位。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Any

from backend.app.tools.base import BaseTool, ToolError


def _days_ago(n: int) -> str:
    """演示数据日期相对今天生成，避免价保/退货窗口随时间失效。"""
    return (date.today() - timedelta(days=n)).isoformat()


# 演示订单库（全部虚构数据；日期相对当前时间动态生成，
# 保证签收后 7 天退货窗口、下单后 30 天价保窗口恒可演示）
DEMO_ORDERS: dict[str, dict[str, Any]] = {
    "ORD20261105001": {
        "order_id": "ORD20261105001", "user_id": "U1001", "channel": "self",
        "items": [{"sku": "ZP-SF-001", "name": "北欧布艺三人沙发", "qty": 1, "paid": 1299.0}],
        "paid_amount": 1299.0, "status": "已发货", "order_date": _days_ago(13),
        "sign_date": None, "coupon_code": None,
    },
    "ORD20261108002": {
        "order_id": "ORD20261108002", "user_id": "U1002", "channel": "self",
        "items": [{"sku": "ZP-CP-001", "name": "独立袋装弹簧乳胶床垫1.8m", "qty": 1, "paid": 1899.0}],
        "paid_amount": 1899.0, "status": "已签收", "order_date": _days_ago(10),
        "sign_date": _days_ago(3), "coupon_code": "CP2026A",
    },
    "ORD20261020003": {
        "order_id": "ORD20261020003", "user_id": "U1003", "channel": "market",
        "items": [{"sku": "ZP-DG-001", "name": "护眼折叠台灯", "qty": 2, "paid": 159.0}],
        "paid_amount": 318.0, "status": "已签收", "order_date": _days_ago(10),
        "sign_date": _days_ago(8), "coupon_code": None,
    },
    "ORD20261112004": {
        "order_id": "ORD20261112004", "user_id": "U1001", "channel": "self",
        "items": [{"sku": "ZP-ZC-001", "name": "天然乳胶枕", "qty": 2, "paid": 249.0}],
        "paid_amount": 498.0, "status": "待发货", "order_date": _days_ago(3),
        "sign_date": None, "coupon_code": "CP2026C",
    },
    "ORD20261114005": {
        "order_id": "ORD20261114005", "user_id": "U1004", "channel": "self",
        "items": [{"sku": "ZP-ZT-001", "name": "白橡木升降书桌120cm", "qty": 1, "paid": 899.0}],
        "paid_amount": 899.0, "status": "已发货", "order_date": _days_ago(4),
        "sign_date": None, "coupon_code": None,
    },
}


class OrderProvider(ABC):
    """订单数据源抽象：Mock 与真实电商中台接口共用同一契约。"""

    @abstractmethod
    def get_order(self, order_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def list_orders(self, user_id: str) -> list[dict[str, Any]]: ...


class MockOrderProvider(OrderProvider):
    """内置演示数据 Provider，无外部依赖。"""

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        return DEMO_ORDERS.get(order_id.upper())

    def list_orders(self, user_id: str) -> list[dict[str, Any]]:
        return [o for o in DEMO_ORDERS.values() if o["user_id"] == user_id]


class HttpOrderProvider(OrderProvider):  # pragma: no cover
    """真实电商中台接口占位：私有化部署时按内部 API 契约实现。"""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url
        self.token = token

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        raise NotImplementedError("需对接内部订单中台 API")

    def list_orders(self, user_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError("需对接内部订单中台 API")


class OrderQueryTool(BaseTool):
    """按订单号/用户查询订单，供所有子图消费。"""

    name = "order_query"
    description = "查询订单详情（状态、金额、下单/签收日期、SKU 明细）。"

    def __init__(self, settings=None, provider: OrderProvider | None = None) -> None:
        super().__init__(settings)
        self.provider = provider or MockOrderProvider()

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号，如 ORD20261105001"},
                "user_id": {"type": "string", "description": "用户 ID，按用户列出订单"},
            },
            "required": [],
        }

    def _execute(self, order_id: str | None = None, user_id: str | None = None) -> Any:
        if order_id:
            order = self.provider.get_order(order_id)
            if order is None:
                raise ToolError(f"ORDER_NOT_FOUND: {order_id}")
            return {"order": order}
        if user_id:
            return {"orders": self.provider.list_orders(user_id)}
        raise ToolError("PARAM_MISSING: 至少提供 order_id 或 user_id")
