"""物流轨迹查询工具（承运商为虚构名称）。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Any

from backend.app.tools.base import BaseTool, ToolError

# 异常状态集合：命中后由物流子图触发升级人工/工单
ABNORMAL_STATUSES = {"异常滞留", "丢件", "破损退回"}


def _ts_days_ago(n: int, hh: int, mm: int) -> str:
    """轨迹时间相对今天生成，与演示订单的窗口逻辑保持一致。"""
    d = date.today() - timedelta(days=n)
    return f"{d.isoformat()} {hh:02d}:{mm:02d}"


def _md(days_ahead: int) -> str:
    d = date.today() + timedelta(days=days_ahead)
    return f"{d.month:02d}-{d.day:02d}"


DEMO_TRACKING: dict[str, dict[str, Any]] = {
    "ORD20261105001": {
        "order_id": "ORD20261105001", "carrier": "智铺速运", "waybill": "ZPYD2026110500001",
        "status": "运输中",
        "events": [
            {"time": _ts_days_ago(12, 9, 20), "desc": "商品已从杭州仓发出"},
            {"time": _ts_days_ago(10, 14, 5), "desc": "到达南京转运中心"},
            {"time": _ts_days_ago(9, 8, 40), "desc": f"运输中，预计 {_md(1)} 送达"},
        ],
    },
    "ORD20261108002": {
        "order_id": "ORD20261108002", "carrier": "智铺速运", "waybill": "ZPYD2026110800002",
        "status": "已签收",
        "events": [
            {"time": _ts_days_ago(4, 10, 0), "desc": "到达上海闵行配送站"},
            {"time": _ts_days_ago(3, 16, 32), "desc": "已由本人签收，大件送装一体完成"},
        ],
    },
    "ORD20261114005": {
        "order_id": "ORD20261114005", "carrier": "迅达物流", "waybill": "XDL2026111400005",
        "status": "异常滞留",
        "events": [
            {"time": _ts_days_ago(3, 11, 12), "desc": "到达郑州转运中心"},
            {"time": _ts_days_ago(2, 9, 0), "desc": "中转延误：分拨场爆仓，包裹滞留"},
        ],
    },
}


class LogisticsProvider(ABC):
    @abstractmethod
    def track(self, order_id: str) -> dict[str, Any] | None: ...


class MockLogisticsProvider(LogisticsProvider):
    def track(self, order_id: str) -> dict[str, Any] | None:
        return DEMO_TRACKING.get(order_id.upper())


class HttpLogisticsProvider(LogisticsProvider):  # pragma: no cover
    """真实物流开放平台/自研轨迹服务占位。"""

    def __init__(self, base_url: str, app_key: str) -> None:
        self.base_url = base_url
        self.app_key = app_key

    def track(self, order_id: str) -> dict[str, Any] | None:
        raise NotImplementedError("需对接内部物流轨迹服务")


class LogisticsTool(BaseTool):
    name = "logistics_api"
    description = "按订单号查询物流轨迹与当前状态，异常件返回异常标记。"

    def __init__(self, settings=None, provider: LogisticsProvider | None = None) -> None:
        super().__init__(settings)
        self.provider = provider or MockLogisticsProvider()

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"},
            },
            "required": ["order_id"],
        }

    def _execute(self, order_id: str) -> Any:
        record = self.provider.track(order_id)
        if record is None:
            raise ToolError(f"TRACKING_NOT_FOUND: {order_id}")
        result = dict(record)
        result["abnormal"] = record["status"] in ABNORMAL_STATUSES
        return result
