"""库存与在售价查询工具。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.app.tools.base import BaseTool, ToolError

# 虚构 SKU 目录：价格与库存与知识库说明书保持一致
DEMO_INVENTORY: dict[str, dict[str, Any]] = {
    "ZP-SF-001": {"name": "北欧布艺三人沙发", "price": 1299.0, "stock": 320, "shop": "自营"},
    "ZP-SF-002": {"name": "原木风实木框架沙发", "price": 2599.0, "stock": 88, "shop": "自营"},
    "ZP-CP-001": {"name": "独立袋装弹簧乳胶床垫1.8m", "price": 1899.0, "stock": 150, "shop": "自营"},
    "ZP-CP-002": {"name": "椰棕硬支撑床垫1.5m", "price": 1099.0, "stock": 210, "shop": "入驻"},
    "ZP-ZT-001": {"name": "白橡木升降书桌120cm", "price": 899.0, "stock": 64, "shop": "自营"},
    "ZP-ZT-002": {"name": "岩板餐桌1.4m", "price": 1699.0, "stock": 45, "shop": "入驻"},
    "ZP-DG-001": {"name": "护眼折叠台灯", "price": 129.0, "stock": 980, "shop": "自营"},  # 双11活动价，日常价 159 元
    "ZP-DG-002": {"name": "落地阅读灯", "price": 349.0, "stock": 260, "shop": "自营"},
    "ZP-SN-001": {"name": "多层收纳柜", "price": 499.0, "stock": 130, "shop": "入驻"},
    "ZP-ZC-001": {"name": "天然乳胶枕", "price": 249.0, "stock": 540, "shop": "自营"},
}


class InventoryProvider(ABC):
    @abstractmethod
    def get(self, sku: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def search(self, keyword: str) -> list[dict[str, Any]]: ...


class MockInventoryProvider(InventoryProvider):
    def get(self, sku: str) -> dict[str, Any] | None:
        item = DEMO_INVENTORY.get(sku.upper())
        if item is None:
            return None
        return {"sku": sku.upper(), **item}

    def search(self, keyword: str) -> list[dict[str, Any]]:
        out = []
        for sku, item in DEMO_INVENTORY.items():
            if keyword and keyword in item["name"]:
                out.append({"sku": sku, **item})
        return out


class HttpInventoryProvider(InventoryProvider):  # pragma: no cover
    """真实库存中心接口占位。"""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url
        self.token = token

    def get(self, sku: str) -> dict[str, Any] | None:
        raise NotImplementedError("需对接内部库存中心 API")

    def search(self, keyword: str) -> list[dict[str, Any]]:
        raise NotImplementedError("需对接内部库存中心 API")


class InventoryTool(BaseTool):
    name = "inventory"
    description = "查询 SKU 实时库存与在售价；导购推荐话术中的价格/库存必须来自本工具。"

    def __init__(self, settings=None, provider: InventoryProvider | None = None) -> None:
        super().__init__(settings)
        self.provider = provider or MockInventoryProvider()

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skus": {"type": "array", "items": {"type": "string"},
                         "description": "待查询 SKU 列表"},
                "keyword": {"type": "string", "description": "按商品名关键词模糊搜索"},
            },
            "required": [],
        }

    def _execute(self, skus: list[str] | None = None, keyword: str | None = None) -> Any:
        if skus:
            items = []
            for sku in skus:
                item = self.provider.get(sku)
                if item is None:
                    raise ToolError(f"SKU_NOT_FOUND: {sku}")
                items.append(item)
            return {"items": items}
        if keyword:
            return {"items": self.provider.search(keyword)}
        raise ToolError("PARAM_MISSING: 至少提供 skus 或 keyword")
