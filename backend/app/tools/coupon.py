"""优惠券查询与核销校验工具。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.app.tools.base import BaseTool, ToolError

DEMO_COUPONS: dict[str, list[dict[str, Any]]] = {
    "U1001": [
        {"code": "CP2026A", "type": "满减", "threshold": 999.0, "amount": 100.0,
         "status": "未使用", "scope": "自营家具", "expire": "2026-12-31"},
        {"code": "CP2026C", "type": "立减", "threshold": 0.0, "amount": 20.0,
         "status": "未使用", "scope": "全平台", "expire": "2026-11-30"},
    ],
    "U1002": [
        {"code": "CP2026B", "type": "满减", "threshold": 199.0, "amount": 20.0,
         "status": "已过期", "scope": "全平台", "expire": "2026-11-01"},
    ],
    "U1003": [],
}


class CouponProvider(ABC):
    @abstractmethod
    def list_by_user(self, user_id: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get(self, code: str) -> dict[str, Any] | None: ...


class MockCouponProvider(CouponProvider):
    def list_by_user(self, user_id: str) -> list[dict[str, Any]]:
        return list(DEMO_COUPONS.get(user_id, []))

    def get(self, code: str) -> dict[str, Any] | None:
        for coupons in DEMO_COUPONS.values():
            for c in coupons:
                if c["code"] == code.upper():
                    return c
        return None


class HttpCouponProvider(CouponProvider):  # pragma: no cover
    """真实营销中心接口占位。"""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url
        self.token = token

    def list_by_user(self, user_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError("需对接内部营销中心 API")

    def get(self, code: str) -> dict[str, Any] | None:
        raise NotImplementedError("需对接内部营销中心 API")


class CouponTool(BaseTool):
    name = "coupon"
    description = "查询用户名下优惠券，或校验券码可用门槛与面额；活动子图计算到手价时必须调用。"

    def __init__(self, settings=None, provider: CouponProvider | None = None) -> None:
        super().__init__(settings)
        self.provider = provider or MockCouponProvider()

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户 ID"},
                "coupon_code": {"type": "string", "description": "券码，校验单张券"},
            },
            "required": [],
        }

    def _execute(self, user_id: str | None = None, coupon_code: str | None = None) -> Any:
        if coupon_code:
            coupon = self.provider.get(coupon_code)
            if coupon is None:
                raise ToolError(f"COUPON_NOT_FOUND: {coupon_code}")
            return {"coupon": coupon}
        if user_id:
            coupons = self.provider.list_by_user(user_id)
            usable = [c for c in coupons if c["status"] == "未使用"]
            return {"coupons": coupons, "usable_count": len(usable)}
        raise ToolError("PARAM_MISSING: 至少提供 user_id 或 coupon_code")
