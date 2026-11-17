"""工单工具：创建/更新工单 + 退款沙箱预演/提交/回滚。

沙箱语义（与 docs/hitl.md 对应）：
- rehearse_refund 只在预演账本写入快照，不产生真实资金动作；
- commit_refund 必须携带人工确认令牌 confirm_token 才允许从预演转正式；
- rollback_refund 依据预演快照回滚，保证高风险动作可撤销。
"""
from __future__ import annotations

import itertools
import threading
from abc import ABC, abstractmethod
from typing import Any

from backend.app.tools.base import BaseTool, ToolError

_VALID_ACTIONS = {"create", "update", "rehearse_refund", "commit_refund", "rollback_refund"}


class TicketStore(ABC):
    @abstractmethod
    def create(self, *, order_id: str, user_id: str, kind: str, summary: str) -> dict[str, Any]: ...

    @abstractmethod
    def update_status(self, ticket_id: str, status: str) -> dict[str, Any]: ...

    @abstractmethod
    def get(self, ticket_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def rehearse_refund(self, *, order_id: str, amount: float) -> dict[str, Any]: ...

    @abstractmethod
    def commit_refund(self, *, rehearsal_id: str, confirm_token: str) -> dict[str, Any]: ...

    @abstractmethod
    def rollback_refund(self, *, rehearsal_id: str) -> dict[str, Any]: ...


class InMemoryTicketStore(TicketStore):
    """线程安全的内存工单账本，Mock 与单测共用。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self.tickets: dict[str, dict[str, Any]] = {}
        self.rehearsals: dict[str, dict[str, Any]] = {}

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}{next(self._seq):05d}"

    def create(self, *, order_id: str, user_id: str, kind: str, summary: str) -> dict[str, Any]:
        with self._lock:
            ticket_id = self._new_id("TK")
            record = {"ticket_id": ticket_id, "order_id": order_id, "user_id": user_id,
                      "kind": kind, "summary": summary, "status": "created"}
            self.tickets[ticket_id] = record
            return dict(record)

    def update_status(self, ticket_id: str, status: str) -> dict[str, Any]:
        with self._lock:
            record = self.tickets.get(ticket_id)
            if record is None:
                raise ToolError(f"TICKET_NOT_FOUND: {ticket_id}")
            record["status"] = status
            return dict(record)

    def get(self, ticket_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self.tickets.get(ticket_id)
            return dict(record) if record else None

    def rehearse_refund(self, *, order_id: str, amount: float) -> dict[str, Any]:
        with self._lock:
            rehearsal_id = self._new_id("RH")
            snapshot = {"rehearsal_id": rehearsal_id, "order_id": order_id, "amount": amount,
                        "status": "rehearsed", "committed": False}
            self.rehearsals[rehearsal_id] = snapshot
            return dict(snapshot)

    def commit_refund(self, *, rehearsal_id: str, confirm_token: str) -> dict[str, Any]:
        with self._lock:
            snapshot = self.rehearsals.get(rehearsal_id)
            if snapshot is None:
                raise ToolError(f"REHEARSAL_NOT_FOUND: {rehearsal_id}")
            if snapshot["committed"]:
                raise ToolError("REHEARSAL_ALREADY_COMMITTED")
            if not confirm_token or not confirm_token.startswith("HITL-"):
                raise ToolError("CONFIRM_TOKEN_INVALID: 缺少以 HITL- 开头的人工确认令牌")
            snapshot["committed"] = True
            snapshot["status"] = "committed"
            return dict(snapshot)

    def rollback_refund(self, *, rehearsal_id: str) -> dict[str, Any]:
        with self._lock:
            snapshot = self.rehearsals.get(rehearsal_id)
            if snapshot is None:
                raise ToolError(f"REHEARSAL_NOT_FOUND: {rehearsal_id}")
            snapshot["committed"] = False
            snapshot["status"] = "rolled_back"
            return dict(snapshot)


# 模块级默认账本（演示/单测共享），测试可 reset_default_store() 清空
_default_store = InMemoryTicketStore()


def reset_default_store() -> None:
    global _default_store
    _default_store = InMemoryTicketStore()


class TicketTool(BaseTool):
    name = "ticket"
    description = ("工单全生命周期：create 创建、update 更新状态、rehearse_refund 退款沙箱预演、"
                   "commit_refund 凭人工令牌转正式、rollback_refund 回滚。")

    def __init__(self, settings=None, store: TicketStore | None = None) -> None:
        super().__init__(settings)
        self.store = store or _default_store

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": sorted(_VALID_ACTIONS)},
                "ticket_id": {"type": "string"},
                "order_id": {"type": "string"},
                "user_id": {"type": "string"},
                "kind": {"type": "string", "enum": ["return", "exchange", "quality", "price_guard", "other"]},
                "summary": {"type": "string"},
                "status": {"type": "string"},
                "amount": {"type": "number"},
                "rehearsal_id": {"type": "string"},
                "confirm_token": {"type": "string", "description": "人工确认令牌，形如 HITL-xxxx"},
            },
            "required": ["action"],
        }

    def _execute(self, action: str, **kwargs: Any) -> Any:
        if action not in _VALID_ACTIONS:
            raise ToolError(f"UNKNOWN_ACTION: {action}")
        if action == "create":
            need = ("order_id", "user_id", "kind", "summary")
            missing = [k for k in need if not kwargs.get(k)]
            if missing:
                raise ToolError(f"PARAM_MISSING: {', '.join(missing)}")
            return self.store.create(order_id=kwargs["order_id"], user_id=kwargs["user_id"],
                                     kind=kwargs["kind"], summary=kwargs["summary"])
        if action == "update":
            if not kwargs.get("ticket_id") or not kwargs.get("status"):
                raise ToolError("PARAM_MISSING: ticket_id, status")
            return self.store.update_status(kwargs["ticket_id"], kwargs["status"])
        if action == "rehearse_refund":
            if not kwargs.get("order_id") or kwargs.get("amount") is None:
                raise ToolError("PARAM_MISSING: order_id, amount")
            return self.store.rehearse_refund(order_id=kwargs["order_id"], amount=float(kwargs["amount"]))
        if action == "commit_refund":
            if not kwargs.get("rehearsal_id"):
                raise ToolError("PARAM_MISSING: rehearsal_id")
            return self.store.commit_refund(rehearsal_id=kwargs["rehearsal_id"],
                                            confirm_token=kwargs.get("confirm_token", ""))
        # rollback_refund
        if not kwargs.get("rehearsal_id"):
            raise ToolError("PARAM_MISSING: rehearsal_id")
        return self.store.rollback_refund(rehearsal_id=kwargs["rehearsal_id"])
