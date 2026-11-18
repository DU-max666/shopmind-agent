"""会话记忆与用户画像：全部落在缓存层，无 DB 依赖即可运行。

- SessionMemory：按 session_id 维护最近 N 轮问答滑窗（settings.memory_window），
  供多轮追问（"刚才那个沙发多少钱"）与上下文补全使用；
- UserProfileStore：按 user_id 沉淀长期偏好（品类/风格/预算）与历史工单号，
  导购子图可在提示词中注入画像摘要，售后子图可快速带出在办工单。
"""
from __future__ import annotations

import threading
from typing import Any

from backend.app.cache import BaseCache, InMemoryCache
from backend.app.config import Settings, get_settings
from backend.app.graphs.guide import extract_constraints

SESSION_KEY = "shopmind:session:{session_id}"
PROFILE_KEY = "shopmind:profile:{user_id}"


class SessionMemory:
    """会话滑窗记忆：超出窗口自动丢弃最旧一轮。"""

    def __init__(self, cache: BaseCache, window: int = 8, ttl_s: float = 3600.0) -> None:
        self.cache = cache
        self.window = max(2, window)
        self.ttl_s = ttl_s

    def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        history = self.recent(session_id)
        history.append({"user": user_text, "assistant": assistant_text})
        self.cache.set(SESSION_KEY.format(session_id=session_id),
                       history[-self.window:], ttl_s=self.ttl_s)

    def recent(self, session_id: str) -> list[dict[str, str]]:
        data = self.cache.get(SESSION_KEY.format(session_id=session_id))
        return list(data) if isinstance(data, list) else []

    def last_intent(self, session_id: str) -> str:
        """供路由层做"追问延续"判断：上一轮意图可作弱先验。"""
        data = self.cache.get(SESSION_KEY.format(session_id=session_id) + ":intent")
        return str(data) if data else ""

    def set_last_intent(self, session_id: str, intent: str) -> None:
        self.cache.set(SESSION_KEY.format(session_id=session_id) + ":intent",
                       intent, ttl_s=self.ttl_s)


class UserProfileStore:
    """用户画像：品类/风格偏好计数 + 最近预算 + 历史工单。"""

    def __init__(self, cache: BaseCache, ttl_s: float = 86400.0) -> None:
        self.cache = cache
        self.ttl_s = ttl_s
        self._lock = threading.Lock()

    def _load(self, user_id: str) -> dict[str, Any]:
        data = self.cache.get(PROFILE_KEY.format(user_id=user_id))
        if isinstance(data, dict):
            return data
        return {"categories": {}, "styles": {}, "last_budget": None, "tickets": []}

    def record_turn(self, user_id: str, state: dict[str, Any]) -> None:
        """从一轮对话状态中增量更新画像（无 user_id 时跳过）。"""
        if not user_id:
            return
        with self._lock:
            profile = self._load(user_id)
            slots = state.get("slots") or {}
            for cat in slots.get("categories") or ([slots["category"]] if slots.get("category") else []):
                profile["categories"][cat] = profile["categories"].get(cat, 0) + 1
            if slots.get("style"):
                profile["styles"][slots["style"]] = profile["styles"].get(slots["style"], 0) + 1
            if slots.get("budget") is not None:
                profile["last_budget"] = slots["budget"]
            ticket_id = state.get("ticket_id")
            if ticket_id and ticket_id not in profile["tickets"]:
                profile["tickets"].append(ticket_id)
                profile["tickets"] = profile["tickets"][-10:]
            self.cache.set(PROFILE_KEY.format(user_id=user_id), profile, ttl_s=self.ttl_s)

    def profile(self, user_id: str) -> dict[str, Any]:
        return self._load(user_id)

    def prompt_summary(self, user_id: str) -> str:
        """压缩成一行注入导购提示词，空画像返回空串。"""
        p = self._load(user_id)
        if not (p["categories"] or p["styles"] or p["last_budget"] or p["tickets"]):
            return ""
        top_cat = max(p["categories"], key=p["categories"].get) if p["categories"] else "-"
        top_style = max(p["styles"], key=p["styles"].get) if p["styles"] else "-"
        budget = p["last_budget"] if p["last_budget"] is not None else "-"
        tickets = ",".join(p["tickets"][-3:]) if p["tickets"] else "-"
        return f"常购品类:{top_cat} 偏好风格:{top_style} 最近预算:{budget} 在办工单:{tickets}"


class MemoryService:
    """组合入口：API 层只依赖它，便于整体替换存储后端。"""

    def __init__(self, settings: Settings | None = None, cache: BaseCache | None = None) -> None:
        s = settings or get_settings()
        c = cache or InMemoryCache(default_ttl_s=3600.0)
        self.session = SessionMemory(c, window=s.memory_window)
        self.profiles = UserProfileStore(c)

    def update_from_turn(self, state: dict[str, Any]) -> None:
        """一轮结束后的统一落库：滑窗 + 画像 + 上一轮意图。"""
        session_id = state.get("session_id", "")
        user_id = state.get("user_id", "")
        user_text = state.get("user_text", "")
        final = state.get("final_answer", "")
        if session_id and final:
            self.session.append_turn(session_id, user_text, final)
            self.session.set_last_intent(session_id, state.get("intent", ""))
        if user_id:
            # 用户话语没有结构化槽位时，补一次确定性约束抽取
            if not (state.get("slots") or {}).get("categories"):
                state.setdefault("slots", {}).update(extract_constraints(user_text))
            self.profiles.record_turn(user_id, state)
