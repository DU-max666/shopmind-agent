"""API 服务入口：FastAPI + SSE 流式对话 + API Key 鉴权 + 令牌桶限流。

fastapi / uvicorn 均为懒加载：未安装时本模块仍可被导入（离线单测不受影响），
仅 create_app() 会抛出明确提示。SSE 事件流与评测共用 supervisor.iter_turn_events，
保证"演示链路 = 评测链路"。

启动（安装依赖后）：
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import threading
import uuid
from typing import Any, Iterator

from backend.app.cache import TokenBucketLimiter, get_cache
from backend.app.config import Settings, get_settings
from backend.app.graphs.state import ShopMindState, make_default_context
from backend.app.graphs.supervisor import iter_turn_events, run_turn
from backend.app.memory.profile import MemoryService


def _try_build_retriever() -> Any:
    """尽力加载知识库检索器（scripts/init_kb 灌出的内存版索引）。"""
    try:
        from scripts.init_kb import build_retriever

        return build_retriever()
    except Exception:
        return None


class ChatService:
    """会话状态 + 记忆 + 共享运行上下文，进程内单例复用。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.context = make_default_context(self.settings, retriever=_try_build_retriever())
        self.memory = MemoryService(self.settings, cache=get_cache(self.settings))
        self.limiter = TokenBucketLimiter(self.settings.rate_limit_qps)
        self._sessions: dict[str, ShopMindState] = {}
        self._lock = threading.Lock()

    def get_state(self, session_id: str, user_id: str) -> ShopMindState:
        with self._lock:
            st = self._sessions.get(session_id)
            if st is None:
                st = {"session_id": session_id, "user_id": user_id, "messages": []}  # type: ignore[typeddict-item]
                self._sessions[session_id] = st
            return st

    def stream_events(self, session_id: str, user_id: str, text: str) -> Iterator[dict[str, Any]]:
        st = self.get_state(session_id, user_id)
        for event in iter_turn_events(text, st, self.context):
            yield event
        self.memory.update_from_turn(st)

    def chat(self, session_id: str, user_id: str, text: str) -> dict[str, Any]:
        st = self.get_state(session_id, user_id)
        final = run_turn(text, st, self.context)
        self.memory.update_from_turn(final)
        return {
            "session_id": session_id,
            "intent": final.get("intent", ""),
            "route_source": final.get("route_source", ""),
            "answer": final.get("final_answer", ""),
            "needs_human": final.get("needs_human", False),
            "ticket_id": final.get("ticket_id", ""),
            "ticket_status": final.get("ticket_status", ""),
            "grounded_score": final.get("grounded_score", 0.0),
        }


def create_app(settings: Settings | None = None):
    """构建 FastAPI 应用（需安装 fastapi）。"""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要 fastapi：pip install fastapi uvicorn") from exc

    try:
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要 pydantic：pip install fastapi uvicorn") from exc

    s = settings or get_settings()
    service = ChatService(s)

    class ChatRequest(BaseModel):
        text: str
        session_id: str = ""
        user_id: str = ""

    app = FastAPI(title=s.app_name, version="0.1.0")

    def check_auth(request: "Request") -> None:
        """配置了 SHOPMIND_API_KEY 才强制鉴权；离线演示默认放行。"""
        if not s.api_key:
            return
        raw = request.headers.get("X-API-Key", "")
        if not raw:
            auth = request.headers.get("Authorization", "")
            raw = auth[7:].strip() if auth.startswith("Bearer ") else ""
        if raw != s.api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    def check_rate() -> None:
        if not service.limiter.allow():
            retry = max(0.1, service.limiter.wait_seconds())
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limited"},
                headers={"Retry-After": f"{retry:.2f}"},
            )
        return None

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "app": s.app_name,
                "llm": s.llm_model if s.llm_api_key else "mock",
                "retriever": service.context.retriever is not None}

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatRequest, request: "Request"):
        denied = check_rate()
        if denied is not None:
            return denied
        check_auth(request)
        if not req.text.strip():
            raise HTTPException(status_code=422, detail="text 不能为空")
        session_id = req.session_id or uuid.uuid4().hex

        def sse() -> Iterator[str]:
            for event in service.stream_events(session_id, req.user_id, req.text):
                payload = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {event['event']}\ndata: {payload}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.post("/api/chat")
    async def chat(req: ChatRequest, request: "Request"):
        denied = check_rate()
        if denied is not None:
            return denied
        check_auth(request)
        if not req.text.strip():
            raise HTTPException(status_code=422, detail="text 不能为空")
        session_id = req.session_id or uuid.uuid4().hex
        return service.chat(session_id, req.user_id, req.text)

    @app.get("/api/ticket/{ticket_id}")
    def ticket_detail(ticket_id: str, request: "Request"):
        check_auth(request)
        tool = service.context.tools["ticket"]
        record = tool.store.get(ticket_id)
        if record is None:
            raise HTTPException(status_code=404, detail="ticket not found")
        return record

    return app


# uvicorn 入口：未安装 fastapi 的离线环境保持 None，不影响其他模块导入
try:
    app = create_app()
except RuntimeError:  # pragma: no cover
    app = None
