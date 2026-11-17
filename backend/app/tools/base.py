"""工具层基础设施：统一结果结构、参数校验、超时执行。"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.app.config import Settings, get_settings

_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="shopmind-tool")


class ToolError(Exception):
    """工具业务错误（参数非法、对象不存在等）。"""


class ToolTimeout(ToolError):
    """工具调用超时。"""


@dataclass
class ToolResult:
    """所有工具的统一返回结构，便于结果校验层与审计日志消费。"""

    ok: bool
    tool: str
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "tool": self.tool, "data": self.data,
                "error": self.error, "latency_ms": self.latency_ms, "meta": self.meta}


def run_with_timeout(fn: Callable[..., Any], timeout_s: float, *args: Any, **kwargs: Any) -> Any:
    """在线程池中执行 fn，超时抛出 ToolTimeout。"""
    future = _EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout as exc:
        future.cancel()
        raise ToolTimeout(f"工具调用超时（>{timeout_s}s）") from exc


def validate_required(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """按 JSON Schema 的 required 字段做最小参数校验，返回缺失字段列表。"""
    missing = []
    for key in schema.get("required", []):
        if key not in args or args[key] in (None, ""):
            missing.append(key)
    return missing


class BaseTool(ABC):
    """工具基类：子类实现 name/description/json_schema/_execute。"""

    name: str = "base"
    description: str = ""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @abstractmethod
    def json_schema(self) -> dict[str, Any]:
        """OpenAI Function Calling 格式的 JSON Schema（parameters 部分）。"""

    @abstractmethod
    def _execute(self, **kwargs: Any) -> Any:
        """实际执行逻辑，返回任意可序列化数据；业务错误抛 ToolError。"""

    def run(self, **kwargs: Any) -> ToolResult:
        """统一入口：参数校验 → 带超时执行 → 异常归一化 → 计时。"""
        schema = self.json_schema()
        missing = validate_required(schema, kwargs)
        if missing:
            return ToolResult(ok=False, tool=self.name,
                              error=f"缺少必填参数: {', '.join(missing)}")
        started = time.perf_counter()
        try:
            data = run_with_timeout(self._execute, self.settings.tool_timeout_s, **kwargs)
            latency = (time.perf_counter() - started) * 1000
            return ToolResult(ok=True, tool=self.name, data=data, latency_ms=round(latency, 2))
        except ToolTimeout as exc:
            return ToolResult(ok=False, tool=self.name, error=str(exc))
        except ToolError as exc:
            return ToolResult(ok=False, tool=self.name, error=str(exc))
        except Exception as exc:  # 兜底：不把内部栈外泄给对话层
            return ToolResult(ok=False, tool=self.name, error=f"工具内部错误: {exc.__class__.__name__}")
