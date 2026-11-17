"""全局配置模块。

设计原则：所有外部依赖（LLM、MySQL、Redis、Milvus）均通过环境变量注入；
任一配置缺失时自动降级为离线 Mock 模式，保证无密钥也能跑通单测与评测。
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """运行期配置快照，全部字段可由环境变量覆盖（见 .env.example）。"""

    app_name: str = "智铺 ShopMind"
    # --- 服务鉴权与限流 ---
    api_key: str = ""
    rate_limit_qps: float = 5.0
    debug: bool = True
    # --- 外部依赖地址（留空 = 离线 Mock）---
    mysql_dsn: str = ""
    redis_url: str = ""
    milvus_uri: str = ""
    milvus_collection: str = "shopmind_kb"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "shopmind-default-model"
    llm_timeout_s: float = 30.0
    # --- 检索 ---
    embedding_dim: int = 256
    retrieval_top_k: int = 5
    rrf_k: int = 60
    # --- 路由置信阈值 ---
    route_answer_threshold: float = 0.75   # >= 直接派发子图
    route_clarify_threshold: float = 0.55  # < 触发澄清追问
    # --- 退款分级阈值（元）---
    auto_refund_max: float = 50.0
    sandbox_refund_max: float = 200.0
    # --- 幻觉治理 ---
    grounded_min_ratio: float = 0.8        # 生成句与证据一致性下限
    tool_timeout_s: float = 3.0            # 工具调用超时
    # --- 会话记忆 ---
    memory_window: int = 8


_cached: Settings | None = None


def get_settings() -> Settings:
    """读取环境变量并缓存配置快照（进程级单例）。"""
    global _cached
    if _cached is not None:
        return _cached
    _cached = Settings(
        api_key=os.getenv("SHOPMIND_API_KEY", "").strip(),
        rate_limit_qps=_env_float("SHOPMIND_RATE_QPS", 5.0),
        debug=_env_bool("SHOPMIND_DEBUG", True),
        mysql_dsn=os.getenv("MYSQL_DSN", "").strip(),
        redis_url=os.getenv("REDIS_URL", "").strip(),
        milvus_uri=os.getenv("MILVUS_URI", "").strip(),
        milvus_collection=os.getenv("MILVUS_COLLECTION", "shopmind_kb").strip() or "shopmind_kb",
        llm_base_url=os.getenv("LLM_BASE_URL", "").strip(),
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_model=os.getenv("LLM_MODEL", "shopmind-default-model").strip(),
        llm_timeout_s=_env_float("LLM_TIMEOUT_S", 30.0),
        route_answer_threshold=_env_float("ROUTE_ANSWER_THRESHOLD", 0.75),
        route_clarify_threshold=_env_float("ROUTE_CLARIFY_THRESHOLD", 0.55),
        auto_refund_max=_env_float("AUTO_REFUND_MAX", 50.0),
        sandbox_refund_max=_env_float("SANDBOX_REFUND_MAX", 200.0),
        grounded_min_ratio=_env_float("GROUNDED_MIN_RATIO", 0.8),
    )
    return _cached


def reset_settings() -> None:
    """测试辅助：清空配置缓存。"""
    global _cached
    _cached = None
