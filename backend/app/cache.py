"""缓存与限流基础设施。

- BaseCache：统一 KV 抽象，会话记忆/画像/幂等键都走这一层；
- InMemoryCache：进程内 TTL 缓存，离线演示与单测默认后端；
- RedisCache：懒加载 redis 客户端，生产配置 REDIS_URL 后启用；
- TokenBucketLimiter：令牌桶限流器，API 网关层按 key 维度限 QPS。
"""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from backend.app.config import Settings, get_settings


class BaseCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl_s: float | None = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class InMemoryCache(BaseCache):
    """线程安全内存缓存：惰性过期（读时判断），演示规模足够。"""

    def __init__(self, default_ttl_s: float = 300.0) -> None:
        self.default_ttl_s = default_ttl_s
        self._lock = threading.Lock()
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expire_at, value = item
            if expire_at < time.monotonic():
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        ttl = self.default_ttl_s if ttl_s is None else ttl_s
        with self._lock:
            self._data[key] = (time.monotonic() + ttl, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


class RedisCache(BaseCache):  # pragma: no cover
    """Redis 后端：值统一 JSON 序列化，TTL 交给 Redis 自身管理。"""

    def __init__(self, redis_url: str) -> None:
        import redis  # 懒加载：离线环境不装 redis 也能跑

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def set(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        if ttl_s:
            self._client.set(key, payload, ex=int(ttl_s))
        else:
            self._client.set(key, payload)

    def delete(self, key: str) -> None:
        self._client.delete(key)


class TokenBucketLimiter:
    """令牌桶：capacity 为突发上限，rate 为每秒回补令牌数。"""

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self.rate = max(rate, 0.001)
        self.capacity = capacity if capacity is not None else max(self.rate, 1.0)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def allow(self, cost: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False

    def wait_seconds(self) -> float:
        """距离下一个可用令牌的估算等待秒数（用于 429 的 Retry-After）。"""
        with self._lock:
            deficit = 1.0 - self._tokens
            return max(0.0, deficit / self.rate)


def get_cache(settings: Settings | None = None) -> BaseCache:
    """配置了 REDIS_URL 用 Redis，否则回退内存缓存。"""
    s = settings or get_settings()
    if s.redis_url:
        try:
            return RedisCache(s.redis_url)
        except Exception:
            return InMemoryCache()
    return InMemoryCache()
