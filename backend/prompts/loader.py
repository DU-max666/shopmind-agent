"""提示词加载器：从本目录读取 `<name>.md`，进程内缓存。"""
from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent
_CACHE: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """加载指定 Agent 的系统提示词。"""
    if name not in _CACHE:
        path = _PROMPT_DIR / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"提示词缺失: {path}")
        _CACHE[name] = path.read_text(encoding="utf-8").strip()
    return _CACHE[name]
