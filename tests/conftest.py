"""pytest 共享配置：把仓库根加入 sys.path，使测试可直接 import backend.app.*。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
