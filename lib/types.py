from __future__ import annotations

from typing import Literal

F32Mode = Literal["highest", "high", "medium"]
Tup3Int = tuple[int,int,int]
Views = Literal["basic", "displace", "none"]  # 'none': Lejepa skips view making

