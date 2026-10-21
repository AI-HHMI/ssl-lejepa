from __future__ import annotations

from typing import Literal

F32Mode = Literal["highest", "high", "medium"]
Tup3Int = tuple[int,int,int]
Views = Literal["basic", "displace", "none"]  # 'none': Lejepa skips view making
TrainData = Literal["hemibrain_eb", "hemibrain_wide", "fish2", "liconn_mouse", "microns", "lmd4"]  # keys of lib.data.TRAIN_BOXES, or JOINT_DATA

