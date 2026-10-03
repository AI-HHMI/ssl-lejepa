from __future__ import annotations

from typing import Literal

F32Mode = Literal["highest", "high", "medium"]
Tup3Int = tuple[int,int,int]
Views = Literal["basic", "displace", "none"]  # 'none': Lejepa skips view making
TrainData = Literal["hemibrain_eb", "hemibrain_wide"]  # keys of lib.data.TRAIN_BOXES
Decoder = Literal["linear", "unetr"]  # experiment.probe(): lib.probe.fit_probe or lib.decoders.unetr.fit_unetr

