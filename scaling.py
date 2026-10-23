"""Model sizes and compute budgets of e00/scaling-law's IsoFLOP grid, for designing size x budget sweeps: parameter counts, FLOPs
per sample and estimated step times of each size (cost model only; analysis reads back each run's real compute)."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from lib.specs import Train, lejepa_config
from lib.models import Lejepa, train_flops

N_PARAMS_CACHE = Path(".cache/lejepa_n_params.json")  # machine-local (gitignored): see lejepa_n_params
# Model sizes and compute budgets of e00/scaling-law's IsoFLOP grid (scaling_budgets), for designing size x budget sweeps.
SIZES = {"xs": (4, 256), "s": (6, 384), "m": (12, 512), "l": (16, 768), "xl": (24, 1024)}  # (n_layers, width); m = Train' default
SCALING_BATCH = 64  # per GPU, fixed across sizes (== gpu_b300's validated default at m); xl OOMs at it on one B300
ANCHOR_STEP_S_M_B84 = 0.32  # cudagraph-fix/d3: 1425 ktok/s at b84, flash + eager-pe cudagraphs (m: n_layers=12, width=512).
# Measured on the multi-view objective (2 global + 4 local views, about 5.5k token-passes per sample); the masked-token objective
# costs about the same by FLOPs (sample_flops) but has no measurement of its own yet.
SCALING_GRID = (12, 12, 12)  # token grid of the view cost estimates assume (a 96^3 view at patch 8)
BUDGET_HOURS_AT_M = {"c1": 1, "c2": 2, "c3": 4, "c4": 8}  # m's wall-clock at each budget; other sizes get more/fewer steps


def lejepa_n_params(n_layers: int, width: int) -> tuple[int, int, int]:
    """(encoder, predictor, projector) parameter counts of the scaling-law grid's Lejepa at this size. Building the model just to
    count them takes seconds, so they are cached in N_PARAMS_CACHE, keyed by size and by a hash of the model and encoder
    code (a changed architecture recounts)."""
    lib = Path(__file__).resolve().parent / "lib"  # the snapshot's own, when run from one
    code = hashlib.sha256(b"".join(p.read_bytes() for d in ("models", "encoders") for p in sorted((lib / d).glob("*.py")))).hexdigest()[:12]
    key = f"{code}/{n_layers}x{width}"
    cache = json.loads(N_PARAMS_CACHE.read_text()) if N_PARAMS_CACHE.is_file() else {}
    if key not in cache:
        model = Lejepa(lejepa_config(replace(Train(), n_layers=n_layers, width=width, views="displace", global_size=(96, 96, 96))))
        cache[key] = [model._n_encoder_params, model._n_predictor_params, model._n_projector_params]
        N_PARAMS_CACHE.parent.mkdir(exist_ok=True)
        N_PARAMS_CACHE.write_text(json.dumps(cache))
    return tuple(cache[key])

def sample_flops(name: str, grid=SCALING_GRID) -> float:
    """Model training FLOPs per sample of SIZES[name] (exact: lib.models.lejepa.train_flops, which Lejepa.forward's own
    out.n_flops uses), for a view with this token grid."""
    n_layers, width = SIZES[name]
    return train_flops(lejepa_config(replace(Train(), n_layers=n_layers, width=width)), grid, *lejepa_n_params(n_layers, width))

def step_seconds(name: str, batch: int, grid=SCALING_GRID) -> float:
    """Estimated seconds per training step of SIZES[name] at this per-GPU batch: ANCHOR_STEP_S_M_B84 scaled by FLOPs per
    step relative to m at batch 84. An a-priori estimate (it under-predicted small sizes' cost in mia-muvit's e07)."""
    return ANCHOR_STEP_S_M_B84 * (batch * sample_flops(name, grid)) / (84 * sample_flops("m"))

def scaling_budgets() -> dict[str, float]:
    """Training FLOPs of each compute budget (c1..c4): BUDGET_HOURS_AT_M x m's estimated FLOPs per hour at SCALING_BATCH.
    A size's steps at a budget: budget / (sample_flops(name) * batch); e00/scaling-law set max_hours to 2x
    step_seconds' estimate of those steps, as a safety cap. Only for grid design: analysis reads back each run's real
    EFLOP (analysis_utils.run_compute), so a wrong estimate over/under-shoots the nominal budget rather than corrupting a fit."""
    flops_per_hour_m = 3600 / step_seconds("m", SCALING_BATCH) * sample_flops("m") * SCALING_BATCH
    return {b: h * flops_per_hour_m for b, h in BUDGET_HOURS_AT_M.items()}
