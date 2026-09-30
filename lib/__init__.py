"""LLM: Large Microscopy Models.

Self-Supervised Learning with LeJEPA (Invariance + SIGReg) on 3D microscopy volumes.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names below are real only for the type checker; __getattr__ loads them lazily at runtime
    from lib.losses import LejepaOutput
    from lib.models import Lejepa, LejepaConfig

__version__ = "0.1.0"
__all__ = [
    "Lejepa",
    "LejepaConfig",
    "LejepaOutput",
]


def __getattr__(name):
    """Lazy: lib.util's CLI helpers (json/os/re/... only) shouldn't pay for torch + the model stack just because
    they live in the same package. Loaded and cached on first access to lib.Lejepa/LejepaConfig/LejepaOutput."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from lib.losses import LejepaOutput
    from lib.models import Lejepa, LejepaConfig
    globals().update(Lejepa=Lejepa, LejepaConfig=LejepaConfig, LejepaOutput=LejepaOutput)
    return globals()[name]
