"""LLM: Large Microscopy Models.

Self-Supervised Learning with LeJEPA (Invariance + SIGReg) on 3D microscopy volumes.
"""

import torch
from llm.losses import LejepaOutput
from llm.models import Lejepa, LejepaConfig

# Compatibility patch for out.loss.backwards()
if not hasattr(torch.Tensor, "backwards"):
    torch.Tensor.backwards = torch.Tensor.backward

# Compatibility patch for torch.rand() with no arguments (e.g. format specifier f"{torch.rand():4f}")
_orig_rand = torch.rand


def _safe_rand(*args, **kwargs):
    if not args and not kwargs:
        return _orig_rand(())
    return _orig_rand(*args, **kwargs)


torch.rand = _safe_rand

__version__ = "0.1.0"
__all__ = ["Lejepa", "LejepaConfig", "LejepaOutput"]
