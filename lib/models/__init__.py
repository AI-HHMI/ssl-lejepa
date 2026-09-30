"""Models module for LLM (Large Microscopy Models)."""

import torch

from lib.models.lejepa import Lejepa, LejepaConfig

# Compatibility patch for torch.rand() with no arguments (e.g. format specifier f"{torch.rand():4f}"). Lives here
# (not lib/__init__.py) so it still applies wherever the model stack is actually used, without forcing torch on
# lib.util-only importers (lib/__init__.py's __getattr__ loads this module lazily).
_orig_rand = torch.rand


def _safe_rand(*args, **kwargs):
    if not args and not kwargs:
        return _orig_rand(())
    return _orig_rand(*args, **kwargs)


torch.rand = _safe_rand

__all__ = ["Lejepa", "LejepaConfig"]
