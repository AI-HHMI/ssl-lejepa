"""LLM: Large Microscopy Models.

Self-Supervised Learning with LeJEPA (Invariance + SIGReg) on 3D microscopy volumes.
"""

from llm.models import Lejepa, LejepaConfig

__version__ = "0.1.0"
__all__ = ["Lejepa", "LejepaConfig"]
