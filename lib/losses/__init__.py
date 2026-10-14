"""Loss functions for self-supervised learning."""

from lib.losses.sigreg import LejepaOutput, SIGReg, SIGRegConfig, lejepa_loss

__all__ = ["LejepaOutput", "SIGReg", "SIGRegConfig", "lejepa_loss"]
