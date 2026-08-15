"""Surrogate model layer."""

from mdal.surrogate.base import Surrogate
from mdal.surrogate.gp import HeteroscedasticGP

__all__ = ["Surrogate", "HeteroscedasticGP"]
