"""Decision layer: observations so far -> next RunConfig(s).

Consumes a fitted surrogate and the domain; emits query points. Never touches
a trajectory. The acquisition function is where the central result of Stage 1
lives — naive max-variance is expected to LOSE against the corrected variants
(§3.6.4), and reproducing that failure deliberately is the point.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from mdal.domain import Domain
from mdal.surrogate.base import Surrogate


@runtime_checkable
class Acquisition(Protocol):
    name: str

    def propose(
        self, surrogate: Surrogate, domain: Domain, observed_X: np.ndarray, batch: int = 1
    ) -> np.ndarray:
        """Return `batch` next query points (shape [batch, domain.ndim])."""
        ...
