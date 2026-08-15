"""Trajectory container — oracle output, estimator input.

Stays INSIDE the oracle+estimator layers. Never crosses into the decision
layer (that seam carries `Observation`s only).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trajectory:
    """Recorded frames from one simulation.

    `per_frame` holds the cheap direct observables (pressure, potential energy)
    as 1-D time series. `positions` is optional and only populated when a
    displacement-based estimator (MSD -> diffusion) needs it, since it is heavy.
    """

    run_hash: str
    per_frame: dict[str, np.ndarray] = field(default_factory=dict)
    positions: np.ndarray | None = None  # (n_frames, n_particles, 3), optional
    wall_clock_s: float = 0.0

    @property
    def n_frames(self) -> int:
        if self.per_frame:
            return len(next(iter(self.per_frame.values())))
        return 0 if self.positions is None else self.positions.shape[0]
