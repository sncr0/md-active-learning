"""Cost model for cost-aware active learning (Stage 2, §4).

Wall-clock of a simulation splits into a FIXED part (setup + equilibration,
paid regardless of production length) and a per-production-step part. Working in
step-equivalent units keeps the demonstration machine-independent; calibrate
`fixed`/`per_step` from the oracle benchmark to make the units real seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearCost:
    fixed: float = 3500.0     # step-equivalent setup + equilibration cost, paid every run
    per_step: float = 1.0     # cost per production step

    def __call__(self, n_steps):
        return self.fixed + self.per_step * np.asarray(n_steps, dtype=float)
