"""Direct per-frame estimators: pressure P*, potential energy U* (§3.3).

The simplest estimator kind, and the one that validates the uncertainty
machinery in isolation before the MSD estimator stresses it. The full honest
recipe, assembled from the two building blocks:

    1. detect_equilibration  -> discard the transient
    2. statistical_inefficiency on the tail -> n_eff
    3. value  = mean(tail)
       sigma  = std(tail) / sqrt(n_eff)   # effective-sample standard error

The resulting Observation carries a sigma that is honestly inflated by
sqrt(g) relative to the naive std/sqrt(n) — which is exactly what the
decision layer needs to avoid resampling noise forever.
"""

from __future__ import annotations

import numpy as np

from mdal.estimator.autocorr import effective_sample_size
from mdal.estimator.equilibration import detect_equilibration
from mdal.oracle.trajectory import Trajectory
from mdal.records import Observation


class FrameAverageEstimator:
    """Implements the Estimator protocol for a direct per-frame observable."""

    def __init__(self, observable: str, frame_key: str):
        self.observable = observable
        self.frame_key = frame_key  # key into Trajectory.per_frame

    def estimate(self, trajectory: Trajectory) -> Observation:
        series = np.asarray(trajectory.per_frame[self.frame_key], dtype=float)
        t0 = detect_equilibration(series)
        tail = series[t0:]

        n_eff = effective_sample_size(tail)
        value = float(np.mean(tail))
        sigma = float(np.std(tail, ddof=1) / np.sqrt(n_eff)) if tail.size > 1 else 0.0

        return Observation(
            run_hash=trajectory.run_hash,
            observable=self.observable,
            value=value,
            sigma=sigma,
            n_eff=float(n_eff),
        )
