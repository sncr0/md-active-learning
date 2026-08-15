"""Naive maximum-posterior-variance acquisition — the deliberate straw man (§3.6.4).

Queries the point of highest TOTAL predictive std (epistemic + observation
noise). With heteroscedastic noise this fixates on the noisiest region and
resamples it, because the irreducible noise term dominates and never shrinks.
Watching it fail on the same domain the corrected variants succeed on is the
single most instructive result in Stage 1, so it is a first-class strategy.
"""

from __future__ import annotations

import numpy as np

from mdal.decision._common import candidate_points, select_top
from mdal.domain import Domain
from mdal.surrogate.base import Surrogate


class MaxVariance:
    name = "max_variance"

    def __init__(self, seed: int = 0, n_candidates: int = 1500):
        self.seed = seed
        self.n_candidates = n_candidates

    def propose(self, surrogate: Surrogate, domain: Domain, observed_X, batch: int = 1) -> np.ndarray:
        n_obs = 0 if observed_X is None else len(observed_X)
        cand = candidate_points(domain, self.n_candidates, self.seed + n_obs)
        _, total_std = surrogate.predict(cand)
        return select_top(cand, total_std, domain, batch)
