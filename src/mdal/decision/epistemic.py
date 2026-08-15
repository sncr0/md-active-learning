"""Epistemic-only maximum-variance acquisition (§3.6.4, fix 1).

Queries the point of highest EPISTEMIC std, excluding the observation-noise
term. Unlike naive max-variance it does not fixate on the noisiest region,
because irreducible noise is removed from the criterion — it goes where the
latent function itself is least known.
"""

from __future__ import annotations

import numpy as np

from mdal.decision._common import candidate_points, select_top
from mdal.domain import Domain
from mdal.surrogate.base import Surrogate


class EpistemicVariance:
    name = "epistemic"

    def __init__(self, seed: int = 0, n_candidates: int = 1500):
        self.seed = seed
        self.n_candidates = n_candidates

    def propose(self, surrogate: Surrogate, domain: Domain, observed_X, batch: int = 1) -> np.ndarray:
        n_obs = 0 if observed_X is None else len(observed_X)
        cand = candidate_points(domain, self.n_candidates, self.seed + n_obs)
        scores = surrogate.epistemic_std(cand)
        return select_top(cand, scores, domain, batch)
