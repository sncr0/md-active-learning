"""Latin-hypercube / maximin space-filling — the honest baseline (§3.5c).

Non-adaptive: it looks only at WHERE points already are, never at their values
or the surrogate. Implemented as a sequential farthest-point design over an LHS
candidate pool, so it fills the domain's largest holes at each round regardless
of batch schedule. This is the design active learning must beat at equal count
(not uniform random, which is a straw man).
"""

from __future__ import annotations

import numpy as np

from mdal.decision._common import candidate_points, maximin_select
from mdal.domain import Domain
from mdal.surrogate.base import Surrogate


class LatinHypercube:
    name = "latin_hypercube"

    def __init__(self, seed: int = 0, n_candidates: int = 2000):
        self.seed = seed
        self.n_candidates = n_candidates

    def propose(self, surrogate: Surrogate, domain: Domain, observed_X, batch: int = 1) -> np.ndarray:
        n_obs = 0 if observed_X is None else len(observed_X)
        cand = candidate_points(domain, self.n_candidates, self.seed + n_obs)
        return maximin_select(cand, domain, batch, observed_X)
