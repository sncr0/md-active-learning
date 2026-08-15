"""Integrated variance reduction — ALC / IMSE (§3.6.4, fix 2).

Decision-theoretic: score each candidate by how much observing it would reduce
the integrated EPISTEMIC variance over the whole domain, not by how uncertain
the candidate is itself. For a GP the reduction is rank-1 (Active Learning
Cohn):

    dvar(x') = k_post(x', x)^2 / ( k_post(x, x) + sigma_n^2(x) )

so the score of a candidate x is the sum of that reduction over a reference set
approximating the integral. The noise term sigma_n^2(x) in the denominator is
what makes this refuse to chase noisy points: a noisy query buys little global
knowledge. This is the strategy expected to win the acquisition comparison.
"""

from __future__ import annotations

import numpy as np

from mdal.decision._common import candidate_points, select_top
from mdal.domain import Domain
from mdal.surrogate.base import Surrogate


class IntegratedVarianceReduction:
    name = "alc_imse"

    def __init__(self, seed: int = 0, n_candidates: int = 400, n_reference: int = 400):
        self.seed = seed
        self.n_candidates = n_candidates
        self.n_reference = n_reference

    def propose(self, surrogate: Surrogate, domain: Domain, observed_X, batch: int = 1) -> np.ndarray:
        n_obs = 0 if observed_X is None else len(observed_X)
        s = self.seed + n_obs
        cand = candidate_points(domain, self.n_candidates, s)
        ref = candidate_points(domain, self.n_reference, s + 10007)

        # k_post(ref, cand): how each candidate co-varies with every reference point
        cross = surrogate.posterior_cov(ref, cand)  # (n_reference, n_candidates)
        denom = surrogate.epistemic_var(cand) + surrogate.predict_noise_var(cand)  # (n_candidates,)
        scores = (cross**2).sum(axis=0) / denom  # integrated variance reduction per candidate

        return select_top(cand, scores, domain, batch)
