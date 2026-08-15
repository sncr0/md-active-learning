"""Cost-aware acquisition (Stage 2, §4): choose (state point, production length).

The observation noise of a length-L run is sigma^2(x, L) = K(x) / L, where the
noise coefficient K(x) = sigma^2 * L is length-invariant (variance of a mean
scales as 1/effective-samples ~ 1/L). Given a linear cost c0 + c1*L, the
expected integrated-variance reduction per unit cost is maximised at a
CLOSED-FORM optimal length

    L*(x) = sqrt( K(x) * c0 / ( epistemic_var(x) * c1 ) ),

clipped to [l_min, l_max]. Longer runs where the noise K is large; SHORTER runs
where the function is already very uncertain (a cheap noisy sample already buys
most of the available reduction there). The policy then picks the state points
whose (reduction / cost) at their own L* is highest.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from mdal.cost import LinearCost
from mdal.decision._common import candidate_points, select_top_indices
from mdal.decision.alc_imse import IntegratedVarianceReduction


class NoiseCoefficientModel:
    """k-NN interpolator of K(x) = sigma^2 * n_steps, giving sigma^2 at any length."""

    def __init__(self, X, noise_var, n_steps, k: int = 6):
        self._X = np.atleast_2d(np.asarray(X, dtype=float))
        self._K = np.asarray(noise_var, dtype=float) * np.asarray(n_steps, dtype=float)
        self._mean = self._X.mean(axis=0)
        self._std = self._X.std(axis=0)
        self._std[self._std == 0] = 1.0
        self._Xs = (self._X - self._mean) / self._std
        self._k = min(k, self._Xs.shape[0])

    def coeff(self, Xq) -> np.ndarray:
        Xs = (np.atleast_2d(np.asarray(Xq, dtype=float)) - self._mean) / self._std
        D = cdist(Xs, self._Xs)
        idx = np.argpartition(D, self._k - 1, axis=1)[:, : self._k]
        dk = np.take_along_axis(D, idx, axis=1)
        w = 1.0 / (dk + 1e-9)
        return (w * self._K[idx]).sum(axis=1) / w.sum(axis=1)

    def noise_var_at(self, Xq, n_steps) -> np.ndarray:
        return self.coeff(Xq) / np.asarray(n_steps, dtype=float)


def optimal_length(noise_coeff, epistemic_var, cost: LinearCost, l_min, l_max) -> np.ndarray:
    """Closed-form L*(x) = sqrt(K c0 / (a c1)), clipped to [l_min, l_max]."""
    a = np.maximum(np.asarray(epistemic_var, dtype=float), 1e-12)
    K = np.maximum(np.asarray(noise_coeff, dtype=float), 0.0)
    L = np.sqrt(K * cost.fixed / (a * cost.per_step))
    return np.clip(L, l_min, l_max)


class CostAwareALC:
    """Proposes (points, lengths) maximising variance-reduction per unit cost."""

    name = "cost_aware"

    def __init__(self, cost: LinearCost | None = None, l_min: int = 1500, l_max: int = 25000,
                 seed: int = 0, n_candidates: int = 400, n_reference: int = 400):
        self.cost = cost or LinearCost()
        self.l_min = l_min
        self.l_max = l_max
        self.seed = seed
        self.n_candidates = n_candidates
        self.n_reference = n_reference

    def propose(self, surrogate, noise_model: NoiseCoefficientModel, domain, observed_X, batch: int = 1):
        n = 0 if observed_X is None else len(observed_X)
        s = self.seed + n
        cand = candidate_points(domain, self.n_candidates, s)
        ref = candidate_points(domain, self.n_reference, s + 10007)

        cross = surrogate.posterior_cov(ref, cand)          # (n_ref, n_cand)
        numerator = (cross**2).sum(axis=0)                  # reduction numerator per candidate
        epi = surrogate.epistemic_var(cand)                 # a(x)
        K = noise_model.coeff(cand)                         # K(x)

        L = optimal_length(K, epi, self.cost, self.l_min, self.l_max)
        reduction = numerator / (epi + K / L)               # ΔV at that length
        score = reduction / self.cost(L)                    # per unit cost

        idx = select_top_indices(cand, score, domain, batch)
        return cand[idx], np.round(L[idx]).astype(int)


class FixedLengthALC:
    """Baseline: ALC on WHERE, but every run at a fixed production length.
    Drives the same cost-aware loop for a like-for-like budget comparison."""

    name = "fixed_alc"

    def __init__(self, length: int = 5000, cost: LinearCost | None = None, seed: int = 0):
        self.length = int(length)
        self.cost = cost or LinearCost()
        self._alc = IntegratedVarianceReduction(seed=seed)

    def propose(self, surrogate, noise_model, domain, observed_X, batch: int = 1):
        pts = self._alc.propose(surrogate, domain, observed_X, batch=batch)
        return pts, np.full(len(pts), self.length, dtype=int)
