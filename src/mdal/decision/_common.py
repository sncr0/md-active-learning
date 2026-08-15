"""Shared machinery for acquisition functions: candidate generation and batch
selection over the domain. Kept here so the four strategies stay one idea each.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats.qmc import LatinHypercube

from mdal.domain import Domain


def candidate_points(domain: Domain, n: int, seed: int) -> np.ndarray:
    """A space-filling (LHS) pool of candidate query points over the domain."""
    u = LatinHypercube(d=domain.ndim, seed=int(seed)).random(n)
    lo = np.asarray(domain.lows, dtype=float)
    hi = np.asarray(domain.highs, dtype=float)
    return lo + u * (hi - lo)


def _normalise(X, domain: Domain) -> np.ndarray:
    lo = np.asarray(domain.lows, dtype=float)
    hi = np.asarray(domain.highs, dtype=float)
    return (np.atleast_2d(np.asarray(X, dtype=float)) - lo) / (hi - lo)


def maximin_select(candidates, domain, batch, observed_X) -> np.ndarray:
    """Greedy farthest-point (maximin) selection — fills the largest holes in
    the union of observed and already-selected points. Non-adaptive."""
    C = _normalise(candidates, domain)
    if observed_X is not None and len(observed_X) > 0:
        mind = cdist(C, _normalise(observed_X, domain)).min(axis=1)
    else:
        mind = np.full(len(C), np.inf)
    idx = []
    for _ in range(min(batch, len(C))):
        i = int(np.argmax(mind))
        idx.append(i)
        mind = np.minimum(mind, np.linalg.norm(C - C[i], axis=1))
    return np.atleast_2d(candidates)[idx]


def select_top_indices(candidates, scores, domain, batch, min_sep=None) -> list[int]:
    """Indices of the top-`batch` candidates by score, keeping a minimum
    separation so a batch doesn't collapse onto one peak. (Deliberately does NOT
    avoid observed points: re-proposing a noisy site is exactly the max-variance
    trap we want to show.)"""
    C = _normalise(candidates, domain)
    if min_sep is None:
        min_sep = max(0.02, 0.4 / np.sqrt(batch + 1))
    order = np.argsort(scores)[::-1]
    chosen: list[int] = []
    chosenC: list[np.ndarray] = []
    for i in order:
        if len(chosen) >= batch:
            break
        c = C[i]
        if chosenC and min(np.linalg.norm(np.asarray(chosenC) - c, axis=1)) < min_sep:
            continue
        chosen.append(int(i))
        chosenC.append(c)
    for i in order:  # relax spacing if it left us short
        if len(chosen) >= batch:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    return chosen[:batch]


def select_top(candidates, scores, domain, batch, min_sep=None) -> np.ndarray:
    return np.atleast_2d(candidates)[select_top_indices(candidates, scores, domain, batch, min_sep)]
