"""Shared analysis helpers: fit a surrogate on (a prefix of) a store's
observations and score it against the Kolafa-Nezbeda ground truth."""

from __future__ import annotations

import numpy as np

from mdal.domain import Domain
from mdal.reference import energy, pressure
from mdal.store.base import Store
from mdal.surrogate import HeteroscedasticGP

_REFERENCE = {"pressure": pressure, "energy": energy}


def reference_surface(observable: str, X: np.ndarray) -> np.ndarray:
    return _REFERENCE[observable](X[:, 0], X[:, 1])


def fit_on_prefix(store: Store, observable: str, n: int | None = None):
    """Fit a surrogate on the first `n` stored observations (all, if n is None).

    The store returns observations in acquisition order, so a prefix reproduces
    the campaign's knowledge state after that many simulations.
    """
    X, y, noise_var = store.observations_for(observable)
    if n is not None:
        X, y, noise_var = X[:n], y[:n], noise_var[:n]
    return HeteroscedasticGP().fit(X, y, noise_var), X


def integrated_abs_error(
    store: Store, domain: Domain, observable: str, n: int | None = None, grid_n: int = 50
) -> float:
    """Mean |surrogate - reference| over a dense grid — the campaign's true error."""
    gp, _ = fit_on_prefix(store, observable, n)
    grid = domain.grid(grid_n)
    pred, _ = gp.predict(grid)
    return float(np.mean(np.abs(pred - reference_surface(observable, grid))))
