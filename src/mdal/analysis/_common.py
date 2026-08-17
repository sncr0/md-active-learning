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


def error_map_figure(surrogate, X: np.ndarray, domain: Domain, observable: str, grid_n: int = 60):
    """Signed error heatmap (surrogate - reference) with sample points overlaid.

    Shows simultaneously what the surrogate knows (prediction error vs the
    Kolafa-Nezbeda ground truth) and where the campaign chose to look. Returns
    (Figure, mean |error|) — caller decides what to do with the figure
    (`mdal.analysis.error_map` saves it to disk; `mdal.tracking` logs it to
    MLflow as a round artifact) and must close it when done.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = domain.grid(grid_n)
    pred, _ = surrogate.predict(grid)
    err = pred - reference_surface(observable, grid)

    T = grid[:, 0].reshape(grid_n, grid_n)
    R = grid[:, 1].reshape(grid_n, grid_n)
    E = err.reshape(grid_n, grid_n)
    lim = float(np.abs(err).max()) or 1e-9

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    pc = ax.pcolormesh(T, R, E, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.scatter(X[:, 0], X[:, 1], s=14, c="k", edgecolors="w", linewidths=0.5,
               label=f"{len(X)} simulations")
    fig.colorbar(pc, ax=ax, label=f"surrogate - reference  ({observable})")
    ax.set_xlabel("T*")
    ax.set_ylabel("rho*")
    ax.set_title(f"mean |err| = {np.mean(np.abs(err)):.4f}")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig, float(np.mean(np.abs(err)))
