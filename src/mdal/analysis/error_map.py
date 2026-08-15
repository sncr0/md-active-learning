"""Error map: surrogate - reference EOS over a dense grid, points overlaid (§3.5b).

Shows simultaneously what the surrogate knows (prediction error vs the
Kolafa-Nezbeda ground truth) and where the campaign chose to look.
"""

from __future__ import annotations

import numpy as np

from mdal.analysis._common import fit_on_prefix, reference_surface
from mdal.domain import Domain
from mdal.store.base import Store


def error_map(store: Store, domain: Domain, observable: str, out_path: str, grid_n: int = 80) -> float:
    """Render the signed error heatmap with sample overlay; return the mean |error|."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gp, X = fit_on_prefix(store, observable)
    grid = domain.grid(grid_n)
    pred, _ = gp.predict(grid)
    err = pred - reference_surface(observable, grid)

    T = grid[:, 0].reshape(grid_n, grid_n)
    R = grid[:, 1].reshape(grid_n, grid_n)
    E = err.reshape(grid_n, grid_n)
    lim = float(np.abs(err).max())

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    pc = ax.pcolormesh(T, R, E, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.scatter(X[:, 0], X[:, 1], s=16, c="k", edgecolors="w", linewidths=0.5,
               label=f"{len(X)} simulations")
    fig.colorbar(pc, ax=ax, label=f"surrogate - reference  ({observable})")
    ax.set_xlabel("T*")
    ax.set_ylabel("rho*")
    ax.set_title(f"Error map — {observable}  (mean |err| = {np.mean(np.abs(err)):.4f})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return float(np.mean(np.abs(err)))
