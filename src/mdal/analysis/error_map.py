"""Error map: surrogate - reference EOS over a dense grid, points overlaid (§3.5b).

Shows simultaneously what the surrogate knows (prediction error vs the
Kolafa-Nezbeda ground truth) and where the campaign chose to look.
"""

from __future__ import annotations

from mdal.analysis._common import error_map_figure, fit_on_prefix
from mdal.domain import Domain
from mdal.store.base import Store


def error_map(store: Store, domain: Domain, observable: str, out_path: str, grid_n: int = 80) -> float:
    """Render the signed error heatmap with sample overlay; return the mean |error|."""
    import matplotlib.pyplot as plt

    gp, X = fit_on_prefix(store, observable)
    fig, mean_abs_err = error_map_figure(gp, X, domain, observable, grid_n)
    fig.axes[0].set_title(f"Error map — {observable}  (mean |err| = {mean_abs_err:.4f})")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return mean_abs_err
