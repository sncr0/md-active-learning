"""Acquisition comparison: integrated abs. error vs #simulations, per strategy (§3.5c).

The headline result. The expected finding is that naive max-variance loses,
and that active learning reaches a target error in fewer simulations than the
space-filling baseline.
"""

from __future__ import annotations

from mdal.analysis._common import integrated_abs_error
from mdal.domain import Domain
from mdal.store.base import Store


def learning_curve(store: Store, domain: Domain, observable: str, ns, grid_n: int = 50):
    """[(n, integrated_abs_error after n sims)] for each n in ns."""
    return [(n, integrated_abs_error(store, domain, observable, n, grid_n)) for n in ns]


def acquisition_comparison(
    stores: dict[str, Store], domain: Domain, observable: str, out_path: str,
    ns=None, grid_n: int = 50,
):
    """Overlay each strategy's error-vs-budget curve; return the curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves: dict[str, list] = {}
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for name, store in stores.items():
        total = len(store.observations_for(observable)[0])
        grid_ns = [n for n in (ns or range(4, total + 1, 4)) if 2 <= n <= total]
        curve = learning_curve(store, domain, observable, grid_ns, grid_n)
        curves[name] = curve
        xs, ys = zip(*curve)
        ax.plot(xs, ys, marker="o", markersize=4, label=name)

    ax.set_xlabel("number of simulations")
    ax.set_ylabel(f"integrated |surrogate - reference|  ({observable})")
    ax.set_yscale("log")
    ax.set_title("Active learning vs. space-filling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return curves
