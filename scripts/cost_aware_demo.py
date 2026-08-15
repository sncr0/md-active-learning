"""Stage 2 demo: cost-aware AL vs fixed-length AL under an equal wall-clock budget.

Both run the same budget-driven loop; the only difference is whether the
production length is chosen adaptively (cost_aware) or held fixed (fixed_len).
Produces: error-vs-cumulative-cost, and the cost-aware length-allocation map.

Usage:  python scripts/cost_aware_demo.py [budget] [seed] [fixed_len]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from mdal.analysis._common import integrated_abs_error
from mdal.campaign import run_cost_aware_campaign
from mdal.cost import LinearCost
from mdal.decision.cost_aware import CostAwareALC, FixedLengthALC
from mdal.domain import Domain
from mdal.store import DuckDBStore


def main(budget: float = 340000, seed: int = 0, fixed_len: int = 5000):
    domain = Domain()
    cost = LinearCost(fixed=3500.0, per_step=1.0)
    run_defaults = dict(
        n_particles=864, equil_steps=3000, sample_interval=100, timestep=0.005,
        thermostat="langevin", friction=1.0, cutoff=2.5, tail_correction=True, seed=seed,
    )
    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    acqs = {
        "fixed_len": FixedLengthALC(length=fixed_len, cost=cost, seed=seed),
        "cost_aware": CostAwareALC(cost=cost, l_min=1500, l_max=25000, seed=seed),
    }
    stores = {}
    for name, acq in acqs.items():
        store = DuckDBStore(f"data/cost_{name}_s{seed}.duckdb")
        t = time.perf_counter()
        run_cost_aware_campaign(
            domain, store, budget, run_defaults=run_defaults, acquisition=acq,
            cost=cost, n_initial=8, init_len=4000, batch=8, seed=seed,
            log_fn=lambda m: print("  ", m, flush=True),
        )
        print(f"{name}: {time.perf_counter()-t:.0f}s", flush=True)
        stores[name] = store

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) error vs cumulative cost
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for name, store in stores.items():
        L = store.lengths_for("pressure")
        cc = np.cumsum(cost(L))
        prefixes = list(range(8, len(L) + 1, 2))
        xs = [cc[n - 1] for n in prefixes]
        ys = [integrated_abs_error(store, domain, "pressure", n, 60) for n in prefixes]
        ax.plot(xs, ys, marker="o", ms=4, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cumulative cost (step-equivalents)")
    ax.set_ylabel("integrated |surrogate - KN|  (pressure)")
    ax.set_title("Cost-aware vs. fixed-length  (equal budget)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/cost_aware_curve.png", dpi=130)
    plt.close(fig)

    # (2) length-allocation map
    ca = stores["cost_aware"]
    X = ca.observations_for("pressure")[0]
    L = ca.lengths_for("pressure")
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    sc = ax.scatter(X[:, 0], X[:, 1], c=L, s=48, cmap="viridis", edgecolors="k", linewidths=0.4)
    fig.colorbar(sc, ax=ax, label="chosen production length (steps)")
    ax.set_xlabel("T*")
    ax.set_ylabel("rho*")
    ax.set_title("Cost-aware length allocation")
    ax.set_xlim(1.35, 3.0)
    ax.set_ylim(0.05, 0.9)
    fig.tight_layout()
    fig.savefig("results/cost_aware_lengthmap.png", dpi=130)
    plt.close(fig)

    print("\n=== summary ===")
    for name, store in stores.items():
        L = store.lengths_for("pressure")
        err = integrated_abs_error(store, domain, "pressure", None, 80)
        print(f"{name:11s}: {len(L):3d} runs | total cost {cost(L).sum():.0f} | "
              f"length mean {L.mean():.0f} range [{int(L.min())},{int(L.max())}] | final err {err:.4f}")

    Xc, Lc = ca.observations_for("pressure")[0], ca.lengths_for("pressure")
    lowrho = Lc[Xc[:, 1] < 0.35]
    highrho = Lc[Xc[:, 1] > 0.55]
    print(f"\ncost-aware length by region:  low rho* (<0.35) mean={lowrho.mean():.0f}   "
          f"high rho* (>0.55) mean={highrho.mean():.0f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if len(a) >= 1:
        kw["budget"] = float(a[0])
    if len(a) >= 2:
        kw["seed"] = int(a[1])
    if len(a) >= 3:
        kw["fixed_len"] = int(a[2])
    main(**kw)
