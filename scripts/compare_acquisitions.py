"""Compare the four acquisition strategies, averaged over seeds (§3.5c).

Each (strategy, seed) is an independent campaign in its own store; the seed sets
BOTH the acquisition/initial-design and the MD realisation, so averaging over
seeds removes single-run luck. Produces a seed-averaged comparison curve with
std bands, plus error maps.

Usage:  python scripts/compare_acquisitions.py [n_total] [n_steps] [seeds_csv]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from mdal.analysis.acquisition_comparison import learning_curve
from mdal.analysis.error_map import error_map
from mdal.campaign import CampaignSpec, run_campaign
from mdal.domain import Domain
from mdal.store import PostgresStore

STRATEGIES = ["latin_hypercube", "max_variance", "epistemic", "alc_imse"]
COLORS = {"latin_hypercube": "C0", "max_variance": "C1", "epistemic": "C2", "alc_imse": "C3"}


def main(n_total: int = 48, n_steps: int = 12000, seeds: str = "0,1,2",
         n_initial: int = 8, batch: int = 8):
    seed_list = [int(s) for s in str(seeds).split(",")]
    domain = Domain()
    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    ns = list(range(n_initial, n_total + 1, batch))
    curves = {s: np.zeros((len(seed_list), len(ns))) for s in STRATEGIES}

    for si, seed in enumerate(seed_list):
        for strat in STRATEGIES:
            run_defaults = dict(
                n_particles=864, n_steps=n_steps, equil_steps=n_steps // 2,
                sample_interval=100, timestep=0.005, thermostat="langevin",
                friction=1.0, cutoff=2.5, tail_correction=True, seed=seed,
            )
            spec = CampaignSpec(
                name=f"{strat}_s{seed}", observable="pressure", domain=domain,
                strategy=strat, n_initial=n_initial, n_total=n_total, batch=batch,
                seed=seed, run_defaults=run_defaults,
            )
            store = PostgresStore(f"cmp_{strat}_s{seed}")
            t = time.perf_counter()
            run_campaign(store=store, spec=spec, log_fn=lambda m: print("  ", m, flush=True))
            lc = learning_curve(store, domain, "pressure", ns, grid_n=60)
            curves[strat][si] = [e for _, e in lc]
            store.close()
            print(f"{strat} seed{seed}: {time.perf_counter()-t:.0f}s", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for strat in STRATEGIES:
        arr = curves[strat]
        m, s = arr.mean(0), arr.std(0)
        ax.plot(ns, m, marker="o", ms=4, color=COLORS[strat], label=strat)
        ax.fill_between(ns, m - s, m + s, color=COLORS[strat], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("number of simulations")
    ax.set_ylabel("integrated |surrogate - KN|  (pressure)")
    ax.set_title(f"Active learning vs. space-filling  (mean of {len(seed_list)} seeds)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/acquisition_comparison_avg.png", dpi=130)
    plt.close(fig)

    for strat in ("alc_imse", "epistemic", "latin_hypercube"):
        st = PostgresStore(f"cmp_{strat}_s{seed_list[0]}")
        error_map(st, domain, "pressure", f"results/errmap_{strat}.png")
        st.close()

    print(f"\nmean +/- std integrated |surrogate-KN| pressure error ({len(seed_list)} seeds):")
    print(f"  {'n':>4} " + " ".join(f"{s[:12]:>18}" for s in STRATEGIES))
    for i, n in enumerate(ns):
        cells = " ".join(f"{curves[s][:, i].mean():>8.4f}+-{curves[s][:, i].std():<7.4f}" for s in STRATEGIES)
        print(f"  {n:>4} {cells}")


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if len(a) >= 1:
        kw["n_total"] = int(a[0])
    if len(a) >= 2:
        kw["n_steps"] = int(a[1])
    if len(a) >= 3:
        kw["seeds"] = a[2]
    main(**kw)
