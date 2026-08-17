"""Cost-aware active-learning loop (Stage 2, §4).

Same infrastructure as Stage 1, two differences: (1) the acquisition proposes a
production LENGTH per point as well as a location, and (2) the campaign runs to
a wall-clock BUDGET (sum of run costs) rather than a fixed simulation count.
Equilibration length is held fixed in run_defaults — only production varies.
"""

from __future__ import annotations

import numpy as np

from mdal import tracking
from mdal.cost import LinearCost
from mdal.decision.cost_aware import CostAwareALC, NoiseCoefficientModel
from mdal.decision.latin_hypercube import LatinHypercube
from mdal.executor import run_batch
from mdal.config import RunConfig
from mdal.store.base import Store
from mdal.surrogate import HeteroscedasticGP


def _config_at(run_defaults, point, n_steps):
    return RunConfig(
        temperature=float(point[0]), density=float(point[1]),
        n_steps=int(n_steps), **run_defaults,
    )


def run_cost_aware_campaign(
    domain, store: Store, budget: float, observable: str = "pressure",
    run_defaults: dict | None = None, acquisition=None, cost: LinearCost | None = None,
    n_initial: int = 8, init_len: int = 4000, batch: int = 8, seed: int = 0,
    max_workers: int = 8, log_fn=None,
) -> Store:
    cost = cost or LinearCost()
    acquisition = acquisition or CostAwareALC(cost=cost, seed=seed)
    surrogate = HeteroscedasticGP()
    run_defaults = run_defaults or dict(
        n_particles=864, equil_steps=3000, sample_interval=100, timestep=0.005,
        thermostat="langevin", friction=1.0, cutoff=2.5, tail_correction=True, seed=seed,
    )

    def observed_X():
        return store.observations_for(observable)[0]

    def spent():
        return float(cost(store.lengths_for(observable)).sum())

    def log(msg):
        if log_fn:
            log_fn(msg)

    # initial design at a fixed length
    if len(observed_X()) < n_initial:
        pts = LatinHypercube(seed=seed).propose(
            surrogate, domain, observed_X(), batch=n_initial - len(observed_X())
        )
        run_batch([_config_at(run_defaults, p, init_len) for p in pts], store, max_workers)
        log(f"initial: {len(observed_X())} runs, spent {spent():.0f}/{budget:.0f}")

    # budget-driven rounds — each surrogate refit logged the same way as the
    # fixed-length loop (mdal.tracking; no-op if not set up)
    campaign_id = getattr(store, "campaign_id", f"cost-aware-{observable}-s{seed}")
    with tracking.campaign_run(
        campaign_id, strategy=type(acquisition).__name__, observable=observable,
        params={"budget": budget, "n_initial": n_initial, "init_len": init_len,
                "batch": batch, "seed": seed},
    ):
        round_idx = 0
        while spent() < budget:
            X, y, noise_var = store.observations_for(observable)
            surrogate.fit(X, y, noise_var)
            round_idx += 1
            tracking.log_round(round_idx, surrogate, X, observable, domain, campaign_id)
            noise_model = NoiseCoefficientModel(X, noise_var, store.lengths_for(observable))
            pts, lengths = acquisition.propose(surrogate, noise_model, domain, X, batch=batch)
            new = run_batch([_config_at(run_defaults, p, ln) for p, ln in zip(pts, lengths)],
                            store, max_workers)
            log(f"runs={len(observed_X())} spent={spent():.0f}/{budget:.0f} "
                f"lengths={sorted(int(x) for x in lengths)}")
            if not new:
                break

    return store
