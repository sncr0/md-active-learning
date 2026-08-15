"""The active-learning loop (§1 three-layer architecture, assembled).

Each layer is reached only through its narrow interface: the loop reads
(X, y, noise_var) from the store, fits the surrogate, asks the acquisition for
points, builds RunConfigs, and hands them to the executor — which runs the
oracle in a pool and writes results back. The loop never sees a trajectory.

Resumable by construction: progress is measured by counting stored observations,
and the executor skips any run already present, so a killed session restarts
from whatever the store already holds.
"""

from __future__ import annotations

from mdal.campaign.spec import CampaignSpec
from mdal.config import RunConfig
from mdal.decision import REGISTRY
from mdal.decision.latin_hypercube import LatinHypercube
from mdal.executor import run_batch
from mdal.store.base import Store
from mdal.surrogate import HeteroscedasticGP


def _config_at(spec: CampaignSpec, point) -> RunConfig:
    """Build a RunConfig for a state point from the campaign's run defaults."""
    return RunConfig(temperature=float(point[0]), density=float(point[1]), **spec.run_defaults)


def run_campaign(spec: CampaignSpec, store: Store, max_workers: int = 8, log_fn=None) -> Store:
    """Drive the campaign to its budget, resuming from whatever is in `store`."""
    acquisition = REGISTRY[spec.strategy](seed=spec.seed)
    surrogate = HeteroscedasticGP()

    def observed_X():
        return store.observations_for(spec.observable)[0]

    def log(msg):
        if log_fn:
            log_fn(msg)

    # 1. initial space-filling design (LHS), whatever is still missing
    n = len(observed_X())
    if n < spec.n_initial:
        pts = LatinHypercube(seed=spec.seed).propose(
            surrogate, spec.domain, observed_X(), batch=spec.n_initial - n
        )
        run_batch([_config_at(spec, p) for p in pts], store, max_workers)
        log(f"{spec.name}: initial design -> {len(observed_X())} runs")

    # 2. active-learning rounds
    while True:
        X, y, noise_var = store.observations_for(spec.observable)
        if len(X) >= spec.n_total:
            break
        surrogate.fit(X, y, noise_var)
        batch = min(spec.batch, spec.n_total - len(X))
        pts = acquisition.propose(surrogate, spec.domain, X, batch=batch)
        new = run_batch([_config_at(spec, p) for p in pts], store, max_workers)
        log(f"{spec.name}: {len(observed_X())}/{spec.n_total} runs")
        if not new:  # safety: no progress (every proposal already computed)
            log(f"{spec.name}: no new runs this round; stopping")
            break

    return store
