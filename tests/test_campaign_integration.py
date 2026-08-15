"""End-to-end campaign: propose -> simulate (pool) -> estimate -> store -> refit.

Skips without the `md` extra. Uses tiny runs so it stays fast.
"""

import numpy as np
import pytest

pytest.importorskip("openmm")

from mdal.campaign import CampaignSpec, run_campaign
from mdal.domain import Domain
from mdal.store import DuckDBStore

RUN_DEFAULTS = dict(n_particles=864, n_steps=1500, equil_steps=800, sample_interval=100)


def _spec(strategy):
    return CampaignSpec(
        name=strategy, observable="pressure", domain=Domain(), strategy=strategy,
        n_initial=4, n_total=8, batch=4, seed=0, run_defaults=RUN_DEFAULTS,
    )


def test_campaign_runs_end_to_end(tmp_path):
    store = DuckDBStore(str(tmp_path / "c.duckdb"))
    run_campaign(_spec("epistemic"), store, max_workers=4)

    X, y, noise_var = store.observations_for("pressure")
    assert len(X) == 8
    assert np.all(np.isfinite(y)) and np.all(noise_var > 0)
    assert np.all(Domain().contains(X))
    # both observables are estimated and stored from each trajectory
    assert len(store.observations_for("energy")[0]) == 8
    store.close()


def test_campaign_is_resumable(tmp_path):
    db = str(tmp_path / "c.duckdb")
    run_campaign(_spec("alc_imse"), DuckDBStore(db), max_workers=4)
    # reopening and re-running recomputes nothing and does not exceed the budget
    store = DuckDBStore(db)
    run_campaign(_spec("alc_imse"), store, max_workers=4)
    assert len(store.observations_for("pressure")[0]) == 8
    store.close()
