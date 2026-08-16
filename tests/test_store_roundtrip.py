"""Postgres store machinery: resumability check, roundtrip, idempotent writes,
and the (X, y, noise_var) read side the surrogate consumes.

Uses the `store` fixture from conftest.py — a PostgresStore scoped to a fresh,
throwaway campaign_id per test (skips if Postgres isn't reachable)."""

import numpy as np

from mdal.config import RunConfig
from mdal.records import Observation, RunRecord


def _run(t, rho):
    cfg = RunConfig(temperature=t, density=rho)
    rec = RunRecord(run_hash=cfg.content_hash(), wall_clock_s=1.2, equil_cutoff=15, n_frames=800)
    return cfg, rec


def test_exists_is_the_resumability_check(store):
    cfg, rec = _run(1.5, 0.3)
    assert store.exists(cfg.content_hash()) is False
    store.put_run(cfg, rec)
    assert store.exists(cfg.content_hash()) is True


def test_run_roundtrip(store):
    cfg, rec = _run(1.5, 0.3)
    store.put_run(cfg, rec)
    got = store.get_run(cfg.content_hash())
    assert got.run_hash == rec.run_hash
    assert got.equil_cutoff == 15
    assert got.n_frames == 800
    assert got.status == "complete"


def test_writes_are_idempotent(store):
    cfg, rec = _run(1.5, 0.3)
    store.put_run(cfg, rec)
    store.put_run(cfg, rec)  # re-put after a hypothetical crash-resume: no error, no dup
    obs = Observation(cfg.content_hash(), "pressure", 2.5, 0.05, 42.0)
    store.put_observation(obs)
    store.put_observation(obs)
    X, y, nv = store.observations_for("pressure")
    assert X.shape == (1, 2)


def test_observations_for_returns_arrays_not_trajectories(store):
    for t, rho, p, sig in [(1.5, 0.3, 2.5, 0.05), (2.0, 0.6, 4.1, 0.02)]:
        cfg, rec = _run(t, rho)
        store.put_run(cfg, rec)
        store.put_observation(Observation(cfg.content_hash(), "pressure", p, sig, 40.0))
    X, y, noise_var = store.observations_for("pressure")
    assert X.shape == (2, 2)
    assert y.shape == (2,)
    np.testing.assert_allclose(np.sort(y), [2.5, 4.1])
    # noise_var is sigma**2, per point (heteroscedastic)
    np.testing.assert_allclose(np.sort(noise_var), np.sort([0.05**2, 0.02**2]))


def test_empty_read_is_well_shaped(store):
    X, y, noise_var = store.observations_for("pressure")
    assert X.shape == (0, 2) and y.shape == (0,) and noise_var.shape == (0,)
