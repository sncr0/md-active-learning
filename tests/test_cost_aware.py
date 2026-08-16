"""Cost-aware acquisition (Stage 2): the closed-form length policy and the loop."""

import numpy as np
import pytest

from mdal.cost import LinearCost
from mdal.decision.cost_aware import (
    CostAwareALC,
    FixedLengthALC,
    NoiseCoefficientModel,
    optimal_length,
)
from mdal.domain import Domain
from mdal.surrogate import HeteroscedasticGP

COST = LinearCost(fixed=3500.0, per_step=1.0)


def test_optimal_length_scaling():
    # longer where noise K is larger; shorter where the function is more uncertain
    assert optimal_length(4.0, 1.0, COST, 1, 1e9) > optimal_length(1.0, 1.0, COST, 1, 1e9)
    assert optimal_length(1.0, 4.0, COST, 1, 1e9) < optimal_length(1.0, 1.0, COST, 1, 1e9)
    # exact closed form L* = sqrt(K c0 / (a c1))
    assert optimal_length(2.0, 0.5, COST, 0, 1e12) == pytest.approx(np.sqrt(2.0 * 3500 / 0.5))


def test_optimal_length_clips_to_bounds():
    assert optimal_length(1e12, 1e-9, COST, 1000, 20000) == 20000  # would blow up -> clamped
    assert optimal_length(1e-12, 1e9, COST, 1000, 20000) == 1000   # would vanish -> clamped


def test_noise_coefficient_model_recovers_K_and_scales_with_length():
    X = np.array([[1.5, 0.3], [2.5, 0.7]])
    noise_var = np.array([0.04, 0.01])
    n_steps = np.array([5000.0, 5000.0])
    m = NoiseCoefficientModel(X, noise_var, n_steps)
    # at a training point K = sigma^2 * L
    assert m.coeff([[1.5, 0.3]])[0] == pytest.approx(0.04 * 5000, rel=1e-6)
    # predicted noise halves when the run is twice as long
    L = 8000
    assert m.noise_var_at([[1.5, 0.3]], L)[0] == pytest.approx(0.04 * 5000 / L, rel=1e-6)


def test_longer_runs_allocated_to_noisier_states():
    # same length observations, but region A is much noisier than region B
    X = np.array([[1.4, 0.2], [1.45, 0.25], [2.8, 0.8], [2.85, 0.85]])
    noise_var = np.array([0.25, 0.25, 0.0025, 0.0025])
    m = NoiseCoefficientModel(X, noise_var, np.full(4, 5000.0))
    a = 0.1  # comparable epistemic variance
    L_noisy = optimal_length(m.coeff([[1.42, 0.22]]), a, COST, 1000, 40000)[0]
    L_quiet = optimal_length(m.coeff([[2.82, 0.82]]), a, COST, 1000, 40000)[0]
    assert L_noisy > 2 * L_quiet


def test_cost_aware_proposes_valid_varied_lengths():
    rng = np.random.default_rng(0)
    domain = Domain()
    X = np.column_stack([rng.uniform(1.35, 3.0, 30), rng.uniform(0.05, 0.9, 30)])
    y = X[:, 0] * X[:, 1]
    nv = rng.uniform(1e-3, 0.1, 30)
    gp = HeteroscedasticGP(n_restarts=2).fit(X, y, nv)
    m = NoiseCoefficientModel(X, nv, np.full(30, 5000.0))

    acq = CostAwareALC(cost=COST, l_min=1500, l_max=25000, seed=1)
    pts, lengths = acq.propose(gp, m, domain, X, batch=8)
    assert pts.shape == (8, 2)
    assert lengths.shape == (8,)
    assert np.all((lengths >= 1500) & (lengths <= 25000))
    assert len(np.unique(lengths)) > 1  # it adapts the length, not a single value
    assert np.all(domain.contains(pts))


def test_fixed_length_alc_is_constant():
    rng = np.random.default_rng(1)
    domain = Domain()
    X = np.column_stack([rng.uniform(1.35, 3.0, 20), rng.uniform(0.05, 0.9, 20)])
    gp = HeteroscedasticGP(n_restarts=1).fit(X, rng.normal(size=20), np.full(20, 0.01))
    pts, lengths = FixedLengthALC(length=6000).propose(gp, None, domain, X, batch=5)
    assert pts.shape == (5, 2)
    assert np.all(lengths == 6000)


def test_cost_aware_campaign_runs_and_varies_length(campaign_id):
    pytest.importorskip("openmm")
    from mdal.campaign import run_cost_aware_campaign
    from mdal.store import PostgresStore

    domain = Domain()
    run_defaults = dict(n_particles=864, equil_steps=800, sample_interval=100, seed=0)
    cost = LinearCost(fixed=1500.0, per_step=1.0)
    store = PostgresStore(campaign_id)
    run_cost_aware_campaign(
        domain, store, budget=30000, run_defaults=run_defaults,
        acquisition=CostAwareALC(cost=cost, l_min=1000, l_max=3000, seed=0),
        cost=cost, n_initial=4, init_len=1200, batch=4, max_workers=4,
    )
    X = store.observations_for("pressure")[0]
    lengths = store.lengths_for("pressure")
    assert len(X) >= 4
    assert lengths.min() >= 1000 and lengths.max() <= 3000
    assert len(np.unique(lengths)) > 1  # acquired runs differ in length from the init
    store.close()
