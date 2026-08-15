"""Heteroscedastic GP surrogate: the epistemic/aleatoric split must be real.

The decomposition is the whole reason this class exists (§3.6.3-4): with
per-point noise in `alpha`, the latent (epistemic) posterior must stay small
where data is dense EVEN IF that data is very noisy, while the total predictive
variance tracks the per-point noise. If that fails, the acquisition trap can't
be demonstrated or fixed.
"""

import numpy as np
import pytest

from mdal.reference import pressure
from mdal.surrogate import HeteroscedasticGP


def test_epistemic_excludes_observation_noise():
    # flat truth y=0; left half heavily noisy, right half quiet; both dense
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 40)[:, None]
    sd = np.where(x[:, 0] < 0.5, 0.5, 0.02)
    y = rng.normal(0, sd)
    gp = HeteroscedasticGP(n_restarts=2).fit(x, y, sd**2)

    epi_noisy = gp.epistemic_std([[0.25]])[0]
    _, tot_noisy = gp.predict([[0.25]])
    tot_noisy = tot_noisy[0]

    # dense data -> epistemic tiny; but total is dominated by irreducible noise
    assert epi_noisy < 0.1
    assert tot_noisy > 0.3
    assert epi_noisy < 0.2 * tot_noisy
    # the interpolated aleatoric noise recovers the ~0.5 sd there
    assert gp.predict_noise_var([[0.25]])[0] == pytest.approx(0.5**2, rel=0.4)


def test_total_variance_tracks_per_point_noise():
    # two equally-dense clusters, different noise -> equal epistemic, unequal total
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 40)[:, None]
    sd = np.where(x[:, 0] < 0.5, 0.5, 0.02)
    y = rng.normal(0, sd)
    gp = HeteroscedasticGP(n_restarts=2).fit(x, y, sd**2)

    _, tot_noisy = gp.predict([[0.25]])
    _, tot_quiet = gp.predict([[0.75]])
    assert tot_noisy[0] > 5 * tot_quiet[0]  # total reflects the 0.5 vs 0.02 noise
    # epistemic is small in BOTH (data is dense in both) -> the noise is not epistemic
    assert gp.epistemic_std([[0.25]])[0] < 0.1
    assert gp.epistemic_std([[0.75]])[0] < 0.1


def test_epistemic_grows_in_gaps():
    x = np.array([[0.05], [0.1], [0.15], [0.85], [0.9], [0.95]])
    y = np.sin(6 * x[:, 0])
    gp = HeteroscedasticGP(n_restarts=2).fit(x, y, np.full(6, 1e-4))
    at_data = gp.epistemic_std([[0.1]])[0]
    in_gap = gp.epistemic_std([[0.5]])[0]
    assert in_gap > 5 * at_data


def test_recovers_reference_surface():
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.uniform(1.35, 3.0, 60), rng.uniform(0.05, 0.9, 60)])
    y = pressure(X[:, 0], X[:, 1]) + rng.normal(0, 0.02, 60)
    gp = HeteroscedasticGP(n_restarts=4).fit(X, y, np.full(60, 0.02**2))

    Xg = np.column_stack([rng.uniform(1.4, 2.9, 400), rng.uniform(0.1, 0.85, 400)])
    pred, _ = gp.predict(Xg)
    truth = pressure(Xg[:, 0], Xg[:, 1])
    assert np.median(np.abs(pred - truth)) < 0.02
    assert np.max(np.abs(pred - truth)) < 0.15


def test_shapes_single_and_batch():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, (20, 2))
    gp = HeteroscedasticGP(n_restarts=1).fit(X, rng.normal(size=20), np.full(20, 0.01))
    m, s = gp.predict([[0.5, 0.5]])
    assert m.shape == (1,) and s.shape == (1,)
    m, s = gp.predict(rng.uniform(0, 1, (7, 2)))
    assert m.shape == (7,) and s.shape == (7,)
    assert np.all(s > 0)


def test_total_never_below_epistemic():
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, (25, 2))
    gp = HeteroscedasticGP(n_restarts=1).fit(X, rng.normal(size=25), rng.uniform(1e-3, 0.2, 25))
    Xg = rng.uniform(0, 1, (50, 2))
    _, tot = gp.predict(Xg)
    epi = gp.epistemic_std(Xg)
    assert np.all(tot >= epi - 1e-9)
