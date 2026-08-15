"""Acquisition functions: reproduce the trap and its fixes (§3.6.4).

Scenario (1-D): an INTERIOR densely-sampled noisy cluster (so its epistemic is
low but its total variance is high), pinned by low-noise anchors, and a wide
empty gap flanked by low-noise anchors (so it has high epistemic and LOW
noise). This cleanly separates the three behaviours:

  * naive max-variance chases the noisy cluster (irreducible noise),
  * epistemic escapes to the gap (where the function is unknown),
  * ALC/IMSE escapes to the gap AND prefers its low-noise interior.
"""

import numpy as np
import pytest

from mdal.decision import (
    EpistemicVariance,
    IntegratedVarianceReduction,
    LatinHypercube,
    MaxVariance,
)
from mdal.domain import Domain
from mdal.surrogate import HeteroscedasticGP

DOMAIN = Domain(names=("x",), lows=(0.0,), highs=(1.0,))


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(0)
    f = lambda x: 0.3 * np.sin(3 * np.pi * x)
    left = np.array([0.02, 0.06])                 # pin the left boundary
    noisy = rng.uniform(0.12, 0.38, 60)           # interior noisy dense cluster
    mid = np.array([0.44, 0.49])                  # pin the cluster/gap boundary
    right = np.array([0.85, 0.90, 0.95, 1.00])    # gap = (0.49, 0.85)
    X = np.concatenate([left, noisy, mid, right])[:, None]
    sd = np.where((X[:, 0] > 0.1) & (X[:, 0] < 0.4), 0.5, 0.02)
    y = f(X[:, 0]) + rng.normal(0, sd)
    gp = HeteroscedasticGP(n_restarts=6).fit(X, y, sd**2)
    return gp, X


def _propose1(acq, gp, X):
    return acq.propose(gp, DOMAIN, X, batch=1)[0, 0]


def test_max_variance_is_trapped_by_noise(scene):
    gp, X = scene
    x = _propose1(MaxVariance(1), gp, X)
    assert 0.1 < x < 0.4  # the noisy cluster
    assert np.sqrt(gp.predict_noise_var([[x]])[0]) > 0.3  # chose a high-noise site


def test_epistemic_escapes_to_the_gap(scene):
    gp, X = scene
    x_epi = _propose1(EpistemicVariance(1), gp, X)
    x_mv = _propose1(MaxVariance(1), gp, X)
    assert x_epi > 0.49  # in the empty gap, not the noisy cluster
    # it targets genuinely higher latent uncertainty than the noise-trapped choice
    assert gp.epistemic_std([[x_epi]])[0] > gp.epistemic_std([[x_mv]])[0]


def test_alc_avoids_noise_and_targets_the_gap(scene):
    gp, X = scene
    x_alc = _propose1(IntegratedVarianceReduction(1), gp, X)
    x_mv = _propose1(MaxVariance(1), gp, X)
    assert x_alc > 0.49  # in the gap
    noise_alc = np.sqrt(gp.predict_noise_var([[x_alc]])[0])
    noise_mv = np.sqrt(gp.predict_noise_var([[x_mv]])[0])
    assert noise_alc < 0.1            # picks a low-noise, information-rich site
    assert noise_alc < 0.3 * noise_mv  # the essence of the trap: far less noise than naive


def test_latin_hypercube_fills_space_non_adaptively(scene):
    gp, X = scene
    pts = LatinHypercube(0).propose(gp, DOMAIN, X, batch=8)
    assert pts.shape == (8, 1)
    assert np.all((pts >= 0.0) & (pts <= 1.0))
    assert np.any((pts[:, 0] > 0.49) & (pts[:, 0] < 0.85))  # fills the empty gap


@pytest.mark.parametrize("acq", [MaxVariance(1), EpistemicVariance(1), IntegratedVarianceReduction(1)])
def test_batch_shape_and_distinct(scene, acq):
    gp, X = scene
    pts = acq.propose(gp, DOMAIN, X, batch=5)
    assert pts.shape == (5, 1)
    assert np.all((pts >= 0.0) & (pts <= 1.0))
    assert len(np.unique(np.round(pts[:, 0], 4))) == 5  # no collapsed duplicates


def test_works_in_two_dimensions():
    rng = np.random.default_rng(1)
    dom = Domain(names=("T", "rho"), lows=(1.35, 0.05), highs=(3.0, 0.9))
    X = np.column_stack([rng.uniform(1.35, 3.0, 25), rng.uniform(0.05, 0.9, 25)])
    gp = HeteroscedasticGP(n_restarts=2).fit(X, rng.normal(size=25), np.full(25, 0.01))
    for acq in (MaxVariance(1), EpistemicVariance(1), IntegratedVarianceReduction(1), LatinHypercube(1)):
        pts = acq.propose(gp, dom, X, batch=3)
        assert pts.shape == (3, 2)
        assert np.all(dom.contains(pts))
