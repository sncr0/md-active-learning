"""Deep-learning UQ baselines: same predict/epistemic_std shape as the GP,
but — unlike the sklearn baselines in model_comparison.py — with a genuine,
non-zero epistemic signal that should grow away from the training data.
"""

import numpy as np
import pytest

from mdal.reference import pressure

torch = pytest.importorskip("torch", reason="torch not installed — uv sync --extra dl")

from mdal.analysis.dl_surrogates import DeepEnsemble, EvidentialNet, MCDropoutNet  # noqa: E402

FAMILIES = {
    "deep_ensemble": lambda: DeepEnsemble(n_members=3, epochs=400),
    "mc_dropout": lambda: MCDropoutNet(epochs=600),
    "evidential": lambda: EvidentialNet(epochs=600),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", FAMILIES)
def test_fits_and_predicts_reasonably(name):
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(1.35, 3.0, 150), rng.uniform(0.05, 0.9, 150)])
    y = pressure(X[:, 0], X[:, 1]) + rng.normal(0, 0.02, 150)
    noise_var = np.full(150, 0.02**2)
    model = FAMILIES[name]().fit(X, y, noise_var)

    Xg = np.column_stack([rng.uniform(1.4, 2.9, 100), rng.uniform(0.1, 0.85, 100)])
    truth = pressure(Xg[:, 0], Xg[:, 1])
    mean, total_std = model.predict(Xg)
    assert mean.shape == (100,)
    assert total_std.shape == (100,)
    assert np.all(total_std > 0)
    assert np.median(np.abs(mean - truth)) < 1.0


@pytest.mark.slow
@pytest.mark.parametrize("name", FAMILIES)
def test_epistemic_grows_away_from_data(name):
    rng = np.random.default_rng(1)
    # data only in the left half of the box
    X = np.column_stack([rng.uniform(1.35, 2.0, 100), rng.uniform(0.05, 0.9, 100)])
    y = pressure(X[:, 0], X[:, 1]) + rng.normal(0, 0.02, 100)
    model = FAMILIES[name]().fit(X, y, np.full(100, 0.02**2))

    at_data = model.epistemic_std(np.array([[1.6, 0.5]]))[0]
    far_away = model.epistemic_std(np.array([[2.9, 0.5]]))[0]
    assert far_away > at_data
