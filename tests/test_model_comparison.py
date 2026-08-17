"""Classical-ML baselines: same predict/epistemic_std shape as the GP, so
scoring/plotting code doesn't need a model-family special case.
"""

import numpy as np

from mdal.analysis.model_comparison import SklearnBaseline, model_families
from mdal.reference import pressure
from mdal.surrogate import HeteroscedasticGP


def test_all_families_fit_and_predict_reasonably():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(1.35, 3.0, 200), rng.uniform(0.05, 0.9, 200)])
    y = pressure(X[:, 0], X[:, 1]) + rng.normal(0, 0.02, 200)
    noise_var = np.full(200, 0.02**2)

    Xg = np.column_stack([rng.uniform(1.4, 2.9, 300), rng.uniform(0.1, 0.85, 300)])
    truth = pressure(Xg[:, 0], Xg[:, 1])

    for name, model in model_families().items():
        if not isinstance(model, (HeteroscedasticGP, SklearnBaseline)):
            continue  # deep-learning families train at full strength — covered
            # separately (fast hyperparameters) by test_dl_surrogates.py's @slow tests
        model.fit(X, y, noise_var)
        mean, total_std = model.predict(Xg)
        assert mean.shape == (300,), name
        assert total_std.shape == (300,), name
        # pls is linear (2 components) and can't capture the strongly nonlinear
        # LJ pressure surface — that's real signal, not a bug, so it just gets a
        # sanity bound (finite, right order of magnitude) instead of an accuracy one
        bound = 3.0 if name == "pls" else 0.3
        assert np.median(np.abs(mean - truth)) < bound, name


def test_sklearn_baselines_report_zero_epistemic_uncertainty():
    # only the plain-sklearn wrappers fake zero — the GP, and (if the `dl`
    # extra is installed) the deep-learning families, have a real estimate
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (30, 2))
    y = rng.normal(size=30)
    for name, model in model_families().items():
        if not isinstance(model, SklearnBaseline):
            continue
        model.fit(X, y, np.full(30, 0.01))
        std = model.epistemic_std(rng.uniform(0, 1, (5, 2)))
        assert np.all(std == 0.0), name
