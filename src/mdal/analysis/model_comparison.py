"""Classical-ML baselines vs. the GP surrogate, holding the training data fixed.

Answers a narrower question than `acquisition_comparison.py`: given a set of
already-run simulations (an AL campaign's picks, or any other observed set),
how much does the model FAMILY matter? PLS / random forest / gradient
boosting / a small MLP have no principled epistemic-uncertainty decomposition
the way the GP does (§3.6.3) — that's why none of them are wired into
decision/, they're accuracy-only baselines. `SklearnBaseline` still exposes
predict()/epistemic_std() with the GP's shape so `mdal.tracking
.score_vs_reference` and `mdal.analysis.error_map_figure` work unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from mdal.analysis import dl_surrogates
from mdal.store import PostgresStore
from mdal.surrogate import HeteroscedasticGP

# The canonical six real campaigns (see CLAUDE.md). Pooling their already-run
# observations costs nothing — no new oracle calls — and gives every model
# family a few hundred points instead of one campaign's ~50.
REAL_CAMPAIGNS = (
    "alc_imse", "cost_aware", "epistemic", "fixed_len", "latin_hypercube", "max_variance",
)


class SklearnBaseline:
    """Wraps a plain sklearn regressor to look enough like a Surrogate to reuse
    the GP's scoring/plotting code unchanged.

    Manual X/y standardization mirrors `HeteroscedasticGP.fit` so every family
    sees the same conditioning. `epistemic_std` always returns zero — these
    estimators have no reducible-uncertainty estimate, and reporting zero
    (never a fabricated number) is the honest choice.
    """

    def __init__(self, name: str, make_estimator, weighted: bool = False):
        self.name = name
        self._make_estimator = make_estimator
        self._weighted = weighted  # pass 1/noise_var as sample_weight, if the estimator supports it
        self._fitted = False

    def fit(self, X, y, noise_var) -> "SklearnBaseline":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        self._x_mean, self._x_std = X.mean(axis=0), X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0
        self._y_mean = y.mean()
        self._y_std = y.std() or 1.0

        Xs = (X - self._x_mean) / self._x_std
        ys = (y - self._y_mean) / self._y_std

        self._model = self._make_estimator()
        fit_kwargs = {}
        if self._weighted:
            w = 1.0 / np.maximum(np.asarray(noise_var, dtype=float), 1e-10)
            fit_kwargs["sample_weight"] = w / w.mean()
        self._model.fit(Xs, ys, **fit_kwargs)
        self._fitted = True
        return self

    def _standardize(self, X):
        return (np.atleast_2d(np.asarray(X, dtype=float)) - self._x_mean) / self._x_std

    def predict(self, X):
        pred = np.asarray(self._model.predict(self._standardize(X))).ravel()
        mean = pred * self._y_std + self._y_mean
        return mean, np.zeros_like(mean)

    def epistemic_std(self, X) -> np.ndarray:
        return np.zeros(np.atleast_2d(X).shape[0])

    @property
    def fitted_estimator(self):
        return self._model if self._fitted else None

    def diagnostics(self) -> dict:
        return {}


def model_families() -> dict[str, object]:
    """Fresh, unfitted instance of each model family — call again for each fit.

    The sklearn families report epistemic_std=0 (honest — they have no
    principled uncertainty estimate). If the `dl` extra is installed
    (`uv sync --extra dl`), three PyTorch-based families are added that
    DO have a real epistemic/aleatoric split — see dl_surrogates.py.
    """
    families = {
        "gp": HeteroscedasticGP(),
        "pls": SklearnBaseline("pls", lambda: PLSRegression(n_components=2)),
        "random_forest": SklearnBaseline(
            "random_forest",
            lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0),
            weighted=True,
        ),
        "gradient_boosting": SklearnBaseline(
            "gradient_boosting",
            lambda: HistGradientBoostingRegressor(max_iter=300, random_state=0),
            weighted=True,
        ),
        "neural_net": SklearnBaseline(
            "neural_net",
            lambda: MLPRegressor(
                hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=20000,
                early_stopping=True, random_state=0,
            ),
        ),
    }
    if dl_surrogates.TORCH_AVAILABLE:
        families["deep_ensemble"] = dl_surrogates.DeepEnsemble()
        families["mc_dropout"] = dl_surrogates.MCDropoutNet()
        families["evidential"] = dl_surrogates.EvidentialNet()
    else:
        warnings.warn(
            "torch not installed — skipping deep-learning surrogates (uv sync --extra dl)",
            stacklevel=2,
        )
    return families


def pooled_observations(
    observable: str, campaign_ids=REAL_CAMPAIGNS, dsn: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate `observable` observations across several campaigns' stores.

    Free (reads what's already in Postgres, no new simulations). Different
    campaigns explore the same domain box with different strategies/seeds, so
    pooling is a legitimate bigger, more space-filling-ish training set.
    """
    Xs, ys, noise_vars = [], [], []
    for cid in campaign_ids:
        with PostgresStore(cid, dsn=dsn) as store:
            X, y, noise_var = store.observations_for(observable)
        Xs.append(X)
        ys.append(y)
        noise_vars.append(noise_var)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(noise_vars)
