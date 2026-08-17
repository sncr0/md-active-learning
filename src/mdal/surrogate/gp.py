"""Heteroscedastic Gaussian-process surrogate (§3.6.3).

The one non-negotiable property: per-point observation noise enters through
sklearn's `alpha` VECTOR, never a WhiteKernel. Consequences:

  * `alpha` sits on the training-covariance diagonal, so the posterior std
    sklearn returns is the LATENT-function (epistemic) std — it excludes the
    observation noise. That is the reducible uncertainty.
  * Total predictive variance for a new observation is epistemic + the
    observation noise AT that point. The noise at an unobserved point is not
    something the main GP knows, so a small deterministic noise interpolator
    (Nadaraya-Watson over the training noise) supplies sigma_n^2(x*).

Inputs and outputs are in physical (T*, rho*) / observable units; the class
standardises X and y internally so the kernel length scales stay well
conditioned.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


class HeteroscedasticGP:
    """Implements the Surrogate protocol."""

    def __init__(self, kernel=None, noise_floor: float = 1e-10, n_restarts: int = 4,
                 random_state: int = 0):
        self._kernel = kernel
        self.noise_floor = noise_floor
        self.n_restarts = n_restarts
        self.random_state = random_state  # seeds the optimizer restarts -> reproducible fits
        self._fitted = False

    # --- fitting ----------------------------------------------------------
    def fit(self, X, y, noise_var) -> "HeteroscedasticGP":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        noise_var = np.maximum(np.asarray(noise_var, dtype=float).ravel(), self.noise_floor)

        # standardise inputs and outputs
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0
        self._y_mean = y.mean()
        self._y_std = y.std() or 1.0

        Xs = (X - self._x_mean) / self._x_std
        ys = (y - self._y_mean) / self._y_std
        alpha = noise_var / self._y_std**2  # per-point noise in standardised y units

        d = X.shape[1]
        kernel = self._kernel or (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=[1.0] * d, length_scale_bounds=[(1e-2, 1e2)] * d)
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=False,
            n_restarts_optimizer=self.n_restarts, random_state=self.random_state,
        )
        self._gp.fit(Xs, ys)

        # deterministic noise model over standardised inputs: k-NN inverse-distance
        # weighting. Scale-adaptive (always averages the k nearest samples), so it
        # never collapses to the global mean the way a fixed bandwidth does when the
        # design is clustered.
        self._noise_Xs = Xs
        self._noise_var = noise_var
        self._noise_k = min(6, Xs.shape[0])

        self._fitted = True
        return self

    # --- prediction -------------------------------------------------------
    def _standardize(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return (X - self._x_mean) / self._x_std

    def predict_noise_var(self, X) -> np.ndarray:
        """Interpolated observation-noise variance sigma_n^2(x*) (aleatoric).

        k-NN inverse-distance weighting over the training noise.
        """
        Xs = self._standardize(X)
        D = cdist(Xs, self._noise_Xs)
        k = self._noise_k
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        dk = np.take_along_axis(D, idx, axis=1)
        w = 1.0 / (dk + 1e-9)
        return (w * self._noise_var[idx]).sum(axis=1) / w.sum(axis=1)

    def epistemic_std(self, X) -> np.ndarray:
        """Latent-function posterior std (reducible). Excludes observation noise."""
        _, std = self._gp.predict(self._standardize(X), return_std=True)
        return std * self._y_std

    def epistemic_var(self, X) -> np.ndarray:
        return self.epistemic_std(X) ** 2

    def predict(self, X):
        """Return (mean, total_std) — total_std includes observation noise at X."""
        mean, std = self._gp.predict(self._standardize(X), return_std=True)
        mean = mean * self._y_std + self._y_mean
        epi_var = (std * self._y_std) ** 2
        total_var = epi_var + self.predict_noise_var(X)
        return mean, np.sqrt(total_var)

    @property
    def fitted_estimator(self) -> GaussianProcessRegressor | None:
        """The underlying sklearn estimator, for export/tracking. None until fit()."""
        return self._gp if self._fitted else None

    def diagnostics(self) -> dict:
        """Flat dict of fit diagnostics for experiment tracking — log-marginal-
        likelihood plus every numeric learned kernel hyperparameter (length
        scales, amplitude). Not used by the surrogate/decision layers
        themselves; generic over whatever kernel `fit()` ended up with, so it
        doesn't assume the default ConstantKernel * RBF structure.
        """
        if not self._fitted:
            return {}
        out = {"log_marginal_likelihood": float(self._gp.log_marginal_likelihood())}
        for name, val in self._gp.kernel_.get_params().items():
            if isinstance(val, (int, float, np.floating)):
                out[f"kernel_{name}"] = float(val)
            elif isinstance(val, np.ndarray) and val.ndim == 1:
                for i, v in enumerate(val):
                    out[f"kernel_{name}_{i}"] = float(v)
        return out

    def posterior_cov(self, A, B=None) -> np.ndarray:
        """Latent (epistemic) posterior covariance, in observable^2 units.

        posterior_cov(A) -> full cov over A; posterior_cov(A, B) -> the A x B
        cross-covariance block. Used by ALC/IMSE to score how observing at one
        point would reduce variance at others.
        """
        As = self._standardize(A)
        if B is None:
            _, cov = self._gp.predict(As, return_cov=True)
            return cov * self._y_std**2
        Bs = self._standardize(B)
        n_a = As.shape[0]
        _, cov = self._gp.predict(np.vstack([As, Bs]), return_cov=True)
        return cov[:n_a, n_a:] * self._y_std**2
