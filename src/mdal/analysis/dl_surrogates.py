"""Deep-learning uncertainty-quantification baselines (PyTorch), extending
`model_comparison.py`'s question ("does model family matter, holding the
training set fixed?") into deep learning specifically.

At n~300 in 2D the GP already saturates accuracy (R^2~1), so raw predictive
power isn't the interesting axis for these — it's whether a NEURAL model can
produce a genuine epistemic/aleatoric split the way HeteroscedasticGP does
(§3.6.3), instead of the flat zero every `SklearnBaseline` reports. All three
below expose predict()/epistemic_std() with that shape, so they plug into
`mdal.tracking.score_vs_reference` / `mdal.analysis.error_map_figure`
unchanged, same as everything in model_comparison.py:

  * DeepEnsemble  — Lakshminarayanan et al. 2017. Several small MLPs, each
    with a heteroscedastic (mean, log-variance) head trained via Gaussian
    NLL. Ensemble-mean of the means = prediction; mean of the predicted
    variances = aleatoric; VARIANCE ACROSS member means = epistemic.
  * MCDropoutNet  — Gal & Ghahramani 2016. One network, dropout left ON at
    inference; K stochastic forward passes approximate a predictive
    distribution (spread = epistemic). No learned variance head, so
    aleatoric noise is supplied the same way HeteroscedasticGP supplies it
    at unobserved points: k-NN inverse-distance interpolation of the known
    per-point training noise (mirrors gp.py's predict_noise_var; duplicated
    here in miniature rather than refactoring that load-bearing file for an
    exploratory module).
  * EvidentialNet — Amini et al. 2020 ("deep evidential regression").
    Structurally different from the two above: ONE deterministic forward
    pass predicts the 4 parameters of a Normal-Inverse-Gamma distribution
    over (mean, variance) directly — no ensembling, no sampling.

Entirely optional: importing this module never requires torch. Each class
raises ImportError with an actionable message (`uv sync --extra dl`) if
instantiated without it; `TORCH_AVAILABLE` lets callers skip these families
gracefully (see `model_comparison.model_families`).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import cdist

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def _require_torch(name: str) -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(f"{name} needs the 'dl' extra: uv sync --extra dl")


def _standardize_fit(X, y):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    x_mean, x_std = X.mean(axis=0), X.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean, y_std = y.mean(), (y.std() or 1.0)
    return (X - x_mean) / x_std, (y - y_mean) / y_std, x_mean, x_std, y_mean, y_std


def _knn_noise_var(X_query: np.ndarray, X_train: np.ndarray, noise_var_train: np.ndarray, k: int = 6):
    """Aleatoric noise at unobserved points: k-NN inverse-distance interpolation
    of the known training noise. See surrogate/gp.py's predict_noise_var."""
    D = cdist(X_query, X_train)
    kk = min(k, X_train.shape[0])
    idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    dk = np.take_along_axis(D, idx, axis=1)
    w = 1.0 / (dk + 1e-9)
    return (w * noise_var_train[idx]).sum(axis=1) / w.sum(axis=1)


# --- deep ensemble ----------------------------------------------------------

if TORCH_AVAILABLE:

    class _MeanVarNet(nn.Module):
        def __init__(self, n_in: int, hidden=(32, 32)):
            super().__init__()
            layers, d = [], n_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.Tanh()]
                d = h
            self.trunk = nn.Sequential(*layers)
            self.mean_head = nn.Linear(d, 1)
            self.logvar_head = nn.Linear(d, 1)

        def forward(self, x):
            z = self.trunk(x)
            return self.mean_head(z).squeeze(-1), self.logvar_head(z).squeeze(-1)


def _train_mean_var_net(Xs, ys, hidden, epochs, lr, seed):
    torch.manual_seed(seed)
    net = _MeanVarNet(Xs.shape[1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.as_tensor(Xs, dtype=torch.float32)
    yt = torch.as_tensor(ys, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        mean, logvar = net(Xt)
        var = nn.functional.softplus(logvar) + 1e-6
        loss = (0.5 * torch.log(var) + 0.5 * (yt - mean) ** 2 / var).mean()
        loss.backward()
        opt.step()
    net.eval()
    return net


class DeepEnsemble:
    """Ensemble of heteroscedastic MLPs (Lakshminarayanan et al. 2017)."""

    def __init__(self, n_members: int = 5, hidden=(32, 32), epochs: int = 2000,
                 lr: float = 1e-2, seed: int = 0):
        _require_torch("DeepEnsemble")
        self.n_members, self.hidden, self.epochs, self.lr, self.seed = (
            n_members, hidden, epochs, lr, seed,
        )
        self._fitted = False

    def fit(self, X, y, noise_var) -> "DeepEnsemble":
        Xs, ys, self._x_mean, self._x_std, self._y_mean, self._y_std = _standardize_fit(X, y)
        self._members = [
            _train_mean_var_net(Xs, ys, self.hidden, self.epochs, self.lr, self.seed + i)
            for i in range(self.n_members)
        ]
        self._fitted = True
        return self

    def _standardize(self, X):
        return (np.atleast_2d(np.asarray(X, dtype=float)) - self._x_mean) / self._x_std

    def _member_outputs(self, X):
        Xt = torch.as_tensor(self._standardize(X), dtype=torch.float32)
        means, variances = [], []
        with torch.no_grad():
            for net in self._members:
                m, lv = net(Xt)
                means.append(m.numpy())
                variances.append(nn.functional.softplus(lv).numpy() + 1e-6)
        return np.stack(means), np.stack(variances)  # (n_members, n_points), standardized-y units

    def predict(self, X):
        means, variances = self._member_outputs(X)
        mean_s = means.mean(axis=0)
        aleatoric_s = variances.mean(axis=0)
        epistemic_s = means.var(axis=0)
        mean = mean_s * self._y_std + self._y_mean
        total_std = np.sqrt(aleatoric_s + epistemic_s) * self._y_std
        return mean, total_std

    def epistemic_std(self, X) -> np.ndarray:
        means, _ = self._member_outputs(X)
        return means.std(axis=0) * self._y_std

    @property
    def fitted_estimator(self):
        return None  # not an sklearn estimator — MLflow model-artifact logging is skipped for this family

    def diagnostics(self) -> dict:
        return {}


# --- MC dropout ---------------------------------------------------------

if TORCH_AVAILABLE:

    class _DropoutMLP(nn.Module):
        def __init__(self, n_in: int, hidden=(64, 64), p: float = 0.15):
            super().__init__()
            layers, d = [], n_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.Tanh(), nn.Dropout(p)]
                d = h
            layers += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)


class MCDropoutNet:
    """Dropout left on at inference; K sampled forward passes approximate a
    predictive distribution (Gal & Ghahramani 2016)."""

    def __init__(self, hidden=(64, 64), p: float = 0.15, epochs: int = 3000,
                 lr: float = 5e-3, n_samples: int = 50, seed: int = 0):
        _require_torch("MCDropoutNet")
        self.hidden, self.p, self.epochs, self.lr, self.n_samples, self.seed = (
            hidden, p, epochs, lr, n_samples, seed,
        )
        self._fitted = False

    def fit(self, X, y, noise_var) -> "MCDropoutNet":
        Xs, ys, self._x_mean, self._x_std, self._y_mean, self._y_std = _standardize_fit(X, y)
        self._noise_Xs = Xs
        self._noise_var = np.maximum(np.asarray(noise_var, dtype=float), 1e-10)

        torch.manual_seed(self.seed)
        self._net = _DropoutMLP(Xs.shape[1], self.hidden, self.p)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        w = 1.0 / self._noise_var
        w = torch.as_tensor(w / w.mean(), dtype=torch.float32)
        Xt = torch.as_tensor(Xs, dtype=torch.float32)
        yt = torch.as_tensor(ys, dtype=torch.float32)
        self._net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = (w * (self._net(Xt) - yt) ** 2).mean()
            loss.backward()
            opt.step()
        self._fitted = True
        return self

    def _standardize(self, X):
        return (np.atleast_2d(np.asarray(X, dtype=float)) - self._x_mean) / self._x_std

    def _mc_samples(self, X):
        Xt = torch.as_tensor(self._standardize(X), dtype=torch.float32)
        self._net.train()  # keep dropout active for sampling
        with torch.no_grad():
            samples = np.stack([self._net(Xt).numpy() for _ in range(self.n_samples)])
        return samples  # (n_samples, n_points), standardized-y units

    def predict(self, X):
        samples = self._mc_samples(X)
        mean_s = samples.mean(axis=0)
        epistemic_var = samples.var(axis=0) * self._y_std**2
        aleatoric_var = _knn_noise_var(self._standardize(X), self._noise_Xs, self._noise_var)
        mean = mean_s * self._y_std + self._y_mean
        return mean, np.sqrt(epistemic_var + aleatoric_var)

    def epistemic_std(self, X) -> np.ndarray:
        return self._mc_samples(X).std(axis=0) * self._y_std

    @property
    def fitted_estimator(self):
        return None

    def diagnostics(self) -> dict:
        return {}


# --- deep evidential regression ---------------------------------------------

if TORCH_AVAILABLE:

    class _EvidentialHead(nn.Module):
        def __init__(self, n_in: int, hidden=(64, 64)):
            super().__init__()
            layers, d = [], n_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.Tanh()]
                d = h
            self.trunk = nn.Sequential(*layers)
            self.out = nn.Linear(d, 4)

        def forward(self, x):
            raw = self.out(self.trunk(x))
            gamma = raw[:, 0]
            nu = nn.functional.softplus(raw[:, 1]) + 1e-6
            alpha = nn.functional.softplus(raw[:, 2]) + 1.0 + 1e-6
            beta = nn.functional.softplus(raw[:, 3]) + 1e-6
            return gamma, nu, alpha, beta


def _nig_nll(y, gamma, nu, alpha, beta):
    two_b_lambda = 2 * beta * (1 + nu)
    return (
        0.5 * (math.log(math.pi) - torch.log(nu))
        - alpha * torch.log(two_b_lambda)
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + two_b_lambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )


def _nig_reg(y, gamma, nu, alpha):
    return torch.abs(y - gamma) * (2 * nu + alpha)


class EvidentialNet:
    """Deep evidential regression (Amini et al. 2020): one deterministic
    forward pass predicts a Normal-Inverse-Gamma distribution over (mean,
    variance). aleatoric = beta/(alpha-1); epistemic = beta/(nu*(alpha-1))."""

    def __init__(self, hidden=(64, 64), epochs: int = 3000, lr: float = 5e-3,
                 reg_weight: float = 1e-2, seed: int = 0):
        _require_torch("EvidentialNet")
        self.hidden, self.epochs, self.lr, self.reg_weight, self.seed = (
            hidden, epochs, lr, reg_weight, seed,
        )
        self._fitted = False

    def fit(self, X, y, noise_var) -> "EvidentialNet":
        Xs, ys, self._x_mean, self._x_std, self._y_mean, self._y_std = _standardize_fit(X, y)
        torch.manual_seed(self.seed)
        self._net = _EvidentialHead(Xs.shape[1], self.hidden)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        Xt = torch.as_tensor(Xs, dtype=torch.float32)
        yt = torch.as_tensor(ys, dtype=torch.float32)
        for _ in range(self.epochs):
            opt.zero_grad()
            gamma, nu, alpha, beta = self._net(Xt)
            loss = (
                _nig_nll(yt, gamma, nu, alpha, beta).mean()
                + self.reg_weight * _nig_reg(yt, gamma, nu, alpha).mean()
            )
            loss.backward()
            opt.step()
        self._net.eval()
        self._fitted = True
        return self

    def _standardize(self, X):
        return (np.atleast_2d(np.asarray(X, dtype=float)) - self._x_mean) / self._x_std

    def _forward(self, X):
        Xt = torch.as_tensor(self._standardize(X), dtype=torch.float32)
        with torch.no_grad():
            gamma, nu, alpha, beta = self._net(Xt)
        return gamma.numpy(), nu.numpy(), alpha.numpy(), beta.numpy()

    def predict(self, X):
        gamma, nu, alpha, beta = self._forward(X)
        aleatoric_s = beta / (alpha - 1.0)
        epistemic_s = beta / (nu * (alpha - 1.0))
        mean = gamma * self._y_std + self._y_mean
        total_std = np.sqrt(aleatoric_s + epistemic_s) * self._y_std
        return mean, total_std

    def epistemic_std(self, X) -> np.ndarray:
        _, nu, alpha, beta = self._forward(X)
        return np.sqrt(beta / (nu * (alpha - 1.0))) * self._y_std

    @property
    def fitted_estimator(self):
        return None

    def diagnostics(self) -> dict:
        return {}
