"""The estimator's ground-truth test (§8 build order): validate the
uncertainty machinery against synthetic series of KNOWN statistical
properties, before any real MD exists.

Fixture: AR(1)  x_t = phi*x_{t-1} + eps_t, whose analytic statistical
inefficiency is g = (1 + phi) / (1 - phi). If these functions cannot recover a
known g, nothing built on them is trustworthy.
"""

import numpy as np
import pytest

from mdal.estimator import FrameAverageEstimator
from mdal.estimator.autocorr import (
    effective_sample_size,
    standard_error,
    statistical_inefficiency,
)
from mdal.estimator.equilibration import detect_equilibration
from mdal.oracle.trajectory import Trajectory

# --- synthetic fixtures (known ground truth) --------------------------------

N_LONG = 200_000  # long series -> statistical_inefficiency pins g down tightly


def ar1(n, phi, seed, mean=0.0, sigma_eps=1.0):
    """Stationary AR(1). Analytic g = (1+phi)/(1-phi), stationary var = 1/(1-phi^2)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma_eps, size=n)
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1.0 - phi**2)  # draw first point from stationary dist
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x + mean


def analytic_g(phi):
    return (1.0 + phi) / (1.0 - phi)


def with_transient(n, phi, seed, mean, t_eq, offset):
    """Stationary AR(1) around `mean`, preceded by a decaying transient of
    length t_eq starting `offset` away — mimics an unequilibrated MD start."""
    x = ar1(n, phi, seed, mean=mean)
    k = np.arange(t_eq)
    x[:t_eq] += offset * np.exp(-k / (t_eq / 4.0))
    return x


# --- g recovery: the core claim ---------------------------------------------


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.8, 0.9, 0.95])
def test_recovers_known_statistical_inefficiency(phi):
    x = ar1(N_LONG, phi, seed=12345)
    assert statistical_inefficiency(x) == pytest.approx(analytic_g(phi), rel=0.10)


def test_white_noise_is_uncorrelated():
    x = ar1(N_LONG, 0.0, seed=7)
    assert statistical_inefficiency(x) == pytest.approx(1.0, abs=0.1)


def test_effective_sample_size_tracks_g():
    x = ar1(N_LONG, 0.8, seed=1)
    assert effective_sample_size(x) == pytest.approx(N_LONG / 9.0, rel=0.10)


def test_naive_sem_underestimates_by_sqrt_g():
    """The §3.6.1 failure, quantified: naive std/sqrt(N) is too small by ~sqrt(g)."""
    phi = 0.9
    x = ar1(N_LONG, phi, seed=99)
    naive = np.std(x, ddof=1) / np.sqrt(x.size)
    honest = standard_error(x)
    assert honest / naive == pytest.approx(np.sqrt(analytic_g(phi)), rel=0.15)


def test_degenerate_series_are_safe():
    assert statistical_inefficiency([1.0, 1.0, 1.0]) == 1.0  # constant
    assert statistical_inefficiency([1.0]) == 1.0  # too short
    assert standard_error([5.0]) == 0.0


# --- the full estimator path: Trajectory -> Observation ---------------------


def test_frame_average_estimator_produces_honest_observation():
    true_mean, phi, n = 2.5, 0.85, 4000  # ~ a real run's frame count
    series = ar1(n, phi, seed=3, mean=true_mean)
    traj = Trajectory(run_hash="deadbeef", per_frame={"pressure": series})

    obs = FrameAverageEstimator("pressure", "pressure").estimate(traj)

    assert obs.observable == "pressure" and obs.run_hash == "deadbeef"
    assert abs(obs.value - true_mean) < 5 * obs.sigma  # value within a few honest sigma
    assert obs.n_eff < n  # correlation -> fewer effective than raw samples
    assert obs.n_eff == pytest.approx(n / analytic_g(phi), rel=0.30)
    assert obs.sigma == pytest.approx(
        np.std(series, ddof=1) / np.sqrt(obs.n_eff), rel=0.05
    )


def test_equilibration_discards_transient():
    true_mean, phi, n, t_eq = 1.0, 0.8, 3000, 800
    series = with_transient(n, phi, seed=11, mean=true_mean, t_eq=t_eq, offset=10.0)

    assert detect_equilibration(series) > 0  # a transient is found

    obs = FrameAverageEstimator("pressure", "pressure").estimate(
        Trajectory(run_hash="x", per_frame={"pressure": series})
    )
    # discarding the transient beats naively averaging the whole series
    full_series_bias = abs(np.mean(series) - true_mean)
    equilibrated_bias = abs(obs.value - true_mean)
    assert equilibrated_bias < full_series_bias
