"""Effective sample size from a correlated time series (§3.6.1).

MD frames are not independent. `std / sqrt(n_frames)` underestimates the
uncertainty by 5-50x because it counts correlated samples as if independent.
The statistical inefficiency g is the number of correlated samples per
*independent* one, so the honest sample count is n_eff = n / g and the honest
standard error is std / sqrt(n_eff).

For an AR(1) process x_t = phi*x_{t-1} + eps_t the analytic value is
g = (1 + phi) / (1 - phi); the synthetic test verifies these functions
recover it. pymbar.timeseries.statistical_inefficiency is the estimator.
"""

from __future__ import annotations

import numpy as np
from pymbar import timeseries as _ts


def _as_series(series) -> np.ndarray:
    return np.asarray(series, dtype=float).ravel()


def statistical_inefficiency(series) -> float:
    """g: correlated samples per independent sample (>= 1).

    Degenerate inputs (fewer than 3 points, or a constant series) have no
    definable correlation time and return g = 1.
    """
    a = _as_series(series)
    if a.size < 3 or np.ptp(a) == 0.0:
        return 1.0
    return float(_ts.statistical_inefficiency(a))


def effective_sample_size(series) -> float:
    """n_eff = n / g: the count of statistically independent samples."""
    a = _as_series(series)
    return a.size / statistical_inefficiency(a)


def standard_error(series) -> float:
    """Std error of the mean using n_eff, NOT n. Equals std * sqrt(g / n)."""
    a = _as_series(series)
    if a.size < 2:
        return 0.0
    n_eff = effective_sample_size(a)
    return float(np.std(a, ddof=1) / np.sqrt(n_eff))
