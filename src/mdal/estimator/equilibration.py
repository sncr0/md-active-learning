"""Automatic equilibration detection (§3.6.2).

The initial transient must be discarded before averaging, otherwise the
starting configuration biases the estimate. pymbar's detect_equilibration
chooses the cutoff t0 that maximises the number of effective samples in the
retained tail, yielding a per-run, auditable number (it matters most near
criticality, where relaxation is slowest and the campaign samples most).
"""

from __future__ import annotations

import numpy as np
from pymbar import timeseries as _ts


def detect_equilibration(series, nskip: int = 1) -> int:
    """Return the frame index at which `series` is considered equilibrated.

    `nskip` subsamples the search over candidate cutoffs; raise it for very
    long series where scanning every index is wasteful.
    """
    a = np.asarray(series, dtype=float).ravel()
    if a.size < 3 or np.ptp(a) == 0.0:
        return 0
    t0, _g, _neff = _ts.detect_equilibration(a, nskip=nskip)
    return int(t0)
