"""Estimator layer: Trajectory -> Observation (value + HONEST uncertainty).

This is the hard part of the whole project (§1). Build and test it FIRST,
against synthetic correlated series of known statistical inefficiency, before
any real MD. If it can't recover a known autocorrelation time, nothing built
on it is trustworthy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mdal.oracle.trajectory import Trajectory
from mdal.records import Observation


@runtime_checkable
class Estimator(Protocol):
    observable: str

    def estimate(self, trajectory: Trajectory) -> Observation:
        """Reduce a trajectory to one observable with an effective-sample error.

        Every implementation MUST: (1) detect & discard equilibration, then
        (2) base its sigma on the EFFECTIVE sample size, never the raw frame
        count.
        """
        ...
