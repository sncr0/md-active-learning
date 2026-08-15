"""Self-diffusion D* from a mean-squared-displacement fit (§3.3).

CONTENT — not implemented. Second-week work.

A DIFFERENT kind of estimator: the observable comes from the slope of a
linear fit to MSD(t), not a frame average, so its uncertainty propagates from
the fit. This is the stress test for whether the uncertainty machinery
generalizes beyond simple averaging.
"""

from __future__ import annotations

from mdal.oracle.trajectory import Trajectory
from mdal.records import Observation


class DiffusionEstimator:
    """Implements the Estimator protocol via an MSD slope fit."""

    observable = "diffusion"

    def estimate(self, trajectory: Trajectory) -> Observation:  # noqa: D102
        raise NotImplementedError("content: MSD slope fit + fit uncertainty")
