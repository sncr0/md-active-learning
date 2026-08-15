"""Boundary contracts between the three layers.

These small immutable records are the ONLY things that cross layer
boundaries. Trajectories stay inside oracle+estimator; the decision layer
never sees anything but `Observation`s (via the store).

  ORACLE   --> Trajectory (oracle-internal) --> ESTIMATOR --> Observation
                                                              + RunRecord
                                                                   |
                                                                   v
                                                                 STORE
                                                                   |
                                          (X, y, noise_var) arrays v
                                                              DECISION
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunRecord:
    """Run-level result: what the oracle produced, minus the observables.

    One per completed simulation. `exists(run_hash)` on this is the
    resumability check.
    """

    run_hash: str
    wall_clock_s: float
    equil_cutoff: int  # frames discarded, as DETECTED (not the nominal equil_steps)
    n_frames: int  # total frames recorded
    status: str = "complete"  # "complete" | "failed" | "phase_separated"


@dataclass(frozen=True)
class Observation:
    """A single observable estimated from a trajectory, with honest error.

    `sigma` MUST be a standard error computed from the EFFECTIVE sample size
    (autocorrelation-corrected), never from the raw frame count. The decision
    layer consumes `variance` directly; a wrong sigma here silently corrupts
    every acquisition choice downstream.
    """

    run_hash: str
    observable: str  # "pressure" | "energy" | "diffusion"
    value: float
    sigma: float  # std error of the mean, from n_eff
    n_eff: float  # effective (independent) sample count

    @property
    def variance(self) -> float:
        return self.sigma * self.sigma
