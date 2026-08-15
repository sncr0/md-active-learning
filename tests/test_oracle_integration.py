"""End-to-end: real LJ trajectory -> estimator -> honest Observation.

Skips cleanly if the `md` extra (openmm) is not installed, so the core suite
still runs without it. This is the first point where real physics meets the
validated estimator.
"""

import numpy as np
import pytest

pytest.importorskip("openmm")

from mdal.config import RunConfig
from mdal.estimator import FrameAverageEstimator
from mdal.oracle.lennard_jones import LennardJonesOracle


def test_oracle_feeds_estimator_endtoend():
    cfg = RunConfig(
        temperature=2.0, density=0.3, n_particles=864,
        equil_steps=1000, n_steps=4000, sample_interval=100,
    )
    traj = LennardJonesOracle().run(cfg)

    assert set(traj.per_frame) == {"pressure", "potential_energy"}
    assert traj.n_frames == 40
    assert np.all(np.isfinite(traj.per_frame["pressure"]))
    assert traj.wall_clock_s > 0

    obs = FrameAverageEstimator("pressure", "pressure").estimate(traj)
    assert obs.run_hash == cfg.content_hash()
    assert 0 < obs.n_eff <= traj.n_frames
    assert obs.sigma > 0
    # supercritical, moderate density -> pressure is finite and O(0.1-1), not absurd
    assert -1.0 < obs.value < 3.0


def test_low_density_pressure_near_ideal_gas():
    """Physical sanity: as rho* -> 0, P* -> rho* T* (ideal gas), from below
    (attraction). A cheap check that the virial finite-difference has the right sign."""
    cfg = RunConfig(
        temperature=2.0, density=0.05, n_particles=864,
        equil_steps=1000, n_steps=4000, sample_interval=100,
    )
    traj = LennardJonesOracle().run(cfg)
    obs = FrameAverageEstimator("pressure", "pressure").estimate(traj)
    ideal = 0.05 * 2.0  # rho* T* = 0.1
    # real pressure sits modestly below ideal here; require the right ballpark & sign
    assert 0.0 < obs.value < ideal * 1.3
