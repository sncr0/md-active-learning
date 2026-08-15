"""Kolafa-Nezbeda reference EOS: port correctness + physical validation.

Two kinds of check:
  * thermodynamic self-consistency (needs NO external data) — if the pressure
    and energy ports disagree with derivatives of the free-energy port, a
    coefficient was mistranscribed;
  * agreement with the independent MD oracle — the calibration loop (§3.7).
"""

import numpy as np
import pytest

from mdal.reference import energy, pressure
from mdal.reference.kn_eos import helmholtz, helmholtz_res

POINTS = [(2.0, 0.3), (1.5, 0.5), (1.35, 0.2), (3.0, 0.8), (2.5, 0.05)]


@pytest.mark.parametrize("T,rho", POINTS)
def test_pressure_consistent_with_free_energy(T, rho):
    # P = rho^2 d(A/N)/drho
    h = 1e-6
    dA = (helmholtz(T, rho + h) - helmholtz(T, rho - h)) / (2 * h)
    assert float(pressure(T, rho)) == pytest.approx(rho**2 * float(dA), rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("T,rho", POINTS)
def test_energy_consistent_with_free_energy(T, rho):
    # U_res = d(A_res / T)/d(1/T) at fixed rho
    h = 1e-6
    beta = 1.0 / T
    f = lambda b: float(helmholtz_res(1.0 / b, rho)) * b
    dU = (f(beta + h) - f(beta - h)) / (2 * h)
    assert float(energy(T, rho)) == pytest.approx(dU, rel=1e-5, abs=1e-6)


def test_vectorises_over_grid():
    T = np.array([1.5, 2.0, 2.5])[:, None]
    R = np.array([0.1, 0.3, 0.6])[None, :]
    P = pressure(T, R)
    assert P.shape == (3, 3)
    assert np.all(np.isfinite(P))


def test_ideal_gas_limit():
    # as rho -> 0, P* -> rho* T*  and residual U* -> 0
    T = 2.0
    for rho in (1e-4, 5e-4):
        assert float(pressure(T, rho)) == pytest.approx(rho * T, rel=2e-2)
    assert abs(float(energy(T, 1e-4))) < 1e-2


def test_pressure_monotone_in_density_supercritical():
    # along a supercritical isotherm, P rises with density
    T = 2.0
    rhos = np.linspace(0.05, 0.9, 12)
    P = pressure(T, rhos)
    assert np.all(np.diff(P) > 0)


# --- the calibration loop: independent MD must match the analytic EOS --------
def test_md_agrees_with_reference():
    pytest.importorskip("openmm")
    from mdal.config import RunConfig
    from mdal.estimator import FrameAverageEstimator
    from mdal.oracle.lennard_jones import LennardJonesOracle

    T, rho = 2.0, 0.3
    cfg = RunConfig(
        temperature=T, density=rho, n_particles=864,
        equil_steps=5000, n_steps=15000, sample_interval=100,
    )
    traj = LennardJonesOracle().run(cfg)
    p_obs = FrameAverageEstimator("pressure", "pressure").estimate(traj)
    u_obs = FrameAverageEstimator("energy", "potential_energy").estimate(traj)

    p_ref = float(pressure(T, rho))
    u_ref = float(energy(T, rho))
    p_z = abs(p_obs.value - p_ref) / p_obs.sigma
    u_z = abs(u_obs.value - u_ref) / u_obs.sigma
    print(f"\n  P*: MD {p_obs.value:+.3f}±{p_obs.sigma:.3f}  KN {p_ref:+.3f}  ({p_z:.1f} sigma)")
    print(f"  U*: MD {u_obs.value:+.3f}±{u_obs.sigma:.3f}  KN {u_ref:+.3f}  ({u_z:.1f} sigma)")

    # calibration: independent MD within a few percent (finite-size + noise) of KN
    assert p_obs.value == pytest.approx(p_ref, rel=0.06)
    assert u_obs.value == pytest.approx(u_ref, rel=0.03)
