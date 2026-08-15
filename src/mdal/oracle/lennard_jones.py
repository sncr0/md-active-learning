"""Stage 1 oracle: single-site Lennard-Jones fluid in a periodic box (OpenMM).

REDUCED UNITS via OpenMM's native unit system. OpenMM's internal units make
the reduced LJ system almost free: with sigma = 1 nm, eps = 1 kJ/mol,
m = 1 amu, the reduced time unit tau = sigma*sqrt(m/eps) works out to exactly
1 ps, so a reduced timestep dt* maps to dt* ps and reduced friction gamma*
maps to gamma*/ps directly. The only awkward constant is temperature: OpenMM
fixes k_B, so to realise T* = k_B T / eps we set the thermostat to
T = T* / k_B kelvin. Energies in kJ/mol are then already in units of eps, and
numeric pressures in kJ/mol/nm^3 are already reduced (P* = P sigma^3 / eps).

PRESSURE. OpenMM does not expose the virial, so the configurational pressure
is obtained by finite-difference of the energy under an isotropic box scaling
r -> lambda*r, L -> lambda*L:  P_conf = -(dU/dlambda)/(3V). This captures the
analytic long-range (dispersion) tail correction to the pressure automatically,
since useDispersionCorrection makes U an explicit function of V (§3.1).

openmm is imported lazily so the rest of the package installs and imports
without the `md` extra.
"""

from __future__ import annotations

import os
import time

import numpy as np

from mdal.config import RunConfig
from mdal.oracle.trajectory import Trajectory

# Boltzmann constant in OpenMM units (kJ/mol/K), CODATA.
KB = 0.00831446261815324


def _fcc_positions(n_particles: int, box_length: float) -> np.ndarray:
    """FCC lattice filling a cubic box of side `box_length` (nm). N = 4 n_cells^3."""
    n_cells = round((n_particles / 4) ** (1 / 3))
    if 4 * n_cells**3 != n_particles:
        raise ValueError(f"N={n_particles} is not 4*n^3 (needed for clean FCC init)")
    a = box_length / n_cells
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array([(i, j, k) for i in range(n_cells) for j in range(n_cells) for k in range(n_cells)])
    pos = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a
    return pos


def _build_system(n_particles: int, cutoff: float, tail_correction: bool):
    import openmm as mm

    system = mm.System()
    for _ in range(n_particles):
        system.addParticle(1.0)  # mass = 1 amu (reduced)
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(cutoff)  # 2.5 sigma = 2.5 nm
    nb.setUseSwitchingFunction(False)  # hard truncation + analytic tail correction
    nb.setUseDispersionCorrection(tail_correction)
    for _ in range(n_particles):
        nb.addParticle(0.0, 1.0, 1.0)  # charge=0, sigma=1 nm, epsilon=1 kJ/mol
    system.addForce(nb)
    return system


def _make_context(config: RunConfig, threads: int = 1, minimize: bool = True):
    """Build an OpenMM Context for `config`. Returns (context, integrator, N, L)."""
    import openmm as mm
    from openmm import unit

    N = config.n_particles
    L = (N / config.density) ** (1 / 3)  # nm, from rho* = N sigma^3 / V
    positions = _fcc_positions(N, L)

    system = _build_system(N, config.cutoff, config.tail_correction)
    box = [mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, L)] * unit.nanometer
    system.setDefaultPeriodicBoxVectors(*box)

    integrator = mm.LangevinMiddleIntegrator(
        config.temperature / KB,  # T* -> kelvin
        config.friction,          # gamma* -> 1/ps
        config.timestep,          # dt* -> ps
    )
    integrator.setRandomNumberSeed(config.seed + 1)  # 0 means "random" in OpenMM

    platform = mm.Platform.getPlatformByName("CPU")
    context = mm.Context(system, integrator, platform, {"Threads": str(threads)})
    context.setPeriodicBoxVectors(*box)
    context.setPositions(positions * unit.nanometer)
    if minimize:
        mm.LocalEnergyMinimizer.minimize(context, maxIterations=200)
    context.setVelocitiesToTemperature(config.temperature / KB, config.seed + 1)
    return context, integrator, N, L


def _reduced_pressure(context, n_particles: int, temperature: float, delta: float = 1e-4) -> float:
    """Instantaneous reduced pressure via box-scaling finite difference.

    P* = rho* T* + P_conf, with P_conf = -(dU/dlambda)/(3V). The ideal term uses
    the thermostat T (its time-average equals <2 KE>/3V but adds no noise).
    """
    from openmm import Vec3, unit

    st = context.getState(getPositions=True)
    box = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    L = box[0, 0]
    V = L**3

    def energy_at(lam):
        b = [Vec3(L * lam, 0, 0), Vec3(0, L * lam, 0), Vec3(0, 0, L * lam)] * unit.nanometer
        context.setPeriodicBoxVectors(*b)
        context.setPositions((pos * lam) * unit.nanometer)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

    u_plus = energy_at(1 + delta)
    u_minus = energy_at(1 - delta)
    # restore the true state so dynamics continue unperturbed
    b0 = [Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L)] * unit.nanometer
    context.setPeriodicBoxVectors(*b0)
    context.setPositions(pos * unit.nanometer)

    dU_dlambda = (u_plus - u_minus) / (2 * delta)
    p_conf = -dU_dlambda / (3 * V)
    p_ideal = n_particles * temperature / V  # k_B T = T* (kJ/mol) in reduced units
    return p_ideal + p_conf


class LennardJonesOracle:
    """Implements the Oracle protocol for the LJ fluid."""

    def run(self, config: RunConfig) -> Trajectory:
        from openmm import unit

        t_start = time.perf_counter()
        threads = int(os.environ.get("OPENMM_CPU_THREADS", "1"))
        context, integrator, N, _L = _make_context(config, threads=threads)

        integrator.step(config.equil_steps)

        n_frames = config.n_steps // config.sample_interval
        pressure = np.empty(n_frames)
        energy = np.empty(n_frames)
        for k in range(n_frames):
            integrator.step(config.sample_interval)
            st = context.getState(getEnergy=True)
            energy[k] = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / N
            pressure[k] = _reduced_pressure(context, N, config.temperature)

        return Trajectory(
            run_hash=config.content_hash(),
            per_frame={"pressure": pressure, "potential_energy": energy},
            wall_clock_s=time.perf_counter() - t_start,
        )
