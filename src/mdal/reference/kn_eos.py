"""Kolafa-Nezbeda (1994) equation of state for the Lennard-Jones fluid.

Ground truth for the error map (§3.4). The brief named JZG (1993); this is the
Kolafa-Nezbeda EOS instead, chosen because (a) per the Stephan et al. (2020)
review it is the most robust and accurate analytic LJ EOS, especially near
criticality where our supercritical domain lives, and (b) its reference FORTRAN
implementation could be sourced exactly, whereas JZG's coefficients could not.

Ported verbatim from the authors' reference code:
  J. Kolafa & I. Nezbeda, "The Lennard-Jones fluid: an accurate analytic and
  theoretically-based equation of state", Fluid Phase Equilibria 100 (1994) 1.

All quantities are in reduced LJ units. `pressure` returns total P*(T*, rho*);
`energy` returns the residual (configurational) internal energy per particle
U*(T*, rho*) — directly comparable to the MD oracle's potential energy per
particle. Functions are numpy-vectorised over T and rho.
"""

from __future__ import annotations

import math

import numpy as np

GAMMA_BH = 1.92907278  # the single nonlinear parameter (gammaBH in the reference)
_PI = math.pi


# --- hard-sphere / hBH building blocks --------------------------------------
def _d_hbh(T):
    """Effective hard-sphere (hBH) diameter dC(T)."""
    T = np.asarray(T, dtype=float)
    sT = np.sqrt(T)
    return (
        -0.063920968 * np.log(T)
        + 0.011117524 / T
        - 0.076383859 / sT
        + 1.080142248
        + 0.000693129 * sT
    )


def _d_hbh_dbeta(T):
    """d(dC)/d(1/T), as used in the internal-energy expression (dCdT)."""
    T = np.asarray(T, dtype=float)
    sT = np.sqrt(T)
    return 0.063920968 * T + 0.011117524 + (-0.5 * 0.076383859 - 0.5 * 0.000693129 * T) * sT


def _b2_hbh(T):
    """hBH second virial correction BC(T)."""
    T = np.asarray(T, dtype=float)
    isT = 1.0 / np.sqrt(T)
    return (
        (((((-0.58544978 * isT + 0.43102052) * isT + 0.87361369) * isT - 4.13749995) * isT
          + 2.90616279) * isT - 7.02181962) / T
        + 0.02459877
    )


def _b2_hbh_dbeta(T):
    """d(BC)/d(1/T), as used in the internal-energy expression (BCdT)."""
    T = np.asarray(T, dtype=float)
    isT = 1.0 / np.sqrt(T)
    return (
        (((-0.58544978 * 3.5 * isT + 0.43102052 * 3) * isT + 0.87361369 * 2.5) * isT
         - 4.13749995 * 2) * isT + 2.90616279 * 1.5
    ) * isT - 7.02181962


def _z_hs(eta):
    """Hard-sphere compressibility factor (Carnahan-Starling-like)."""
    return (1 + eta * (1 + eta * (1 - eta / 1.5 * (1 + eta)))) / (1 - eta) ** 3


def _beta_a_hs(eta):
    """Hard-sphere residual Helmholtz free energy, betaAHS(eta)."""
    return np.log(1 - eta) / 0.6 + eta * ((4.0 / 6 * eta - 33.0 / 6) * eta + 34.0 / 6) / (1 - eta) ** 2


def _delta_a(T, rho):
    """The C_ij correction to the residual Helmholtz free energy, DALJ(T, rho)."""
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    return (
        (2.01546797 + rho * (-28.17881636 + rho * (28.28313847 + rho * (-10.42402873))))
        + (-19.58371655 + rho * (75.62340289 + rho * (-120.70586598 + rho * (93.92740328 + rho * (-27.37737354))))) / np.sqrt(T)
        + (
            (29.34470520 + rho * (-112.35356937 + rho * (170.64908980 + rho * (-123.06669187 + rho * 34.42288969))))
            + (-13.37031968 + rho * (65.38059570 + rho * (-115.09233113 + rho * (88.91973082 + rho * (-25.62099890))))) / T
        ) / T
    ) * rho * rho


# --- public thermodynamic functions -----------------------------------------
def helmholtz(T, rho):
    """Total Helmholtz free energy per particle A*(T*, rho*) (includes T ln rho)."""
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    eta = _PI / 6.0 * rho * _d_hbh(T) ** 3
    return (np.log(rho) + _beta_a_hs(eta) + rho * _b2_hbh(T) / np.exp(GAMMA_BH * rho**2)) * T + _delta_a(T, rho)


def helmholtz_res(T, rho):
    """Residual Helmholtz free energy per particle (excludes the ideal T ln rho)."""
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    eta = _PI / 6.0 * rho * _d_hbh(T) ** 3
    return (_beta_a_hs(eta) + rho * _b2_hbh(T) / np.exp(GAMMA_BH * rho**2)) * T + _delta_a(T, rho)


def pressure(T, rho):
    """Total reduced pressure P*(T*, rho*)."""
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    eta = _PI / 6.0 * rho * _d_hbh(T) ** 3
    s = (
        (2.01546797 * 2 + rho * (-28.17881636 * 3 + rho * (28.28313847 * 4 + rho * (-10.42402873) * 5)))
        + (-19.58371655 * 2 + rho * (75.62340289 * 3 + rho * (-120.70586598 * 4 + rho * (93.92740328 * 5 + rho * (-27.37737354) * 6)))) / np.sqrt(T)
        + (
            (29.34470520 * 2 + rho * (-112.35356937 * 3 + rho * (170.64908980 * 4 + rho * (-123.06669187 * 5 + rho * 34.42288969 * 6))))
            + (-13.37031968 * 2 + rho * (65.38059570 * 3 + rho * (-115.09233113 * 4 + rho * (88.91973082 * 5 + rho * (-25.62099890) * 6)))) / T
        ) / T
    ) * rho**2
    hs = _z_hs(eta) + _b2_hbh(T) * rho * (1 - 2 * GAMMA_BH * rho**2) / np.exp(GAMMA_BH * rho**2)
    return (hs * T + s) * rho


def energy(T, rho):
    """Residual (configurational) internal energy per particle U*(T*, rho*)."""
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    d = _d_hbh(T)
    eta = _PI / 6.0 * rho * d**3
    s = (
        (2.01546797 + rho * (-28.17881636 + rho * (28.28313847 + rho * (-10.42402873))))
        + (-19.58371655 * 1.5 + rho * (75.62340289 * 1.5 + rho * (-120.70586598 * 1.5 + rho * (93.92740328 * 1.5 + rho * (-27.37737354) * 1.5)))) / np.sqrt(T)
        + (
            (29.34470520 * 2 + rho * (-112.35356937 * 2 + rho * (170.64908980 * 2 + rho * (-123.06669187 * 2 + rho * 34.42288969 * 2))))
            + (-13.37031968 * 3 + rho * (65.38059570 * 3 + rho * (-115.09233113 * 3 + rho * (88.91973082 * 3 + rho * (-25.62099890) * 3)))) / T
        ) / T
    ) * rho * rho
    return 3 * (_z_hs(eta) - 1) * _d_hbh_dbeta(T) / d + rho * _b2_hbh_dbeta(T) / np.exp(GAMMA_BH * rho**2) + s
