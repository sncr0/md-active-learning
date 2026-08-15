"""Ground-truth analytic reference for the LJ fluid.

Kolafa-Nezbeda (1994) — see kn_eos.py for why this EOS rather than JZG.
Exported under EOS-agnostic names so callers depend on "the reference", not the
particular fit.
"""

from mdal.reference.kn_eos import energy, pressure

reference_pressure = pressure
reference_energy = energy

__all__ = ["pressure", "energy", "reference_pressure", "reference_energy"]
