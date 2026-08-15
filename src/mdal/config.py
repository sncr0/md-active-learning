"""Canonical run configuration and content hashing.

`RunConfig` is the single source of truth for *one* simulation. Its content
hash is the store key: a row keyed by this hash means "never recompute this".

DESIGN CONTRACT (do not break):
  * The hash MUST cover everything physical that affects the trajectory:
    state point, system size, dynamics length, thermostat, cutoff, seed.
  * `n_steps` (simulation length) lives HERE and nowhere else. Stage 2
    (cost-aware AL) turns length into a decision variable; if any other
    module hard-codes it, Stage 2 is blocked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines a single simulation. Immutable and hashable."""

    # --- state point (the two continuous knobs the surrogate learns over) ---
    temperature: float  # T* reduced
    density: float  # rho* reduced

    # --- system ---
    n_particles: int = 864  # 4 * 6**3, clean FCC init
    cutoff: float = 2.5  # sigma
    tail_correction: bool = True

    # --- dynamics ---
    n_steps: int = 100_000  # LENGTH — Stage 2 knob. Never hard-code elsewhere.
    equil_steps: int = 20_000  # nominal warmup; estimator detects the real cutoff
    sample_interval: int = 100  # steps between recorded frames
    timestep: float = 0.005  # reduced units
    thermostat: str = "langevin"  # "langevin" | "nose_hoover"
    friction: float = 1.0  # thermostat coupling (reduced units)

    seed: int = 0

    def content_hash(self) -> str:
        """Deterministic, order-independent key over all fields."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_dict(self) -> dict:
        return asdict(self)
