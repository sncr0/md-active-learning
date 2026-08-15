"""Oracle layer: RunConfig -> Trajectory.

The ONLY layer that touches a simulator. Swapping LJ (Stage 1) for alanine
dipeptide (Stage 3) means adding a sibling of `lennard_jones.py` and changing
nothing downstream — the decision layer must never learn which oracle ran.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mdal.config import RunConfig
from mdal.oracle.trajectory import Trajectory


@runtime_checkable
class Oracle(Protocol):
    def run(self, config: RunConfig) -> Trajectory:
        """Run one simulation to completion and return its trajectory."""
        ...
