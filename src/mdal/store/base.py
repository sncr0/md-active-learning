"""Store interface — the seam between 'what was measured' and 'what to do next'.

Kept narrow on purpose so DuckDB can be swapped for Postgres at the
distributed-executor phase (Stage 2+) without touching any caller.

CONCURRENCY CONTRACT: exactly one writer. The orchestrator owns the store;
pool workers return results to it and never open the store themselves.
DuckDB takes an exclusive file lock, so a second writer is impossible anyway.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from mdal.config import RunConfig
from mdal.records import Observation, RunRecord


@runtime_checkable
class Store(Protocol):
    def exists(self, run_hash: str) -> bool:
        """True if this simulation has already been run (resumability check)."""
        ...

    def get_run(self, run_hash: str) -> RunRecord | None: ...

    def put_run(self, config: RunConfig, record: RunRecord) -> None: ...

    def put_observation(self, obs: Observation) -> None: ...

    def observations_for(
        self, observable: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read side for the surrogate: (X[n,d], y[n], noise_var[n]).

        Returns arrays ONLY — never trajectories. This is the wall that keeps
        the surrogate from reaching into raw simulation data.
        """
        ...
