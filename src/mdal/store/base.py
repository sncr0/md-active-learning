"""Store interface — the seam between 'what was measured' and 'what to do next'.

Kept narrow on purpose so the backend can be swapped without touching any
caller — this is exactly what let PostgresStore replace the original
DuckDB-file store: every campaign/executor/analysis module below only ever
calls through this Protocol.

CONCURRENCY CONTRACT: the orchestrator owns the store; pool workers compute
and return results, never opening the store themselves. Postgres's MVCC means
readers (the dashboard API) never block on a live writer — unlike the old
DuckDB file, which took an exclusive lock for as long as any connection was
open, forcing a separate JSON-snapshot side channel just to expose progress.
That's gone now; the API queries the same database live.

One thing Postgres does NOT give you for free: running two `run_campaign.py`
processes against the *same* campaign_id concurrently. Nothing at the DB layer
stops it, but the acquisition loop still assumes a single driver reading its
own writes — two drivers would each propose from a stale view of the other's
in-flight batch. Not a supported use case; each campaign_id has one owner.
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
