"""Process-pool executor with a single-writer discipline (§8).

CONTENT — worker body not implemented; the CONCURRENCY MACHINERY is the point
and is sketched in full here.

Model (§8): OPENMM_CPU_THREADS=1 per worker, 6-8 INDEPENDENT simulations in a
pool — ~3-4x the aggregate throughput of one multithreaded run. Workers compute
and return `(RunRecord, [Observation])`; the PARENT is the sole store writer and
commits each result as its future completes, so a killed session costs exactly
one in-flight simulation (§2 resumability). Postgres would tolerate concurrent
writers fine, but there's no reason to introduce that race — one campaign_id
still has one owning process (see store/base.py).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

from mdal.config import RunConfig
from mdal.records import Observation, RunRecord
from mdal.store.base import Store


def _simulate(config: RunConfig) -> tuple[RunRecord, list[Observation]]:
    """Worker entry point: runs in a child process, writes NOTHING to the store.

    Sets single-threaded MD, runs the oracle, estimates every observable, and
    returns plain records for the parent to persist.
    """
    os.environ["OPENMM_CPU_THREADS"] = "1"
    from mdal.estimator import FrameAverageEstimator
    from mdal.estimator.equilibration import detect_equilibration
    from mdal.oracle.lennard_jones import LennardJonesOracle

    traj = LennardJonesOracle().run(config)
    estimators = [
        FrameAverageEstimator("pressure", "pressure"),
        FrameAverageEstimator("energy", "potential_energy"),
    ]
    observations = [est.estimate(traj) for est in estimators]
    record = RunRecord(
        run_hash=config.content_hash(),
        wall_clock_s=traj.wall_clock_s,
        equil_cutoff=int(detect_equilibration(traj.per_frame["pressure"])),
        n_frames=traj.n_frames,
        status="complete",
    )
    return record, observations


def run_batch(
    configs: Iterable[RunConfig],
    store: Store,
    max_workers: int = 8,
) -> list[str]:
    """Run configs in parallel; the parent process is the ONLY writer.

    Skips configs already present in the store (resumability). Returns the
    hashes actually computed this call.
    """
    pending = [c for c in configs if not store.exists(c.content_hash())]
    computed: list[str] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_simulate, c): c for c in pending}
        for fut in as_completed(futures):
            config = futures[fut]
            record, observations = fut.result()
            # sole writer, per-simulation durability:
            store.put_run(config, record)
            for obs in observations:
                store.put_observation(obs)
            computed.append(config.content_hash())
    return computed
