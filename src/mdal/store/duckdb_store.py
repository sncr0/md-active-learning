"""DuckDB-backed campaign store (WORKING machinery).

Schema mirrors the layer split:
  * `runs`         — one row per simulation (oracle output + cost + equilibration)
  * `observations` — one row per (run, observable) estimate (estimator output)

`exists(run_hash)` short-circuits recomputation; every write is idempotent
(ON CONFLICT DO NOTHING) so a kill mid-commit never double-inserts on resume.

Single-writer only — see store/base.py concurrency contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

from mdal.config import RunConfig
from mdal.records import Observation, RunRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_hash     TEXT PRIMARY KEY,
    config       JSON NOT NULL,
    temperature  DOUBLE NOT NULL,
    density      DOUBLE NOT NULL,
    n_steps      BIGINT NOT NULL,
    wall_clock_s DOUBLE NOT NULL,
    equil_cutoff BIGINT NOT NULL,
    n_frames     BIGINT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT now()
);
CREATE TABLE IF NOT EXISTS observations (
    run_hash   TEXT NOT NULL,
    observable TEXT NOT NULL,
    value      DOUBLE NOT NULL,
    sigma      DOUBLE NOT NULL,
    n_eff      DOUBLE NOT NULL,
    PRIMARY KEY (run_hash, observable)
);
"""


class DuckDBStore:
    """The one object that owns the campaign database."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._con = duckdb.connect(self.path)
        self._con.execute(_SCHEMA)

    # --- resumability -----------------------------------------------------
    def exists(self, run_hash: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM runs WHERE run_hash = ?", [run_hash]
        ).fetchone()
        return row is not None

    def get_run(self, run_hash: str) -> RunRecord | None:
        row = self._con.execute(
            "SELECT run_hash, wall_clock_s, equil_cutoff, n_frames, status "
            "FROM runs WHERE run_hash = ?",
            [run_hash],
        ).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_hash=row[0],
            wall_clock_s=row[1],
            equil_cutoff=int(row[2]),
            n_frames=int(row[3]),
            status=row[4],
        )

    # --- writes (orchestrator only) --------------------------------------
    def put_run(self, config: RunConfig, record: RunRecord) -> None:
        self._con.execute(
            "INSERT INTO runs "
            "(run_hash, config, temperature, density, n_steps, "
            " wall_clock_s, equil_cutoff, n_frames, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (run_hash) DO NOTHING",
            [
                record.run_hash,
                json.dumps(config.as_dict()),
                config.temperature,
                config.density,
                config.n_steps,
                record.wall_clock_s,
                record.equil_cutoff,
                record.n_frames,
                record.status,
            ],
        )

    def put_observation(self, obs: Observation) -> None:
        self._con.execute(
            "INSERT INTO observations "
            "(run_hash, observable, value, sigma, n_eff) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (run_hash, observable) DO NOTHING",
            [obs.run_hash, obs.observable, obs.value, obs.sigma, obs.n_eff],
        )

    # --- read side for the surrogate -------------------------------------
    def observations_for(
        self, observable: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self._con.execute(
            "SELECT r.temperature, r.density, o.value, o.sigma "
            "FROM observations o JOIN runs r USING (run_hash) "
            "WHERE o.observable = ? ORDER BY r.created_at",
            [observable],
        ).fetchall()
        if not rows:
            empty = np.empty((0, 2)), np.empty((0,)), np.empty((0,))
            return empty
        arr = np.asarray(rows, dtype=float)
        X = arr[:, 0:2]
        y = arr[:, 2]
        noise_var = arr[:, 3] ** 2
        return X, y, noise_var

    def lengths_for(self, observable: str) -> np.ndarray:
        """Production length (n_steps) per observation, aligned with observations_for.

        Cost-aware AL needs each observation's run length to recover the
        length-invariant noise coefficient K = sigma^2 * n_steps.
        """
        rows = self._con.execute(
            "SELECT r.n_steps FROM observations o JOIN runs r USING (run_hash) "
            "WHERE o.observable = ? ORDER BY r.created_at",
            [observable],
        ).fetchall()
        return np.asarray([r[0] for r in rows], dtype=float)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "DuckDBStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
