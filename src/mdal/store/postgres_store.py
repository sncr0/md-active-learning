"""Postgres-backed campaign store (WORKING machinery).

Schema (docker/postgres/init.sql) mirrors the layer split:
  * `runs`         — one row per simulation (oracle output + cost + equilibration)
  * `observations` — one row per (run, observable) estimate (estimator output)
  * `campaigns`    — optional metadata (name/strategy/budget) for the dashboard

A store instance is scoped to one `campaign_id`; all campaigns share one
physical database, but `runs`/`observations` are keyed by (campaign_id,
run_hash) rather than run_hash alone, so `exists()` / dedup stays per-campaign
— two campaigns proposing the identical state point never collide. Every
write is idempotent (`ON CONFLICT DO NOTHING`), so a kill mid-commit never
double-inserts on resume, same as the old store.

See store/base.py for the concurrency contract this replaces.
"""

from __future__ import annotations

import json
import os

import numpy as np
import psycopg
from psycopg.rows import tuple_row

from mdal.config import RunConfig
from mdal.records import Observation, RunRecord

DEFAULT_DSN = "postgresql://mdal:mdal@localhost:5432/mdal"


def default_dsn() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DSN)


class PostgresStore:
    """One instance = one campaign's view onto the shared `mdal` database."""

    def __init__(self, campaign_id: str, dsn: str | None = None):
        self.campaign_id = campaign_id
        self.dsn = dsn or default_dsn()
        self._con = psycopg.connect(self.dsn, autocommit=True, row_factory=tuple_row)

    # --- dashboard metadata -------------------------------------------------
    def register_campaign(
        self,
        *,
        name: str,
        strategy: str,
        observable: str,
        seed: int,
        domain: dict,
        n_initial: int,
        n_total: int,
        batch: int,
    ) -> None:
        """Upsert this campaign's metadata row. Optional — writes work without
        it — but the dashboard API needs it to show a name/strategy/budget
        instead of just a bare campaign_id."""
        self._con.execute(
            "INSERT INTO campaigns "
            "(id, name, strategy, observable, seed, domain, n_initial, n_total, batch) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET "
            "  name = EXCLUDED.name, strategy = EXCLUDED.strategy, "
            "  observable = EXCLUDED.observable, seed = EXCLUDED.seed, "
            "  domain = EXCLUDED.domain, n_initial = EXCLUDED.n_initial, "
            "  n_total = EXCLUDED.n_total, batch = EXCLUDED.batch",
            (
                self.campaign_id, name, strategy, observable, seed,
                json.dumps(domain), n_initial, n_total, batch,
            ),
        )

    # --- resumability -----------------------------------------------------
    def exists(self, run_hash: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM runs WHERE campaign_id = %s AND run_hash = %s",
            (self.campaign_id, run_hash),
        ).fetchone()
        return row is not None

    def get_run(self, run_hash: str) -> RunRecord | None:
        row = self._con.execute(
            "SELECT run_hash, wall_clock_s, equil_cutoff, n_frames, status "
            "FROM runs WHERE campaign_id = %s AND run_hash = %s",
            (self.campaign_id, run_hash),
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
            "(campaign_id, run_hash, config, temperature, density, n_steps, "
            " wall_clock_s, equil_cutoff, n_frames, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (campaign_id, run_hash) DO NOTHING",
            (
                self.campaign_id,
                record.run_hash,
                json.dumps(config.as_dict()),
                config.temperature,
                config.density,
                config.n_steps,
                record.wall_clock_s,
                record.equil_cutoff,
                record.n_frames,
                record.status,
            ),
        )

    def put_observation(self, obs: Observation) -> None:
        self._con.execute(
            "INSERT INTO observations "
            "(campaign_id, run_hash, observable, value, sigma, n_eff) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (campaign_id, run_hash, observable) DO NOTHING",
            (self.campaign_id, obs.run_hash, obs.observable, obs.value, obs.sigma, obs.n_eff),
        )

    # --- read side for the surrogate -------------------------------------
    def observations_for(
        self, observable: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self._con.execute(
            "SELECT r.temperature, r.density, o.value, o.sigma "
            "FROM observations o JOIN runs r "
            "  ON r.campaign_id = o.campaign_id AND r.run_hash = o.run_hash "
            "WHERE o.campaign_id = %s AND o.observable = %s "
            "ORDER BY r.created_at",
            (self.campaign_id, observable),
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
            "SELECT r.n_steps FROM observations o JOIN runs r "
            "  ON r.campaign_id = o.campaign_id AND r.run_hash = o.run_hash "
            "WHERE o.campaign_id = %s AND o.observable = %s "
            "ORDER BY r.created_at",
            (self.campaign_id, observable),
        ).fetchall()
        return np.asarray([r[0] for r in rows], dtype=float)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "PostgresStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
