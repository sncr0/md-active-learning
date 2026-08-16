"""One-time migration: import the archived data/*.duckdb result sets into Postgres.

These are the real Stage 1/2 results behind the docs write-up (docs/figdata.js,
the README figures) — never deletes or modifies the source .duckdb files, only
reads them. Safe to re-run: every insert is ON CONFLICT DO NOTHING, same as a
resumed campaign.

Usage:  uv sync --extra migrate && uv run python scripts/migrate_duckdb_to_postgres.py
Requires Postgres running (docker compose up -d) and DATABASE_URL set if not
using the default local dev DSN.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import psycopg

from mdal.store import PostgresStore
from mdal.store.postgres_store import default_dsn

DOMAIN = {"temperature": [1.35, 3.0], "density": [0.05, 0.9]}

# n_total is the actual run count these archived campaigns finished at
# (compare_acquisitions.py / cost_aware_demo.py's own budgets, not the
# configs/*.toml n_total — those scripts override it via CLI defaults).
CAMPAIGNS = [
    dict(id="alc_imse", name="ALC / IMSE", strategy="alc_imse",
         store="data/cmp_alc_imse_s0.duckdb", observable="pressure", seed=0,
         n_initial=8, n_total=48, batch=8),
    dict(id="epistemic", name="Epistemic-only", strategy="epistemic",
         store="data/cmp_epistemic_s0.duckdb", observable="pressure", seed=0,
         n_initial=8, n_total=48, batch=8),
    dict(id="max_variance", name="Max-variance (naive)", strategy="max_variance",
         store="data/cmp_max_variance_s0.duckdb", observable="pressure", seed=0,
         n_initial=8, n_total=48, batch=8),
    dict(id="latin_hypercube", name="Latin hypercube (baseline)", strategy="latin_hypercube",
         store="data/cmp_latin_hypercube_s0.duckdb", observable="pressure", seed=0,
         n_initial=8, n_total=48, batch=8),
    dict(id="cost_aware", name="Cost-aware ALC", strategy="cost_aware_alc",
         store="data/cost_cost_aware_s0.duckdb", observable="pressure", seed=0,
         n_initial=8, n_total=56, batch=8),
    dict(id="fixed_len", name="Fixed-length ALC", strategy="fixed_length_alc",
         store="data/cost_fixed_len_s0.duckdb", observable="pressure", seed=0,
         n_initial=8, n_total=48, batch=8),
    # dashboard demo campaigns (configs/dashboard_demo.toml, configs/dashboard_live.toml)
    dict(id="dashboard_demo", name="dashboard_demo", strategy="alc_imse",
         store="data/dashboard_demo.duckdb", observable="pressure", seed=7,
         n_initial=8, n_total=64, batch=8),
    dict(id="dashboard_live", name="dashboard_live", strategy="max_variance",
         store="data/dashboard_live.duckdb", observable="pressure", seed=11,
         n_initial=8, n_total=80, batch=8),
]


def migrate_one(campaign_id: str, name: str, strategy: str, store_path: str,
                 observable: str, seed: int, n_initial: int, n_total: int, batch: int) -> None:
    path = Path(store_path)
    if not path.exists():
        print(f"skip {campaign_id}: {path} not found")
        return

    con = duckdb.connect(str(path), read_only=True)
    runs = con.execute(
        "SELECT run_hash, config, temperature, density, n_steps, wall_clock_s, "
        "equil_cutoff, n_frames, status FROM runs ORDER BY created_at"
    ).fetchall()
    obs = con.execute(
        "SELECT run_hash, observable, value, sigma, n_eff FROM observations"
    ).fetchall()
    con.close()

    pg = PostgresStore(campaign_id)
    pg.register_campaign(
        name=name, strategy=strategy, observable=observable, seed=seed,
        domain=DOMAIN, n_initial=n_initial, n_total=max(n_total, len(runs)), batch=batch,
    )
    pg.close()

    with psycopg.connect(default_dsn(), autocommit=True) as con, con.cursor() as cur:
        cur.executemany(
            "INSERT INTO runs (campaign_id, run_hash, config, temperature, density, n_steps, "
            " wall_clock_s, equil_cutoff, n_frames, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (campaign_id, run_hash) DO NOTHING",
            [(campaign_id, *row) for row in runs],
        )
        cur.executemany(
            "INSERT INTO observations (campaign_id, run_hash, observable, value, sigma, n_eff) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (campaign_id, run_hash, observable) DO NOTHING",
            [(campaign_id, *row) for row in obs],
        )
    print(f"migrated {campaign_id}: {len(runs)} runs, {len(obs)} observations")


def main() -> None:
    for c in CAMPAIGNS:
        migrate_one(
            campaign_id=c["id"], name=c["name"], strategy=c["strategy"],
            store_path=c["store"], observable=c["observable"], seed=c["seed"],
            n_initial=c["n_initial"], n_total=c["n_total"], batch=c["batch"],
        )


if __name__ == "__main__":
    main()
