"""FastAPI app for the DOE dashboard.

Queries the same Postgres database a live campaign writes to, directly — no
snapshot/export step. That only works because Postgres's MVCC lets readers
see a consistent view without blocking on (or being blocked by) the writer;
the old DuckDB-file store took an exclusive lock that made this impossible
(see store/base.py). A fresh connection per request is plenty for a handful
of dashboard viewers polling every few seconds.

Dev:  uv run python scripts/run_api.py
"""

from __future__ import annotations

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from psycopg.rows import dict_row

from mdal.api import mlflow_client
from mdal.store.postgres_store import default_dsn

app = FastAPI(title="md-active-learning dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _connect() -> psycopg.Connection:
    return psycopg.connect(default_dsn(), row_factory=dict_row)


def _shape_summary(row: dict, tracking: dict) -> dict:
    n_complete = row["n_complete"]
    n_total = max(row["n_total"] or 0, n_complete)
    metrics = tracking.get("metrics", {})
    return {
        "id": row["id"],
        "name": row["name"],
        "strategy": row["strategy"],
        "observable": row["observable"],
        "status": "running" if n_complete < n_total else "complete",
        "n_complete": n_complete,
        "budget": {"n_initial": row["n_initial"] or 0, "n_total": n_total, "batch": row["batch"] or 0},
        "progress": (n_complete / n_total) if n_total else 1.0,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "tracking": {
            "available": bool(tracking),
            "n_rounds": tracking.get("n_rounds"),
            "r_squared": metrics.get("r_squared_vs_reference"),
            "rmse": metrics.get("rmse_vs_reference"),
        },
    }


@app.get("/api/campaigns")
def list_campaigns() -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT r.campaign_id AS id, "
            "       COALESCE(c.name, r.campaign_id) AS name, "
            "       COALESCE(c.strategy, 'unknown') AS strategy, "
            "       COALESCE(c.observable, '') AS observable, "
            "       c.n_initial, c.n_total, c.batch, "
            "       COUNT(*) AS n_complete, MAX(r.created_at) AS updated_at "
            "FROM runs r LEFT JOIN campaigns c ON c.id = r.campaign_id "
            "GROUP BY r.campaign_id, c.name, c.strategy, c.observable, "
            "         c.n_initial, c.n_total, c.batch"
        ).fetchall()
    # MLflow tracking is optional/live telemetry, layered on top of the always-
    # present Postgres provenance — one bulk lookup, fails soft to {} so a down
    # tracking server just means blank performance columns, never a 5xx here.
    tracking_by_id = mlflow_client.campaigns_summary()
    summaries = [_shape_summary(r, tracking_by_id.get(r["id"], {})) for r in rows]
    summaries.sort(key=lambda s: (s["status"] != "running", s["name"]))
    return summaries


@app.get("/api/campaigns/{campaign_id}")
def get_campaign(campaign_id: str) -> dict:
    with _connect() as con:
        meta = con.execute(
            "SELECT name, strategy, observable, seed, domain, n_initial, n_total, batch "
            "FROM campaigns WHERE id = %s",
            (campaign_id,),
        ).fetchone()
        runs = con.execute(
            "SELECT run_hash, config, temperature, density, n_steps, wall_clock_s, "
            "       equil_cutoff, n_frames, status, created_at "
            "FROM runs WHERE campaign_id = %s ORDER BY created_at",
            (campaign_id,),
        ).fetchall()
        obs_rows = con.execute(
            "SELECT run_hash, observable, value, sigma, n_eff "
            "FROM observations WHERE campaign_id = %s",
            (campaign_id,),
        ).fetchall()

    if not runs and meta is None:
        raise HTTPException(status_code=404, detail=f"unknown campaign: {campaign_id}")

    obs_by_run: dict[str, dict] = {}
    for o in obs_rows:
        obs_by_run.setdefault(o["run_hash"], {})[o["observable"]] = {
            "value": o["value"], "sigma": o["sigma"], "n_eff": o["n_eff"],
        }

    n_initial = (meta or {}).get("n_initial") or 0
    batch = (meta or {}).get("batch") or 0
    shaped_runs = []
    for i, r in enumerate(runs):
        round_ = 0 if i < n_initial else 1 + (i - n_initial) // max(batch, 1)
        shaped_runs.append({
            "run_hash": r["run_hash"],
            "index": i,
            "round": round_,
            "temperature": r["temperature"],
            "density": r["density"],
            "n_steps": r["n_steps"],
            "equil_cutoff": r["equil_cutoff"],
            "n_frames": r["n_frames"],
            "wall_clock_s": r["wall_clock_s"],
            "status": r["status"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "config": r["config"],
            "observations": obs_by_run.get(r["run_hash"], {}),
        })

    n_complete = len(shaped_runs)
    n_total = max((meta or {}).get("n_total") or 0, n_complete)
    return {
        "id": campaign_id,
        "name": (meta or {}).get("name") or campaign_id,
        "strategy": (meta or {}).get("strategy") or "unknown",
        "observable": (meta or {}).get("observable") or "",
        "seed": (meta or {}).get("seed") or 0,
        "domain": (meta or {}).get("domain") or {},
        "budget": {"n_initial": n_initial, "n_total": n_total, "batch": batch},
        "status": "running" if n_complete < n_total else "complete",
        "n_complete": n_complete,
        "progress": (n_complete / n_total) if n_total else 1.0,
        "runs": shaped_runs,
    }


@app.get("/api/campaigns/{campaign_id}/tracking")
def get_campaign_tracking(campaign_id: str) -> dict:
    """Round-by-round surrogate-fit metrics from MLflow (mdal.tracking).

    Separate from get_campaign: that's simulation provenance (Postgres, always
    present); this is ML telemetry (MLflow, optional). {"available": False} —
    never a 5xx — whenever there's nothing to show, so the dashboard degrades
    to just hiding the panel rather than erroring.
    """
    return mlflow_client.campaign_tracking(campaign_id)


@app.get("/api/health")
def health() -> dict:
    try:
        with _connect() as con:
            n = con.execute("SELECT count(*) AS n FROM campaigns").fetchone()["n"]
        return {"ok": True, "campaigns": n}
    except psycopg.OperationalError as e:
        raise HTTPException(status_code=503, detail=f"database unreachable: {e}") from e
