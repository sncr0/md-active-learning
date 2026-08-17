"""One-time backfill: log MLflow tracking data for campaigns that ran before
`mdal.tracking` existed.

No simulations are rerun — everything needed is already in Postgres. This
reconstructs each campaign's round-by-round surrogate fits by refitting a
fresh HeteroscedasticGP on successive prefixes of its stored observations
(same trick as `mdal.analysis._common.fit_on_prefix`), using the exact same
round-assignment formula as the dashboard API (`mdal.api.app.get_campaign`)
so round numbers line up with what the dashboard already shows. MLflow run
timestamps are backed out from each run's `created_at`, so the MLflow
timeline reflects when the campaign actually ran, not when this script did.

Usage:
    uv run python scripts/backfill_mlflow.py [campaign_id ...]

With no arguments, backfills every campaign in Postgres that doesn't already
have MLflow tracking data — shortest average simulation wall-clock first (a
proxy for "cheap and likely to have many points"), so results show up fast.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import psycopg
from psycopg.rows import dict_row

import mlflow
import mlflow.sklearn
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

from mdal import tracking
from mdal.domain import Domain
from mdal.store.postgres_store import default_dsn
from mdal.surrogate import HeteroscedasticGP


def _campaigns(con, only: list[str] | None) -> list[dict]:
    rows = con.execute(
        "SELECT r.campaign_id AS id, c.strategy, c.observable, c.seed, c.domain, "
        "       c.n_initial, c.n_total, c.batch, COUNT(*) AS n_runs, "
        "       AVG(r.wall_clock_s) AS avg_wall_clock_s "
        "FROM runs r JOIN campaigns c ON c.id = r.campaign_id "
        "GROUP BY r.campaign_id, c.strategy, c.observable, c.seed, c.domain, "
        "         c.n_initial, c.n_total, c.batch "
        "ORDER BY avg_wall_clock_s ASC NULLS LAST"
    ).fetchall()
    if only:
        rows = [r for r in rows if r["id"] in only]
    return rows


def _runs(con, campaign_id: str, observable: str) -> list[dict]:
    return con.execute(
        "SELECT r.temperature, r.density, r.created_at, o.value, o.sigma "
        "FROM runs r JOIN observations o "
        "  ON o.campaign_id = r.campaign_id AND o.run_hash = r.run_hash "
        "WHERE r.campaign_id = %s AND o.observable = %s "
        "ORDER BY r.created_at",
        (campaign_id, observable),
    ).fetchall()


def _round_of(i: int, n_initial: int, batch: int) -> int:
    """Same formula as mdal.api.app.get_campaign — keeps round numbers
    consistent between the dashboard's run list and MLflow's round runs."""
    return 0 if i < n_initial else 1 + (i - n_initial) // max(batch, 1)


def _ms(dt) -> int:
    return int(dt.timestamp() * 1000)


def _already_tracked(client: MlflowClient, exp_id: str, campaign_id: str) -> bool:
    escaped = campaign_id.replace("'", "''")
    runs = client.search_runs(
        [exp_id], filter_string=f"tags.campaign_id = '{escaped}'", max_results=1,
    )
    return len(runs) > 0


def backfill_campaign(client: MlflowClient, exp_id: str, con, row: dict) -> None:
    campaign_id, observable = row["id"], row["observable"]
    n_initial, n_total, batch = row["n_initial"], row["n_total"], row["batch"]
    runs = _runs(con, campaign_id, observable)
    if len(runs) < n_initial + 1:
        print(f"  {campaign_id}: not enough runs past the initial design, skipping")
        return

    # Postgres JSONB does not preserve key insertion order (it comes back sorted
    # alphabetically, "density" before "temperature") — build the Domain by
    # explicit key name, never by dict iteration order, or T*/rho* silently swap.
    domain_dict = row["domain"] or {}
    if domain_dict:
        names = ("temperature", "density")
        domain = Domain(
            names=names,
            lows=tuple(domain_dict[n][0] for n in names),
            highs=tuple(domain_dict[n][1] for n in names),
        )
    else:
        domain = Domain()
    X = np.array([[r["temperature"], r["density"]] for r in runs])
    y = np.array([r["value"] for r in runs])
    noise_var = np.array([r["sigma"] for r in runs]) ** 2
    rounds = np.array([_round_of(i, n_initial, batch) for i in range(len(runs))])
    max_round = int(rounds.max())

    parent_tags = {
        "campaign_id": campaign_id, "strategy": row["strategy"], "observable": observable,
        "source": "backfill",
    }
    parent = client.create_run(
        exp_id, start_time=_ms(runs[0]["created_at"]), tags=parent_tags, run_name=campaign_id,
    )
    parent_id = parent.info.run_id
    client.log_batch(parent_id, params=[
        Param("n_initial", str(n_initial)), Param("n_total", str(n_total)),
        Param("batch", str(batch)), Param("seed", str(row["seed"])),
        *[Param(f"domain_{n}_{edge}", str(v))
          for n, (lo, hi) in domain_dict.items() for edge, v in (("lo", lo), ("hi", hi))],
    ])

    n_logged = 0
    for r in range(1, max_round + 1):
        prefix_n = int(np.count_nonzero(rounds < r))
        this_round_n = int(np.count_nonzero(rounds == r))
        if prefix_n == 0 or this_round_n == 0:
            continue
        surrogate = HeteroscedasticGP().fit(X[:prefix_n], y[:prefix_n], noise_var[:prefix_n])
        round_start = _ms(runs[prefix_n - 1]["created_at"])
        round_end = _ms(runs[prefix_n + this_round_n - 1]["created_at"])

        run = client.create_run(
            exp_id, start_time=round_start,
            tags={
                "campaign_id": campaign_id, "round": str(r), "source": "backfill",
                "mlflow.parentRunId": parent_id,
            },
            run_name=f"round-{r}",
        )
        metrics = {"round": float(r), "n_points": float(prefix_n), **surrogate.diagnostics()}
        rmse = tracking.rmse_vs_reference(surrogate, observable, domain)
        if rmse is not None:
            metrics["rmse_vs_reference"] = rmse
        client.log_batch(
            run.info.run_id,
            params=[Param("round", str(r)), Param("n_points", str(prefix_n))],
            metrics=[Metric(k, v, round_start, 0) for k, v in metrics.items()
                     if k not in ("round", "n_points")],
        )
        try:
            with mlflow.start_run(run_id=run.info.run_id):
                mlflow.sklearn.log_model(
                    surrogate.fitted_estimator, name="model",
                    skops_trusted_types=tracking.TRUSTED_KERNEL_TYPES,
                )
        except Exception as exc:
            print(f"    round {r}: model artifact not logged ({exc})")
        client.set_terminated(run.info.run_id, status="FINISHED", end_time=round_end)
        n_logged += 1

    client.set_terminated(parent_id, status="FINISHED", end_time=_ms(runs[-1]["created_at"]))
    print(f"  {campaign_id}: {n_logged} rounds logged -> {parent_id}")


def main(campaign_ids: list[str]) -> None:
    # Both the low-level client (explicit tracking_uri) and the fluent API
    # (mlflow.sklearn.log_model, used via mlflow.start_run(run_id=...) below)
    # need to agree on the server, or the fluent half falls back to a local
    # ./mlruns directory.
    mlflow.set_tracking_uri(tracking.DEFAULT_URI)
    client = MlflowClient(tracking_uri=tracking.DEFAULT_URI)
    exp = client.get_experiment_by_name(tracking.EXPERIMENT)
    exp_id = exp.experiment_id if exp else client.create_experiment(tracking.EXPERIMENT)

    with psycopg.connect(default_dsn(), row_factory=dict_row, autocommit=True) as con:
        rows = _campaigns(con, campaign_ids or None)
        print(f"{len(rows)} campaign(s) to consider, shortest average run time first:")
        for row in rows:
            if _already_tracked(client, exp_id, row["id"]):
                print(f"  {row['id']}: already has MLflow data, skipping")
                continue
            t0 = time.monotonic()
            backfill_campaign(client, exp_id, con, row)
            print(f"    ({time.monotonic() - t0:.1f}s)")


if __name__ == "__main__":
    main(sys.argv[1:])
