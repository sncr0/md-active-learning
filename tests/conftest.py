"""Shared pytest fixtures — Postgres store reachability + per-test isolation.

Store-backed tests need a live Postgres (`docker compose up -d db`); they skip
cleanly if it isn't reachable, the same way md-extra tests skip without
openmm. Every test gets its own throwaway campaign_id so tests can share one
Postgres instance without colliding, and its rows are deleted afterward so
the test database doesn't grow without bound. The same teardown also
soft-deletes any MLflow runs logged under that campaign_id, if a tracking
server happens to be reachable — mdal.tracking has no test-mode switch, so
without this, running the suite against a live MLflow instance quietly
accumulates junk "test-xxxx" runs there forever.
"""

from __future__ import annotations

import os
import uuid

import psycopg
import pytest

from mdal.store import PostgresStore
from mdal.store.postgres_store import default_dsn


def _postgres_reachable() -> bool:
    try:
        with psycopg.connect(default_dsn(), connect_timeout=2):
            pass
        return True
    except psycopg.OperationalError:
        return False


@pytest.fixture(scope="session")
def _require_postgres():
    if not _postgres_reachable():
        pytest.skip("Postgres not reachable — `docker compose up -d db` (or set DATABASE_URL)")


def _cleanup_mlflow(campaign_id: str) -> None:
    """Soft-delete any MLflow runs this test's run_campaign() call logged.

    mdal.tracking defaults to localhost:5001 with no opt-in required, so any
    test that exercises the AL loop while a local tracking server happens to
    be running would otherwise leave junk "test-xxxx" runs behind permanently.
    Best-effort: no mlflow installed, or the server unreachable, is not a
    test failure — mirrors mdal.tracking's own fails-soft contract.
    """
    try:
        from mlflow.tracking import MlflowClient

        from mdal.tracking import DEFAULT_URI, EXPERIMENT, URI_ENV

        client = MlflowClient(tracking_uri=os.environ.get(URI_ENV, DEFAULT_URI))
        exp = client.get_experiment_by_name(EXPERIMENT)
        if exp is None:
            return
        escaped = campaign_id.replace("'", "''")
        for run in client.search_runs(
            [exp.experiment_id], filter_string=f"tags.campaign_id = '{escaped}'"
        ):
            client.delete_run(run.info.run_id)
    except Exception:
        pass


@pytest.fixture
def campaign_id(_require_postgres):
    cid = f"test-{uuid.uuid4().hex[:12]}"
    yield cid
    with psycopg.connect(default_dsn(), autocommit=True) as con:
        con.execute("DELETE FROM observations WHERE campaign_id = %s", (cid,))
        con.execute("DELETE FROM runs WHERE campaign_id = %s", (cid,))
        con.execute("DELETE FROM campaigns WHERE id = %s", (cid,))
    _cleanup_mlflow(cid)


@pytest.fixture
def store(campaign_id):
    s = PostgresStore(campaign_id)
    yield s
    s.close()
