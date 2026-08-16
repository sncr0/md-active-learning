"""Shared pytest fixtures — Postgres store reachability + per-test isolation.

Store-backed tests need a live Postgres (`docker compose up -d db`); they skip
cleanly if it isn't reachable, the same way md-extra tests skip without
openmm. Every test gets its own throwaway campaign_id so tests can share one
Postgres instance without colliding, and its rows are deleted afterward so
the test database doesn't grow without bound.
"""

from __future__ import annotations

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


@pytest.fixture
def campaign_id(_require_postgres):
    cid = f"test-{uuid.uuid4().hex[:12]}"
    yield cid
    with psycopg.connect(default_dsn(), autocommit=True) as con:
        con.execute("DELETE FROM observations WHERE campaign_id = %s", (cid,))
        con.execute("DELETE FROM runs WHERE campaign_id = %s", (cid,))
        con.execute("DELETE FROM campaigns WHERE id = %s", (cid,))


@pytest.fixture
def store(campaign_id):
    s = PostgresStore(campaign_id)
    yield s
    s.close()
