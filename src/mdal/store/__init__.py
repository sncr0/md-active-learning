"""Content-hash-keyed campaign store."""

from mdal.store.base import Store
from mdal.store.postgres_store import PostgresStore

__all__ = ["Store", "PostgresStore"]
