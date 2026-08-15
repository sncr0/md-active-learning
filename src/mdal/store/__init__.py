"""Content-hash-keyed campaign store."""

from mdal.store.base import Store
from mdal.store.duckdb_store import DuckDBStore

__all__ = ["Store", "DuckDBStore"]
