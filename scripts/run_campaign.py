"""Detached campaign entry point (§8): run under tmux+nohup or a systemd user unit.

Usage:
    python scripts/run_campaign.py configs/alc_imse.toml data/alc_imse.duckdb

Resumable by construction: point it at an existing store and it picks up where
a killed session left off (the store's `exists` check skips completed runs).
"""

from __future__ import annotations

import sys

from mdal.campaign import CampaignSpec, run_campaign
from mdal.store import DuckDBStore


def main(spec_path: str, store_path: str) -> None:
    spec = CampaignSpec.from_toml(spec_path)
    with DuckDBStore(store_path) as store:
        run_campaign(spec, store)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
