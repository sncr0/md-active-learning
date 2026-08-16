"""Detached campaign entry point (§8): run under tmux+nohup or a systemd user unit.

Usage:
    python scripts/run_campaign.py configs/alc_imse.toml [campaign_id]

campaign_id defaults to the spec's `name`. Resumable by construction: point it
at an existing campaign_id and it picks up where a killed session left off
(the store's `exists` check skips completed runs) — same DATABASE_URL, same
campaign_id is "the same store" now that Postgres holds every campaign.

Progress is live-queryable the moment a run lands (`SELECT ... WHERE
campaign_id = ...`) — no snapshot/export step, unlike the old DuckDB-file
store. The dashboard API (mdal.api.app) reads straight from this database.
"""

from __future__ import annotations

import sys

from mdal.campaign import CampaignSpec, run_campaign
from mdal.store import PostgresStore


def main(spec_path: str, campaign_id: str | None = None) -> None:
    spec = CampaignSpec.from_toml(spec_path)
    campaign_id = campaign_id or spec.name

    with PostgresStore(campaign_id) as store:
        run_campaign(spec, store, log_fn=lambda m: print(m, flush=True))
        print(f"{spec.name}: done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
