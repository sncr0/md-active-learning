-- Schema for the md-active-learning campaign store (see src/mdal/store/postgres_store.py).
--
-- `runs` and `observations` are keyed by (campaign_id, run_hash) rather than run_hash alone:
-- run_hash is a content hash of the physical simulation only (see mdal.config.RunConfig), so two
-- different campaigns can legitimately propose the identical state point. Scoping dedup to
-- (campaign_id, run_hash) preserves the old per-file resumability contract now that all campaigns
-- share one database.
--
-- `campaigns` is supplementary metadata for the dashboard (name/strategy/budget) — runs.campaign_id
-- has no foreign key to it, so ad hoc or test campaign_ids work without registering one first.

CREATE TABLE IF NOT EXISTS campaigns (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    observable  TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    domain      JSONB NOT NULL,
    n_initial   INTEGER NOT NULL,
    n_total     INTEGER NOT NULL,
    batch       INTEGER NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    campaign_id  TEXT NOT NULL,
    run_hash     TEXT NOT NULL,
    config       JSONB NOT NULL,
    temperature  DOUBLE PRECISION NOT NULL,
    density      DOUBLE PRECISION NOT NULL,
    n_steps      BIGINT NOT NULL,
    wall_clock_s DOUBLE PRECISION NOT NULL,
    equil_cutoff BIGINT NOT NULL,
    n_frames     BIGINT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (campaign_id, run_hash)
);
CREATE INDEX IF NOT EXISTS runs_campaign_idx ON runs (campaign_id);

CREATE TABLE IF NOT EXISTS observations (
    campaign_id TEXT NOT NULL,
    run_hash    TEXT NOT NULL,
    observable  TEXT NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    sigma       DOUBLE PRECISION NOT NULL,
    n_eff       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (campaign_id, run_hash, observable),
    FOREIGN KEY (campaign_id, run_hash) REFERENCES runs (campaign_id, run_hash)
);
