"""Thin, dependency-free client for MLflow's REST API — used only by the
dashboard's tracking endpoints.

Deliberately not the `mlflow` package itself: that's a multi-hundred-MB extra
pulling in pandas/sklearn/etc., and the dashboard's `api` extra should stay
light. All the dashboard needs is a couple of read-only lookups, so this wraps
them with stdlib `urllib` instead. Fails soft everywhere, the same "MLflow
being down must never break anything else" contract as `mdal.tracking`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from mdal.tracking import DEFAULT_URI, EXPERIMENT, URI_ENV

_TIMEOUT = 2.0


def _base_uri() -> str:
    return os.environ.get(URI_ENV, DEFAULT_URI).rstrip("/")


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{_base_uri()}{path}", timeout=_TIMEOUT) as resp:
        return json.loads(resp.read())


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{_base_uri()}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read())


def _experiment_id() -> str | None:
    name = urllib.parse.quote(EXPERIMENT)
    resp = _get(f"/api/2.0/mlflow/experiments/get-by-name?experiment_name={name}")
    return resp.get("experiment", {}).get("experiment_id")


def _all_runs(exp_id: str) -> list[dict]:
    """Every run (campaign-level + round-level) in the experiment, one call —
    cheap at this project's scale (dozens of campaigns, tens of rounds each),
    and avoids an MLflow round-trip per campaign for the list endpoint."""
    resp = _post(
        "/api/2.0/mlflow/runs/search",
        {"experiment_ids": [exp_id], "max_results": 5000,
         "order_by": ["attributes.start_time ASC"]},
    )
    return resp.get("runs", [])


def _group_by_campaign(runs: list[dict]) -> dict[str, dict]:
    """{campaign_id: {"parent": RunInfo | None, "rounds": {round_idx: {...}}}}.

    Runs arrive oldest-first (order_by above); a resumed campaign starts a
    fresh parent run and restarts round numbering from 1, so keeping the last
    write per round index here keeps the most recent attempt at each round.
    """
    grouped: dict[str, dict] = {}
    for run in runs:
        info = run["info"]
        tags = {t["key"]: t["value"] for t in run["data"].get("tags", [])}
        cid = tags.get("campaign_id")
        if cid is None:
            continue
        g = grouped.setdefault(cid, {"parent": None, "rounds": {}})
        if "mlflow.parentRunId" not in tags:
            g["parent"] = info
            continue
        round_tag = tags.get("round")
        if round_tag is None:
            continue
        params = {p["key"]: p["value"] for p in run["data"].get("params", [])}
        metrics = {m["key"]: m["value"] for m in run["data"].get("metrics", [])}
        g["rounds"][int(round_tag)] = {
            "round": int(round_tag),
            "n_points": int(params.get("n_points", 0)),
            "metrics": metrics,
        }
    return grouped


def campaign_tracking(campaign_id: str) -> dict:
    """Round-by-round surrogate-fit metrics for one campaign, read from MLflow.

    Returns {"available": False} — never raises — if mlflow isn't reachable,
    the experiment doesn't exist yet, or this campaign never logged anything.
    The dashboard treats all three the same way: no tracking panel to show.
    """
    try:
        exp_id = _experiment_id()
        if exp_id is None:
            return {"available": False}
        g = _group_by_campaign(_all_runs(exp_id)).get(campaign_id)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return {"available": False}

    if not g or not g["rounds"]:
        return {"available": False}
    rounds = [g["rounds"][k] for k in sorted(g["rounds"])]
    parent = g["parent"]
    mlflow_url = f"{_base_uri()}/#/experiments/{exp_id}/runs/{parent['run_id']}" \
        if parent is not None else f"{_base_uri()}/#/experiments/{exp_id}"
    return {"available": True, "mlflow_url": mlflow_url, "rounds": rounds}


def campaigns_summary() -> dict[str, dict]:
    """Cheap 'how well is each campaign's surrogate doing' snapshot, for the
    campaigns list to sort/filter/rank by — one MLflow round-trip regardless
    of how many campaigns exist. {} (never raises) if MLflow is unreachable or
    nothing's been tracked yet; the list endpoint just omits those fields.
    """
    try:
        exp_id = _experiment_id()
        if exp_id is None:
            return {}
        grouped = _group_by_campaign(_all_runs(exp_id))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return {}

    out: dict[str, dict] = {}
    for cid, g in grouped.items():
        if not g["rounds"]:
            continue
        last = g["rounds"][max(g["rounds"])]
        parent = g["parent"]
        mlflow_url = f"{_base_uri()}/#/experiments/{exp_id}/runs/{parent['run_id']}" \
            if parent is not None else f"{_base_uri()}/#/experiments/{exp_id}"
        out[cid] = {
            "n_rounds": len(g["rounds"]), "metrics": last["metrics"], "mlflow_url": mlflow_url,
        }
    return out
