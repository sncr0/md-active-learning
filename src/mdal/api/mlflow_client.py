"""Thin, dependency-free client for MLflow's REST API — used only by the
dashboard's tracking endpoint.

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
        escaped = campaign_id.replace("'", "''")
        resp = _post(
            "/api/2.0/mlflow/runs/search",
            {
                "experiment_ids": [exp_id],
                "filter": f"tags.campaign_id = '{escaped}'",
                "max_results": 1000,
                "order_by": ["attributes.start_time ASC"],
            },
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return {"available": False}

    parent = None
    rounds_by_idx: dict[int, dict] = {}
    for run in resp.get("runs", []):
        info = run["info"]
        tags = {t["key"]: t["value"] for t in run["data"].get("tags", [])}
        if "mlflow.parentRunId" not in tags:
            parent = info
            continue
        round_tag = tags.get("round")
        if round_tag is None:
            continue
        params = {p["key"]: p["value"] for p in run["data"].get("params", [])}
        metrics = {m["key"]: m["value"] for m in run["data"].get("metrics", [])}
        # A resumed campaign starts a fresh parent run and restarts round
        # numbering from 1; runs arrive oldest-first (order_by above), so
        # this keeps the most recent attempt at each round number.
        rounds_by_idx[int(round_tag)] = {
            "round": int(round_tag),
            "n_points": int(params.get("n_points", 0)),
            "metrics": metrics,
        }

    rounds = [rounds_by_idx[k] for k in sorted(rounds_by_idx)]
    if not rounds:
        return {"available": False}
    mlflow_url = f"{_base_uri()}/#/experiments/{exp_id}/runs/{parent['run_id']}" \
        if parent is not None else f"{_base_uri()}/#/experiments/{exp_id}"
    return {"available": True, "mlflow_url": mlflow_url, "rounds": rounds}
