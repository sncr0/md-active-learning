"""MLflow experiment tracking for the surrogate's fits, round over round.

Separate concern from `mdal.store`: the Postgres store is content-hash-
addressed simulation cache — oracle-layer provenance, "was this exact state
point already run". This tracks the ML side instead — how the surrogate's fit
quality evolves as a campaign progresses, so strategies become comparable in
the MLflow UI the way `scripts/compare_acquisitions.py` already compares them
ad hoc.

Entirely optional and fails soft: without the `mlflow` extra installed, or if
the tracking server isn't reachable, every function here is a no-op and
campaigns behave exactly as they did before this module existed. A campaign
must never fail because MLflow is down.

One MLflow experiment ("mdal-campaigns") holds every campaign as a top-level
run (tagged campaign_id/strategy/observable, params mirroring the `campaigns`
Postgres table); each AL round's surrogate fit is a nested run underneath it.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager

import numpy as np

EXPERIMENT = "mdal-campaigns"
URI_ENV = "MLFLOW_TRACKING_URI"
DEFAULT_URI = "http://localhost:5001"

# The default surrogate kernel (ConstantKernel * RBF, see surrogate/gp.py) — skops
# refuses to (de)serialize sklearn types it doesn't recognize as trusted by default.
# These are ordinary sklearn built-ins produced entirely by our own fit() code, not
# untrusted input, so trusting them here is safe. Shared with scripts/backfill_mlflow.py.
TRUSTED_KERNEL_TYPES = [
    "sklearn.gaussian_process.kernels.ConstantKernel",
    "sklearn.gaussian_process.kernels.Product",
    "sklearn.gaussian_process.kernels.RBF",
]

try:
    import mlflow
except ImportError:
    mlflow = None

_warned = False


def _warn_once(msg: str) -> None:
    global _warned
    if not _warned:
        warnings.warn(f"mlflow tracking disabled: {msg}", stacklevel=3)
        _warned = True


@contextmanager
def campaign_run(campaign_id: str, *, strategy: str, observable: str, params: dict):
    """Wrap one campaign's rounds in a top-level MLflow run; call `log_round`
    inside the block for each surrogate fit. No-ops (never raises) if mlflow
    isn't installed or the tracking server can't be reached."""
    active = False
    if mlflow is not None:
        try:
            mlflow.set_tracking_uri(os.environ.get(URI_ENV, DEFAULT_URI))
            mlflow.set_experiment(EXPERIMENT)
            mlflow.start_run(
                run_name=campaign_id,
                tags={"campaign_id": campaign_id, "strategy": strategy, "observable": observable},
            )
            mlflow.log_params(params)
            active = True
        except Exception as exc:  # server down, auth, etc. — degrade, don't block the campaign
            _warn_once(str(exc))
    try:
        yield
    finally:
        if active:
            try:
                mlflow.end_run()
            except Exception:
                pass


def log_round(
    round_idx: int, surrogate, X: np.ndarray, observable: str, domain, campaign_id: str
) -> None:
    """Log one AL round's surrogate fit as a nested run under the active campaign run.

    Also logs the fitted sklearn estimator as a model artifact, if the surrogate
    exposes one (`fitted_estimator`) — so a round's exact posterior is retrievable
    from the MLflow UI/registry, not just its summary metrics.
    """
    if mlflow is None or mlflow.active_run() is None:
        return
    try:
        with mlflow.start_run(run_name=f"round-{round_idx}", nested=True, tags={
            "campaign_id": campaign_id, "round": str(round_idx),
        }):
            mlflow.log_params({"round": round_idx, "n_points": len(X)})
            mlflow.log_metrics(getattr(surrogate, "diagnostics", dict)())
            rmse = rmse_vs_reference(surrogate, observable, domain)
            if rmse is not None:
                mlflow.log_metric("rmse_vs_reference", rmse)
            # Model logging is its own try/except: a serialization failure here
            # (e.g. skops rejecting an unrecognized kernel type) must never cost
            # the metrics above, which are already committed by this point.
            estimator = getattr(surrogate, "fitted_estimator", None)
            if estimator is not None:
                try:
                    import mlflow.sklearn as mlflow_sklearn  # local import shadows `mlflow` else

                    mlflow_sklearn.log_model(
                        estimator, name="model", skops_trusted_types=TRUSTED_KERNEL_TYPES,
                    )
                except Exception as exc:
                    _warn_once(f"model artifact not logged: {exc}")
    except Exception as exc:
        _warn_once(str(exc))


def rmse_vs_reference(surrogate, observable: str, domain) -> float | None:
    """RMSE of the surrogate against the analytic reference EOS on a dense grid.

    None if there's no analytic reference for this observable (e.g. diffusion).
    Shared with scripts/backfill_mlflow.py so live and backfilled rounds are
    scored identically.
    """
    from mdal.analysis._common import reference_surface

    grid = domain.grid(40)
    try:
        truth = reference_surface(observable, grid)
    except KeyError:
        return None  # no analytic reference for this observable (e.g. diffusion)
    pred, _ = surrogate.predict(grid)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))
