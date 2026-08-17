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
import time
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

    Three things happen, each in its own try/except so a failure in one (a
    plotting bug, an unpicklable kernel type) never costs the others:

    1. The round's own metrics/params on its nested run.
    2. The SAME metrics logged with step=round_idx onto the *parent* campaign
       run — MLflow renders any step-indexed metric as a native line chart on
       that run's own Metrics tab, so the campaign's learning curve is visible
       without leaving its page or comparing runs by hand.
    3. An error-map image (surrogate - reference, sample points overlaid) as
       an artifact on the round's own run — the one plot that's actually
       useful per round: where the surrogate is wrong, where it's looked.
       Plus the fitted sklearn estimator as a model artifact, if the surrogate
       exposes one (`fitted_estimator`) — a round's exact posterior is then
       retrievable from the MLflow UI/registry, not just its summary metrics.
    """
    if mlflow is None or mlflow.active_run() is None:
        return
    parent_run_id = mlflow.active_run().info.run_id
    metrics = {**getattr(surrogate, "diagnostics", dict)(), **score_vs_reference(surrogate, observable, domain)}

    if metrics:
        try:
            from mlflow.entities import Metric
            from mlflow.tracking import MlflowClient

            now_ms = int(time.time() * 1000)
            MlflowClient().log_batch(
                parent_run_id,
                metrics=[Metric(k, v, now_ms, round_idx) for k, v in metrics.items()],
            )
        except Exception as exc:
            _warn_once(f"parent trend metrics not logged: {exc}")

    try:
        with mlflow.start_run(run_name=f"round-{round_idx}", nested=True, tags={
            "campaign_id": campaign_id, "round": str(round_idx),
        }):
            mlflow.log_params({"round": round_idx, "n_points": len(X)})
            mlflow.log_metrics(metrics)

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

            try:
                from mdal.analysis._common import error_map_figure

                fig, _ = error_map_figure(surrogate, X, domain, observable)
                mlflow.log_figure(fig, "error_map.png")
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception as exc:
                _warn_once(f"error map not logged: {exc}")
    except Exception as exc:
        _warn_once(str(exc))


def score_vs_reference(surrogate, observable: str, domain) -> dict:
    """Accuracy (RMSE, R²) and confidence (mean epistemic std) of the surrogate
    against the analytic reference EOS on a dense grid.

    - rmse_vs_reference: physical units (pressure/energy) — interpretable once
      you're looking at one campaign, not comparable across observables.
    - r_squared_vs_reference: fraction of the reference surface's variance the
      surrogate explains, in [~-inf, 1]. Dimensionless, so THIS is what a
      campaigns list ranks/sorts by — RMSE in pressure units isn't comparable
      to RMSE in energy units.
    - mean_epistemic_std: the model's own confidence (not accuracy against
      ground truth) — shrinking alongside R² rising is what "the AL loop is
      working" looks like; shrinking without R² rising means it's confidently
      wrong.

    Empty dict if there's no analytic reference for this observable (e.g.
    diffusion). Shared with scripts/backfill_mlflow.py so live and backfilled
    rounds are scored identically.
    """
    from mdal.analysis._common import reference_surface

    grid = domain.grid(40)
    try:
        truth = reference_surface(observable, grid)
    except KeyError:
        return {}  # no analytic reference for this observable (e.g. diffusion)
    pred, _ = surrogate.predict(grid)
    resid = pred - truth
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "rmse_vs_reference": float(np.sqrt(np.mean(resid**2))),
        "r_squared_vs_reference": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mean_epistemic_std": float(np.mean(surrogate.epistemic_std(grid))),
    }
