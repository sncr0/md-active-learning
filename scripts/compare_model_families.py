"""Compare classical-ML model families against the GP, on the SAME pooled
AL-campaign data (no new simulations) — a first pass at "does model family
matter, holding the training set fixed?" Separate axis from
scripts/compare_acquisitions.py, which holds the model family fixed (GP) and
varies the acquisition strategy instead.

Pools observations across the six real campaigns (mdal.analysis
.model_comparison.REAL_CAMPAIGNS) per observable, fits every family in
model_families(), scores each against the Kolafa-Nezbeda reference the same
way mdal.tracking scores AL rounds, and logs one MLflow run per
(observable, family) to a separate "mdal-model-comparison" experiment
(docker compose up -d mlflow) — best-effort, never blocks the comparison if
the tracking server is unreachable.

Usage: python scripts/compare_model_families.py [observable ...]
       (default: pressure energy)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

from mdal.analysis._common import error_map_figure
from mdal.analysis.model_comparison import REAL_CAMPAIGNS, model_families, pooled_observations
from mdal.domain import Domain
from mdal.tracking import DEFAULT_URI, URI_ENV, score_vs_reference

EXPERIMENT = "mdal-model-comparison"

try:
    import mlflow
except ImportError:
    mlflow = None


def _log(observable: str, name: str, n_points: int, metrics: dict, fig) -> None:
    if mlflow is None:
        return
    try:
        mlflow.set_tracking_uri(os.environ.get(URI_ENV, DEFAULT_URI))
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(
            run_name=f"{observable}-{name}",
            tags={"observable": observable, "model_family": name},
        ):
            mlflow.log_params({"model_family": name, "observable": observable, "n_points": n_points})
            mlflow.log_metrics(metrics)
            if fig is not None:
                mlflow.log_figure(fig, "error_map.png")
    except Exception as exc:  # server down, auth, etc. — never block the comparison
        warnings.warn(f"mlflow logging skipped: {exc}", stacklevel=2)


def main(observables=("pressure", "energy")):
    domain = Domain()
    Path("results").mkdir(exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for observable in observables:
        X, y, noise_var = pooled_observations(observable)
        print(f"\n=== {observable}: {len(X)} pooled observations across "
              f"{len(REAL_CAMPAIGNS)} campaigns ===")

        rows = []
        for name, model in model_families().items():
            model.fit(X, y, noise_var)
            metrics = score_vs_reference(model, observable, domain)
            rows.append((name, metrics))

            fig = None
            try:
                fig, _ = error_map_figure(model, X, domain, observable)
                fig.savefig(f"results/errmap_{observable}_{name}.png", dpi=130, bbox_inches="tight")
            except Exception as exc:
                warnings.warn(f"error map skipped for {name}: {exc}")
            _log(observable, name, len(X), metrics, fig)
            if fig is not None:
                plt.close(fig)

        rows.sort(key=lambda r: r[1].get("r_squared_vs_reference", float("-inf")), reverse=True)
        print(f"  {'family':<20} {'R^2':>8} {'RMSE':>10}")
        for name, metrics in rows:
            print(f"  {name:<20} {metrics['r_squared_vs_reference']:>8.4f} "
                  f"{metrics['rmse_vs_reference']:>10.4f}")


if __name__ == "__main__":
    obs = tuple(sys.argv[1:]) or ("pressure", "energy")
    main(obs)
