# md-active-learning

Active learning over a molecular-dynamics oracle: learn the equation of state of a
Lennard-Jones fluid with as few simulations as possible, using **calibrated uncertainty**
to decide what to simulate next — and, cost-aware, *how long* to run each simulation.

> **Interactive version (recommended):** **https://sncr0.github.io/md-active-learning/** —
> the same write-up with interactive Plotly figures. This README mirrors it with static plots.

---

## Abstract

A molecular-dynamics (MD) simulator is an expensive, noisy oracle whose noise can be paid down
by running longer. This project treats the choice of which thermodynamic states to simulate — and
for how long — as sequential experimental design, scored against an analytic equation of state so
that error is measurable everywhere. On a supercritical Lennard-Jones fluid the naive
variance-seeking rule fails exactly as predicted; a decision-theoretic rule is the most accurate
and the most reliable; and a cost-aware policy spends long, precise runs only on the few states
that need them.

## 1. Background

An MD simulator, treated as a data source, has three awkward properties.

- **It is expensive.** Each query costs seconds to hours, so the input space cannot be swept densely.
- **It is noisy, and unevenly so.** Every observable is a finite-time average of a fluctuating,
  correlated signal; the noise varies by more than an order of magnitude across the domain.
- **The noise is a control variable.** Run longer and it shrinks as $1/\sqrt{N_\mathrm{eff}}$. So
  *how precisely* to measure a point is a decision, weighed against compute.

The last two together are the whole problem: an oracle where you choose both *where* to query and
*how precisely*. That is experimental design, not curve fitting.

## 2. The system

**2.1 Interaction.** A single-site Lennard-Jones fluid — structureless particles in a periodic box,
interacting only through one pair potential:

$$U(r) = 4\varepsilon\left[\left(\tfrac{\sigma}{r}\right)^{12} - \left(\tfrac{\sigma}{r}\right)^{6}\right]$$

The $r^{-12}$ term is repulsion, $r^{-6}$ attraction; the well sits at $r=2^{1/6}\sigma$ with depth
$\varepsilon$. Truncated at $2.5\sigma$ with analytic long-range corrections to energy and pressure.

![Lennard-Jones potential](docs/figures/lj_potential.png)

**2.2 Units and ensemble.** Reduced units set $\sigma=\varepsilon=m=1$, so the phase diagram
collapses onto two knobs: temperature T\* and density ρ\* = Nσ³/V. We hold $N=864$, $V$,
and T\* fixed — the **NVT** ensemble — with a Langevin thermostat. The cubic box has side
L = (N/ρ\*)^(1/3)·σ under periodic boundaries; particles start on an FCC lattice and melt.

**2.3 Observables.** Pressure follows from the virial — the ideal-gas term plus the average of every
pair force projected onto its separation:

$$P = \rho\,k_B T + \frac{1}{3V}\left\langle \sum_{i<j}\mathbf{r}_{ij}\cdot\mathbf{f}_{ij}\right\rangle$$

Potential energy per particle U\* comes from the same trajectory; self-diffusion D\* is a
later, harder estimator built on a mean-squared-displacement fit.

**2.4 The target.** For each observable we want the *response surface* over the (T\*, ρ\*)
plane. The reason to start with a Lennard-Jones fluid is that this surface is known analytically —
the **Kolafa–Nezbeda** equation of state, accurate to ~1% — so the surrogate's error can be
evaluated everywhere on a dense grid, not on a small held-out set.

![KN pressure surface](docs/figures/kn_surface.png)

One restriction: below the critical temperature (T\*c ≈ 1.31) an NVT box phase-separates and
pressure stops being smooth. We stay supercritical, T\* ≥ 1.35.

## 3. Noise and the estimator

A run produces a trajectory; an observable is a time average that, by ergodicity, estimates the
ensemble average. Two facts make the obvious error bar wrong. The opening transient is not drawn
from equilibrium and must be discarded at a detected cutoff; and consecutive frames are correlated,
so $M$ frames are worth far fewer than $M$ independent samples.

![Two trajectories](docs/figures/trajectories.png)

*Same temperature, two densities. Dashed line = detected equilibration; solid line = production
mean. The dense fluid's mean carries 15× the standard error of the dilute one — the noise is
**heteroscedastic**.*

The variance of the mean is inflated by the statistical inefficiency $g$:

$$\mathrm{Var}(\bar A)=\frac{\sigma_A^2}{N_\mathrm{eff}},\qquad N_\mathrm{eff}=\frac{M}{g},\qquad g = 1 + 2\sum_{k\ge1}\rho(k)$$

where $\rho(k)$ is the autocorrelation at lag $k$. Typically $g\approx5$–$50$; reporting
$\sigma_A/\sqrt{M}$ instead makes the error bar 2–7× too small. That number feeds the decision layer
directly, so the estimator was built and validated first, against synthetic series of known $g$.

## 4. Surrogate and acquisition

**Three layers.** The **oracle** maps a run configuration to a trajectory; the **estimator** maps a
trajectory to an observable with honest uncertainty; the **decision** layer maps all observations so
far to the next query. The surrogate never touches trajectory data.

**The surrogate** is a Gaussian process. Each datum is a noisy observation of a latent surface,
$y_i = f(\mathbf{x}_i) + \varepsilon_i$ with $\varepsilon_i\sim\mathcal{N}(0,\sigma_n^2(\mathbf{x}_i))$.
Per-point noise enters as the GP's diagonal (not a kernel term), so the predictive variance splits:

$$s_\mathrm{tot}^2(\mathbf{x}) = \underbrace{s_\mathrm{epi}^2(\mathbf{x})}_{\text{reducible}} + \underbrace{\sigma_n^2(\mathbf{x})}_{\text{irreducible}}$$

Epistemic variance shrinks with data; aleatoric variance is the measurement noise at the chosen run
length. Keeping the two apart is the whole game.

**Four ways to pick the next point:** Latin hypercube (space-filling baseline); max variance
($a=s_\mathrm{tot}^2$, expected to chase noise and fail); epistemic ($a=s_\mathrm{epi}^2$); and
ALC/IMSE (score by reduction of integrated variance *everywhere*).

## 5. Results — which acquisition learns fastest

Integrated absolute error against KN over a dense grid, mean of 3 seeds; all strategies share the
same 8-point initial design (error ≈ 0.43).

![Acquisition comparison](docs/figures/acquisition_comparison.png)

| sims | latin hypercube | max variance | epistemic | **alc / imse** |
|---:|---:|---:|---:|---:|
| 16 | **0.0201** | 0.0277 | 0.0277 | 0.1174 |
| 24 | **0.0126** | 0.0138 | 0.0139 | 0.0254 |
| 32 | 0.0123 | 0.0125 | 0.0124 | **0.0111** |
| 40 | 0.0111 | 0.0124 | 0.0108 | **0.0082** |
| 48 | 0.0093 | 0.0116 | 0.0102 | **0.0084** |

- **Max variance — the trap, confirmed.** Worst at the end and *stops improving* past ~32 runs: it
  resamples the noisiest region, where total variance is dominated by irreducible noise. Reproduced
  across all three seeds.
- **Latin hypercube — a real baseline.** Best at small budgets; on a smooth 2-D surface, space-filling
  is hard to beat.
- **ALC/IMSE — wins late, wins on consistency.** Starts worst, crosses over around 31 runs, finishes
  lowest and with the tightest band — ~4× more repeatable than LHS.
- **Epistemic — quietly solid.** Tracks LHS, ends just behind ALC.

**Where the error lives.** Almost all of it is the high-density, high-temperature corner, where
pressure is steepest. A uniform design under-resolves it; ALC concentrates there.

![Error maps](docs/figures/error_maps.png)

*A floor everyone hits:* the finite (864-particle, cutoff) simulations differ from the full-potential
EOS by a small fixed amount (~0.008). Every strategy piles up against it, so the contest is how
*fast* you reach the floor — which is why the active-learning margin here is real but modest.

## 6. Results — spending a budget, not counting runs

Stage 2 lets the policy choose each run's length $\ell$ under a fixed compute budget. Longer runs
cost more ($c_0+c_1\ell$) but return lower noise ($\sigma_n^2=K(\mathbf{x})/\ell$). Maximizing
variance-reduction-per-cost has a closed-form optimal length:

$$\ell^\star(\mathbf{x})=\sqrt{\frac{K(\mathbf{x})\,c_0}{s_\mathrm{epi}^2(\mathbf{x})\,c_1}}$$

Run longer where the noise $K$ is large; shorter where the function is already uncertain.

![Cost-aware length allocation](docs/figures/length_map.png)

Against a fixed-length baseline at equal budget, the cost-aware policy ran **56 shorter simulations**
(mean ~3,300 steps, adaptively 1,500–7,900) versus **48 at 5,000**, for slightly lower error at
slightly lower cost. Spatially it banked ~2× more production in the high-density corner than in the
quiet majority. Many cheap samples or few precise ones? *Both, placed deliberately.*

## 7. Discussion

The headline that did *not* appear is itself the result: on a smooth 2-D surface, against a strong
space-filling baseline and a finite-size floor, active learning's margin is modest. What holds
cleanly is sharper — the naive rule fails in a specific, predicted way, and the principled rule is
both the most accurate and the most reliable. The regimes where adaptivity should dominate are the
ones this benchmark lacks: higher input dimension, a localized target (e.g. tracing the
coexistence-dome boundary), and larger budgets. Cost-aware allocation is the first step there.

---

## Architecture

Three layers with narrow interfaces; the surrogate never reaches into trajectory data.

```
RunConfig ─▶ ORACLE ─▶ trajectory ─▶ ESTIMATOR ─▶ Observation(value, σ, n_eff) ─▶ STORE
                                                                                     │
                                                                    (X, y, noise_var)│
                                                                                     ▼
                                                                 SURROGATE ◀── DECISION
                                                            (heteroscedastic GP)  (where, how long)
```

```
src/mdal/
  config.py      RunConfig + content hash (the store key)      cost.py     linear cost model (Stage 2)
  domain.py      the (T*, ρ*) input domain                     records.py  boundary contracts
  oracle/        settings -> trajectory (OpenMM LJ, swappable)
  estimator/     trajectory -> observable with honest uncertainty (pymbar)
  surrogate/     heteroscedastic GP (scikit-learn)
  decision/      acquisition functions (Stage 1 + cost-aware)
  reference/     Kolafa-Nezbeda analytic EOS (ground truth)
  store/         content-hash-keyed Postgres store (resumable)
  executor/      single-writer process pool
  campaign/      the active-learning loops
  analysis/      error map + acquisition comparison
  api/           read-only dashboard API, live queries (see below)
  tracking.py    optional MLflow logging of surrogate fits (see below)
tests/           unit + integration tests        scripts/   benchmark, campaign, and figure drivers
dashboard/       React + Vite frontend for the campaign dashboard
docker/          Postgres image (docker/postgres) + MLflow tracking server (docker/mlflow)
```

## Running

```bash
docker compose up -d db                             # postgres on :5432 (schema auto-applied)
uv sync --extra md --extra surrogate --extra viz     # openmm, scikit-learn, matplotlib
uv run pytest                                        # full suite (some tests need the md extra or postgres)

uv run python scripts/benchmark_oracle.py           # per-sim cost + thread vs pool throughput
uv run python scripts/compare_acquisitions.py       # Stage 1 acquisition comparison
uv run python scripts/cost_aware_demo.py            # Stage 2 cost-aware vs fixed-length
uv run python scripts/build_figdata.py              # regenerate figure data for the docs/page
uv run python scripts/build_readme_figures.py       # regenerate the static figures in this README
```

The estimator imports `pymbar`; the oracle imports `openmm` lazily, so the core package installs and
imports without the `md` extra. Every campaign needs Postgres (`docker compose up -d db`); all runs
are on one laptop, seconds per simulation.

## Dashboard

Every campaign, ranked: a sortable/filterable list (name, strategy, status, runs, rounds, R², RMSE)
backed by a small read-only API and a React frontend. Opening a row expands it inline — domain/seed,
the surrogate's learning curves, and the individual runs grouped by AL round.

```bash
docker compose up -d db                                      # postgres, if not already running
uv sync --extra api                                          # fastapi, uvicorn
uv run python scripts/run_campaign.py configs/alc_imse.toml  # a live campaign, if you want one

uv run python scripts/run_api.py                             # dashboard API on :8000
cd dashboard && npm install && npm run dev                   # frontend on :5173 (proxies /api -> :8000)
```

## Experiment tracking (MLflow)

The Postgres store is simulation *provenance* — content-hash-keyed, "was this exact state point
already run". It deliberately says nothing about how well the surrogate is learning. That's a
separate, genuinely ML concern — kernel hyperparameters, log-marginal-likelihood, R² and RMSE against
the analytic reference EOS, the model's own epistemic uncertainty, round over round — and it's what
`mdal.tracking` logs to MLflow.

```bash
docker compose up -d db mlflow                                # postgres + mlflow tracking server on :5001
uv sync --extra mlflow                                        # mlflow client
uv run python scripts/run_campaign.py configs/alc_imse.toml   # logs automatically while it runs
```

Then open `http://localhost:5001`. One experiment, `mdal-campaigns`, holds every campaign as a
top-level run (tagged `campaign_id`/`strategy`/`observable`, so the run list is directly comparable
across strategies — the same comparison `scripts/compare_acquisitions.py` does offline); each
active-learning round's surrogate refit is a nested run underneath it. Deliberately leans on what
MLflow already renders natively, rather than only logging numbers into a table:

- **The campaign's own "Model metrics" tab is a live learning curve** — every round's scores
  (R², RMSE, mean epistemic std, log-marginal-likelihood) are logged a second time onto the *parent*
  run with `step=round`, which MLflow auto-plots as an interactive line chart with zero extra
  tooling, right on the run you'd open anyway.
- **Each round's Artifacts tab has an error-map image** — surrogate-minus-reference over the domain,
  sample points overlaid (`mdal.analysis.error_map_figure`, the same plot `compare_acquisitions.py`
  saves to disk, logged instead via `mlflow.log_figure`) — where the surrogate is wrong and where
  it's looked, browsable round by round without leaving MLflow.
- **The exact fitted GP itself** is logged as a real MLflow model (skops-serialized
  `GaussianProcessRegressor`), downloadable from that round's run page.

The dashboard (see above) surfaces the headline version of this without leaving it: each campaign's
expanded row has a "Surrogate learning" panel — R²/RMSE (predictive accuracy) and mean-epistemic-std/
log-marginal-likelihood (model diagnostics) vs. round, plus an "open in MLflow" link for the rest —
read live from MLflow's REST API by `mdal.api.mlflow_client` (stdlib `urllib`, not the `mlflow`
package, so the dashboard's `api` extra stays light). Same fails-soft contract: no tracking data for
a campaign just means that panel is absent, never an error.

Tracking is entirely optional and fails soft, the same way the oracle degrades without the `md`
extra: without `mlflow` installed, or if the tracking server at `MLFLOW_TRACKING_URI` (default
`http://localhost:5001`) isn't reachable, `mdal.tracking` no-ops and campaigns behave exactly as
they did before it existed — a campaign never fails because MLflow is down. MLflow's own schema
lives in a separate `mlflow` database on the same Postgres server (`docker/postgres/00-create-mlflow-db.sql`),
never mixed into `mdal`'s tables. Artifacts are proxied through the tracking server
(`--default-artifact-root=mlflow-artifacts:/`) rather than written to a plain path — campaigns run
on the host, not inside the `mlflow` container, so the client has no direct filesystem access to the
`mdal_mlruns` volume and needs the HTTP round-trip; `--artifacts-destination` is where the server
actually puts the bytes once it's proxying (local-disk for now — swap it for an S3/MinIO URI in
`docker/mlflow` if this stops being a single-machine setup).

Campaigns that ran before `mdal.tracking` existed have no MLflow history by default — nothing was
rerun retroactively. `scripts/backfill_mlflow.py` fills that in without resimulating anything: it
refits the surrogate on successive prefixes of each campaign's already-stored observations (same
round boundaries the dashboard already uses) and logs the result with the run's *actual* historical
timestamps, not "now". Run it with no arguments to backfill every untracked campaign, shortest
average simulation time first (`uv run python scripts/backfill_mlflow.py`), or name specific
campaign IDs to backfill just those.

**Why not put the oracle runs in MLflow too:** MLflow runs aren't content-addressed, so they can't
give you the store's core property — "does this exact simulation already exist, skip it if so".
The two systems track different things and are joined by `campaign_id`, not merged.

**Why Postgres:** every campaign — not just the dashboard's — now lives in one shared database
(`runs`/`observations` keyed by `(campaign_id, run_hash)`, plus a `campaigns` table for
name/strategy/budget metadata). The dashboard API queries it directly, live, with no export step:
Postgres's MVCC lets readers see a consistent view without blocking on, or being blocked by, a
campaign that's actively writing. That's a real upgrade over the DuckDB-file store this replaced —
DuckDB takes an exclusive lock for as long as any connection is open, so a dashboard literally could
not read a live campaign's data at all; the first cut of this dashboard worked around that with a
JSON-snapshot side channel, which is gone now. The one archived script,
`scripts/migrate_duckdb_to_postgres.py`, one-time-imports the old `data/*.duckdb` result sets
(kept as a backup, never deleted) — see `configs/*.toml` for the campaigns behind the write-up.
