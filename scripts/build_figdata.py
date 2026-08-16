"""Generate docs/figdata.js — real data for the Plotly figures on the project page.

Pulls from the existing campaign stores and runs a couple of short simulations
for illustration. Writes `window.FIGDATA = {...}` so the static page stays
self-contained apart from the Plotly/MathJax CDNs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mdal.analysis._common import integrated_abs_error
from mdal.config import RunConfig
from mdal.domain import Domain
from mdal.estimator import FrameAverageEstimator
from mdal.oracle.lennard_jones import LennardJonesOracle, _make_context
from mdal.reference import pressure
from mdal.store import PostgresStore
from mdal.surrogate import HeteroscedasticGP

DOMAIN = Domain()
STRATEGIES = ["latin_hypercube", "max_variance", "epistemic", "alc_imse"]
OUT = Path("docs/figdata.js")


def kn_surface(nt=46, nr=46):
    T = np.linspace(1.35, 3.0, nt)
    R = np.linspace(0.05, 0.9, nr)
    TT, RR = np.meshgrid(T, R)
    Z = pressure(TT, RR)  # z[rho_idx][T_idx]
    return {"T": T.tolist(), "rho": R.tolist(), "P": Z.tolist()}


def thermalized_snapshot(T=1.5, rho=0.6, steps=3000, slab=0.75):
    from openmm import unit

    cfg = RunConfig(temperature=T, density=rho, n_particles=864, equil_steps=0, n_steps=1)
    ctx, integ, N, L = _make_context(cfg, threads=1)
    integ.step(steps)
    pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    pos = np.mod(pos, L)
    zc = L / 2.0
    m = np.abs(pos[:, 2] - zc) < slab
    xs, ys = pos[m, 0], pos[m, 1]
    # a central particle to anchor the cutoff circle
    ci = int(np.argmin((xs - L / 2) ** 2 + (ys - L / 2) ** 2))
    return {"x": xs.tolist(), "y": ys.tolist(), "L": float(L), "cutoff": 2.5,
            "cx": float(xs[ci]), "cy": float(ys[ci]), "T": T, "rho": rho}


def trajectory(T, rho, n_steps=12000, sample=50):
    cfg = RunConfig(temperature=T, density=rho, n_particles=864,
                    equil_steps=0, n_steps=n_steps, sample_interval=sample)
    traj = LennardJonesOracle().run(cfg)
    P = traj.per_frame["pressure"]
    from mdal.estimator.equilibration import detect_equilibration

    t0 = int(detect_equilibration(P))
    obs = FrameAverageEstimator("pressure", "pressure").estimate(traj)
    dt_frame = sample * cfg.timestep
    time = (np.arange(len(P)) * dt_frame).tolist()
    return {"T": T, "rho": rho, "time": time, "P": P.tolist(),
            "t0_time": t0 * dt_frame, "mean": obs.value, "sigma": obs.sigma, "n_eff": obs.n_eff}


def acquisition_curves(ns=(8, 16, 24, 32, 40, 48), seeds=(0, 1, 2)):
    out = {"ns": list(ns), "series": {}}
    for strat in STRATEGIES:
        rows = []
        for s in seeds:
            store = PostgresStore(f"cmp_{strat}_s{s}")
            rows.append([integrated_abs_error(store, DOMAIN, "pressure", n, 55) for n in ns])
            store.close()
        arr = np.array(rows)
        out["series"][strat] = {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist()}
    return out


def error_map(store_name, grid=52):
    store = PostgresStore(store_name)
    X, y, nv = store.observations_for("pressure")
    store.close()
    gp = HeteroscedasticGP().fit(X, y, nv)
    T = np.linspace(1.35, 3.0, grid)
    R = np.linspace(0.05, 0.9, grid)
    TT, RR = np.meshgrid(T, R)
    pts = np.column_stack([TT.ravel(), RR.ravel()])
    pred, _ = gp.predict(pts)
    err = (pred - pressure(pts[:, 0], pts[:, 1])).reshape(grid, grid)
    return {"T": T.tolist(), "rho": R.tolist(), "err": err.tolist(),
            "sx": X[:, 0].tolist(), "sy": X[:, 1].tolist(),
            "mae": float(np.mean(np.abs(err)))}


def length_map(store_name="cost_cost_aware_s0"):
    store = PostgresStore(store_name)
    X = store.observations_for("pressure")[0]
    L = store.lengths_for("pressure")
    store.close()
    return {"T": X[:, 0].tolist(), "rho": X[:, 1].tolist(), "length": L.tolist()}


def main():
    print("kn surface...", flush=True)
    data = {"kn": kn_surface()}
    print("snapshot...", flush=True)
    data["snapshot"] = thermalized_snapshot()
    print("trajectories...", flush=True)
    # same T*, different density -> isolates the density -> noise effect
    data["traj_quiet"] = trajectory(2.0, 0.15)   # dilute: small absolute pressure noise
    data["traj_noisy"] = trajectory(2.0, 0.85)   # dense: large absolute pressure noise
    print("acquisition curves...", flush=True)
    data["acq"] = acquisition_curves()
    print("error maps...", flush=True)
    data["err_lhs"] = error_map("cmp_latin_hypercube_s0")
    data["err_alc"] = error_map("cmp_alc_imse_s0")
    print("length map...", flush=True)
    data["lengths"] = length_map()

    OUT.write_text("window.FIGDATA = " + json.dumps(data) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
