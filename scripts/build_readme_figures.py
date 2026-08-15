"""Render static PNG versions of the project figures for the README.

Reads the same docs/figdata.js the interactive page uses, so the static and
interactive figures always agree. Writes into docs/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "savefig.dpi": 130, "figure.dpi": 130, "font.family": "serif", "font.size": 10,
    "axes.edgecolor": "#c9ced3", "axes.linewidth": 0.8, "axes.titlesize": 11,
    "axes.grid": True, "grid.color": "#e7eaec", "grid.linewidth": 0.7,
    "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.facecolor": "white",
})
CAT = {"latin_hypercube": "#0072B2", "max_variance": "#D55E00",
       "epistemic": "#009E73", "alc_imse": "#CC79A7"}
NAME = {"latin_hypercube": "latin hypercube", "max_variance": "max variance",
        "epistemic": "epistemic", "alc_imse": "alc / imse"}
OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

raw = Path("docs/figdata.js").read_text()
F = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])


def lj_potential():
    r = np.linspace(0.95, 3.0, 400)
    u = 4 * (r ** -12 - r ** -6)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(r, u, color="#14607a", lw=2)
    ax.axhline(0, color="#c9ced3", lw=0.8)
    ax.axvline(2.5, color="#5f6772", lw=1, ls=":")
    ax.plot(2 ** (1 / 6), -1, "o", color="#14607a")
    ax.annotate("min (2$^{1/6}$σ, −ε)", (2 ** (1 / 6), -1), (1.35, -0.6), color="#5f6772", fontsize=9,
                arrowprops=dict(arrowstyle="-", color="#c9ced3"))
    ax.annotate("cutoff 2.5σ", (2.5, 2.3), color="#5f6772", fontsize=9)
    ax.set(xlim=(0.9, 3.0), ylim=(-1.3, 3), xlabel="separation  r / σ", ylabel="U / ε")
    fig.savefig(OUT / "lj_potential.png"); plt.close(fig)


def kn_surface():
    T, R, P = np.array(F["kn"]["T"]), np.array(F["kn"]["rho"]), np.array(F["kn"]["P"])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    cf = ax.contourf(T, R, P, levels=18, cmap="viridis")
    cs = ax.contour(T, R, P, levels=8, colors="white", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    ax.set(xlabel="T*", ylabel="ρ*")
    fig.colorbar(cf, ax=ax, label="P*")
    ax.grid(False)
    fig.savefig(OUT / "kn_surface.png"); plt.close(fig)


def trajectories():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    for ax, key, c in [(axes[0], "traj_quiet", "#3d6b7d"), (axes[1], "traj_noisy", "#c0552a")]:
        d = F[key]
        ax.plot(d["time"], d["P"], color=c, lw=0.8)
        ax.axvline(d["t0_time"], color="#5f6772", lw=1, ls="--")
        ax.axhline(d["mean"], color="#111", lw=1)
        ax.set(xlabel="reduced time", ylabel="P*",
               title=f"ρ*={d['rho']}   σ(mean)={d['sigma']:.4f}")
    fig.tight_layout()
    fig.savefig(OUT / "trajectories.png"); plt.close(fig)


def acquisition():
    ns = np.array(F["acq"]["ns"][1:])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for k in ["latin_hypercube", "max_variance", "epistemic", "alc_imse"]:
        m = np.array(F["acq"]["series"][k]["mean"][1:])
        s = np.array(F["acq"]["series"][k]["std"][1:])
        ax.fill_between(ns, m - s, m + s, color=CAT[k], alpha=0.13, lw=0)
        ax.plot(ns, m, "-o", color=CAT[k], ms=4, lw=2, label=NAME[k])
    ax.set_yscale("log")
    ax.set(xlabel="number of simulations", ylabel="∫ | surrogate − KN |")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    fig.savefig(OUT / "acquisition_comparison.png"); plt.close(fig)


def error_maps():
    m = max(np.abs(np.array(F[k]["err"])).max() for k in ("err_lhs", "err_alc"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for ax, key, title in [(axes[0], "err_lhs", "Latin hypercube"),
                           (axes[1], "err_alc", "ALC / IMSE")]:
        E = F[key]
        pc = ax.pcolormesh(E["T"], E["rho"], np.array(E["err"]), cmap="RdBu_r",
                           vmin=-m, vmax=m, shading="auto")
        ax.scatter(E["sx"], E["sy"], s=9, c="k", edgecolors="w", linewidths=0.4)
        ax.set(xlabel="T*", ylabel="ρ*", title=f"{title}  (|err| {E['mae']:.4f})")
        ax.grid(False)
        fig.colorbar(pc, ax=ax, label="surrogate − KN")
    fig.tight_layout()
    fig.savefig(OUT / "error_maps.png"); plt.close(fig)


def length_map():
    d = F["lengths"]
    fig, ax = plt.subplots(figsize=(5.9, 4.2))
    sc = ax.scatter(d["T"], d["rho"], c=d["length"], cmap="viridis", s=48,
                    edgecolors="w", linewidths=0.4)
    ax.set(xlabel="T*", ylabel="ρ*", xlim=(1.35, 3.0), ylim=(0.05, 0.9))
    ax.grid(False)
    fig.colorbar(sc, ax=ax, label="chosen production length (steps)")
    fig.savefig(OUT / "length_map.png"); plt.close(fig)


for fn in (lj_potential, kn_surface, trajectories, acquisition, error_maps, length_map):
    fn()
    print("wrote", fn.__name__)
print("done ->", OUT)
