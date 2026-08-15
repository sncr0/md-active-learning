"""HANDOFF NOTE #1 (§8): measure the LJ oracle's wall-clock on THIS machine.

Run before any scheduling/budget decision. Reports:
  (a) single-run throughput vs OpenMM CPU thread count — small systems scale
      poorly, so more threads buys little;
  (b) aggregate throughput of N independent 1-thread runs in a process pool —
      the model §8 predicts wins for campaign throughput.

Verify, don't trust: the recommendation prints from measured numbers.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

from mdal.config import RunConfig
from mdal.oracle.lennard_jones import _make_context

STATE = dict(temperature=1.5, density=0.3, n_particles=864)
STEPS = 20_000
WARMUP = 500


def _throughput(threads: int, steps: int = STEPS) -> float:
    cfg = RunConfig(**STATE, equil_steps=0, n_steps=steps)
    context, integrator, _N, _L = _make_context(cfg, threads=threads, minimize=False)
    integrator.step(WARMUP)
    t = time.perf_counter()
    integrator.step(steps)
    return steps / (time.perf_counter() - t)


def _pool_worker(seed: int) -> float:
    return _throughput(threads=1, steps=STEPS)  # seed varies the run; timing is what we want


def main() -> None:
    print(f"System: N={STATE['n_particles']}, T*={STATE['temperature']}, "
          f"rho*={STATE['density']}, cutoff=2.5\n")

    print("(a) single run, thread scaling")
    print(f"    {'threads':>7}  {'ksteps/s':>9}  {'s / 1e5 steps':>13}")
    base = None
    for th in (1, 2, 4, 8):
        r = _throughput(th)
        base = base or r
        print(f"    {th:>7}  {r/1e3:>9.1f}  {1e5/r:>13.2f}   ({r/base:.2f}x)")

    print("\n(b) process pool, 1 thread each (aggregate throughput)")
    print(f"    {'procs':>7}  {'agg ksteps/s':>12}  {'speedup':>7}")
    single = None
    for p in (1, 4, 8):
        t = time.perf_counter()
        with ProcessPoolExecutor(max_workers=p) as ex:
            list(ex.map(_pool_worker, range(p)))
        wall = time.perf_counter() - t
        agg = p * STEPS / wall
        single = single or agg
        print(f"    {p:>7}  {agg/1e3:>12.1f}  {agg/single:>6.2f}x")

    print("\nRule: set OPENMM_CPU_THREADS=1 and parallelise across independent runs.")


if __name__ == "__main__":
    main()
