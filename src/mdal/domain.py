"""The input domain the campaign explores.

Stage 1 default: supercritical LJ, T* in [1.35, 3.0], rho* in [0.05, 0.9]
(§3.2 — deliberately excludes the coexistence dome where the observable
stops being a smooth function of the inputs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Domain:
    """Axis-aligned box over named continuous inputs."""

    names: tuple[str, ...] = ("temperature", "density")
    lows: tuple[float, ...] = (1.35, 0.05)
    highs: tuple[float, ...] = (3.0, 0.9)

    @property
    def ndim(self) -> int:
        return len(self.names)

    def contains(self, x: np.ndarray) -> np.ndarray:
        lo = np.asarray(self.lows)
        hi = np.asarray(self.highs)
        x = np.atleast_2d(x)
        return np.all((x >= lo) & (x <= hi), axis=1)

    def grid(self, n_per_axis: int = 60) -> np.ndarray:
        """Dense evaluation grid for the ground-truth error map (§3.5b)."""
        axes = [np.linspace(lo, hi, n_per_axis) for lo, hi in zip(self.lows, self.highs)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)
