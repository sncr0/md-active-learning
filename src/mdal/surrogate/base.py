"""Surrogate interface — the model the decision layer and analysis both consume.

Kept separate from `decision/` because BOTH the acquisition functions and the
error-map analysis use a fitted surrogate.

The one non-negotiable capability (§3.6.3): it must accept PER-POINT noise
variance and expose the EPISTEMIC (latent-function) variance separately from
the total predictive variance. A single global noise level (sklearn
WhiteKernel) averages away exactly the heteroscedastic structure that drives
the interesting acquisition behaviour.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Surrogate(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, noise_var: np.ndarray) -> None:
        """Fit with a per-point observation-noise variance vector."""
        ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, total_std) including observation noise."""
        ...

    def epistemic_std(self, X: np.ndarray) -> np.ndarray:
        """Latent-function std ONLY, excluding observation noise.

        This is what a non-naive acquisition function should query.
        """
        ...
