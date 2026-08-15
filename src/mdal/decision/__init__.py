"""Decision layer: choose the next query. Registry of acquisition strategies."""

from mdal.decision.alc_imse import IntegratedVarianceReduction
from mdal.decision.base import Acquisition
from mdal.decision.epistemic import EpistemicVariance
from mdal.decision.latin_hypercube import LatinHypercube
from mdal.decision.max_variance import MaxVariance

# Name -> strategy, so a campaign TOML can select by string.
REGISTRY = {
    cls.name: cls
    for cls in (LatinHypercube, MaxVariance, EpistemicVariance, IntegratedVarianceReduction)
}

__all__ = [
    "Acquisition",
    "LatinHypercube",
    "MaxVariance",
    "EpistemicVariance",
    "IntegratedVarianceReduction",
    "REGISTRY",
]
