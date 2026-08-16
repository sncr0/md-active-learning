"""Wiring tests: the full import graph loads, protocols are satisfied, and
campaign specs parse. No physics — just that the machinery holds together."""

import importlib
import pkgutil

import mdal
from mdal.campaign import CampaignSpec
from mdal.decision import REGISTRY
from mdal.decision.base import Acquisition
from mdal.estimator.base import Estimator
from mdal.estimator import FrameAverageEstimator
from mdal.oracle.base import Oracle
from mdal.oracle import LennardJonesOracle
from mdal.surrogate.base import Surrogate
from mdal.surrogate import HeteroscedasticGP


def test_every_module_imports():
    # Import every submodule so a syntax/wiring error anywhere fails a test.
    for mod in pkgutil.walk_packages(mdal.__path__, prefix="mdal."):
        importlib.import_module(mod.name)


def test_protocols_are_satisfied_structurally():
    assert isinstance(LennardJonesOracle(), Oracle)
    assert isinstance(FrameAverageEstimator("pressure", "pressure"), Estimator)
    assert isinstance(HeteroscedasticGP(), Surrogate)


def test_acquisition_registry_complete():
    assert set(REGISTRY) == {"latin_hypercube", "max_variance", "epistemic", "alc_imse"}
    for cls in REGISTRY.values():
        assert isinstance(cls(), Acquisition)


def test_all_campaign_configs_parse():
    import pathlib

    configs = sorted(pathlib.Path("configs").glob("*.toml"))
    assert len(configs) == 6
    for path in configs:
        spec = CampaignSpec.from_toml(path)
        assert spec.strategy in REGISTRY
        assert spec.domain.ndim == 2
        assert spec.n_total >= spec.n_initial
        # run_defaults must NOT carry the state point (acquisition chooses it)
        assert "temperature" not in spec.run_defaults
        assert "density" not in spec.run_defaults
