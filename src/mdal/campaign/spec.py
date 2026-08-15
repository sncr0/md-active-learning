"""CampaignSpec — the declarative description of ONE experiment.

Distinct from RunConfig (which defines one simulation). A CampaignSpec defines
domain + acquisition strategy + budget + run defaults, and is loaded from a
TOML file in configs/. `run_defaults` supplies every RunConfig field EXCEPT the
state point, which the acquisition function chooses.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from mdal.domain import Domain


@dataclass(frozen=True)
class CampaignSpec:
    name: str
    observable: str  # "pressure" | "energy" | "diffusion"
    domain: Domain
    strategy: str  # key into decision.REGISTRY
    n_initial: int  # size of the initial space-filling design
    n_total: int  # simulation budget (total runs)
    batch: int  # runs proposed per AL round
    seed: int
    run_defaults: dict = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: str | Path) -> "CampaignSpec":
        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)
        d = cfg["domain"]
        names = tuple(d.keys())
        domain = Domain(
            names=names,
            lows=tuple(d[k][0] for k in names),
            highs=tuple(d[k][1] for k in names),
        )
        return cls(
            name=cfg["name"],
            observable=cfg["observable"],
            domain=domain,
            strategy=cfg["acquisition"]["strategy"],
            n_initial=cfg["budget"]["n_initial"],
            n_total=cfg["budget"]["n_total"],
            batch=cfg["budget"].get("batch", 1),
            seed=cfg.get("seed", 0),
            run_defaults=cfg.get("run_defaults", {}),
        )
