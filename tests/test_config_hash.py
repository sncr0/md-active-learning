"""RunConfig content-hash contract — the store key must be stable and sensitive."""

from mdal.config import RunConfig


def test_hash_is_deterministic():
    a = RunConfig(temperature=1.5, density=0.3)
    b = RunConfig(temperature=1.5, density=0.3)
    assert a.content_hash() == b.content_hash()


def test_hash_changes_with_any_physical_field():
    base = RunConfig(temperature=1.5, density=0.3)
    variants = [
        RunConfig(temperature=1.6, density=0.3),
        RunConfig(temperature=1.5, density=0.31),
        RunConfig(temperature=1.5, density=0.3, n_steps=200_000),  # LENGTH matters
        RunConfig(temperature=1.5, density=0.3, seed=1),
        RunConfig(temperature=1.5, density=0.3, cutoff=3.0),
        RunConfig(temperature=1.5, density=0.3, thermostat="nose_hoover"),
    ]
    base_h = base.content_hash()
    for v in variants:
        assert v.content_hash() != base_h


def test_hash_is_short_and_hex():
    h = RunConfig(temperature=2.0, density=0.5).content_hash()
    assert len(h) == 16
    int(h, 16)  # raises if not hex


def test_frozen_is_hashable():
    # frozen dataclass -> usable directly in sets/dict keys
    {RunConfig(temperature=1.5, density=0.3)}
