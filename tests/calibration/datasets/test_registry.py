"""Registry invariants: keys match class.name, no duplicates, all
classes satisfy the protocol."""

from __future__ import annotations

from calibration.datasets import ADAPTERS
from calibration.datasets.base import DatasetAdapter


def test_registry_keys_match_name_attr():
    for key, cls in ADAPTERS.items():
        assert cls.name == key, f"{cls.__name__}.name={cls.name!r} != key {key!r}"


def test_registry_has_no_duplicate_names():
    names = [cls.name for cls in ADAPTERS.values()]
    assert len(names) == len(set(names))


def test_all_registered_adapters_satisfy_protocol():
    for cls in ADAPTERS.values():
        instance = cls()
        assert isinstance(instance, DatasetAdapter)


def test_registry_contains_expected_adapters():
    assert set(ADAPTERS.keys()) == {"glaive", "xlam", "toolace", "bfcl"}
