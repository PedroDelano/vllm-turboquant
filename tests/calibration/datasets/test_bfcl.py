"""Tests for BFCLAdapter."""

from __future__ import annotations

import pytest

from calibration.datasets.base import DatasetAdapter
from calibration.datasets.bfcl import BFCLAdapter


_FUNCTION = [
    {
        "name": "calculate_area",
        "description": "Calculate the area of a shape.",
        "parameters": {
            "type": "object",
            "properties": {
                "shape": {"type": "string"},
                "size": {"type": "number"},
            },
            "required": ["shape", "size"],
        },
    }
]

_SAMPLE_ROWS = [
    {
        "id": "simple_0",
        "question": [[{"role": "user", "content": "Area of a circle with radius 3?"}]],
        "function": _FUNCTION,
    },
    {
        "id": "simple_1",
        "question": [[{"role": "user", "content": "Area of a square with side 5?"}]],
        "function": _FUNCTION,
    },
    {
        # malformed — empty question
        "id": "bad",
        "question": [],
        "function": _FUNCTION,
    },
]


@pytest.fixture
def patched_load_dataset(monkeypatch):
    captured: dict[str, object] = {}

    def fake_load_dataset(dataset_id, *, name, split, streaming):
        captured["dataset_id"] = dataset_id
        captured["name"] = name
        captured["split"] = split
        captured["streaming"] = streaming
        return iter(_SAMPLE_ROWS)

    import calibration.datasets.bfcl as m

    monkeypatch.setattr(m, "load_dataset", fake_load_dataset)
    return captured


def test_bfcl_adapter_implements_protocol():
    assert isinstance(BFCLAdapter(), DatasetAdapter)


def test_bfcl_adapter_passes_subset_to_load_dataset(patched_load_dataset):
    adapter = BFCLAdapter(subset="multiple")
    list(adapter.iter_conversations(split="train"))
    assert patched_load_dataset["name"] == "multiple"


def test_bfcl_adapter_default_subset_is_simple(patched_load_dataset):
    adapter = BFCLAdapter()
    list(adapter.iter_conversations(split="train"))
    assert patched_load_dataset["name"] == "simple"


def test_bfcl_adapter_flattens_nested_question(patched_load_dataset):
    adapter = BFCLAdapter()
    convs = list(adapter.iter_conversations(split="train"))
    assert len(convs) == 2
    roles = [m["role"] for m in convs[0].messages]
    assert roles == ["user"]
    assert convs[0].messages[0]["content"] == "Area of a circle with radius 3?"
    assert convs[0].tools == _FUNCTION
