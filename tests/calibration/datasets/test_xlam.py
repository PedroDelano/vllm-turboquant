"""Tests for XLAMAdapter."""

from __future__ import annotations

import json

import pytest

from calibration.datasets.base import DatasetAdapter
from calibration.datasets.xlam import XLAMAdapter


_SAMPLE_ROWS = [
    {
        "query": "What is the weather in Paris?",
        "tools": json.dumps([
            {
                "name": "get_weather",
                "description": "get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]),
        "answers": json.dumps([
            {"name": "get_weather", "arguments": {"city": "Paris"}}
        ]),
    },
    {
        "query": "",  # malformed — should be skipped
        "tools": "[]",
        "answers": "[]",
    },
    {
        "query": "Multi-call example",
        "tools": json.dumps([
            {"name": "a", "parameters": {}},
            {"name": "b", "parameters": {}},
        ]),
        "answers": json.dumps([
            {"name": "a", "arguments": {}},
            {"name": "b", "arguments": {}},
        ]),
    },
]


@pytest.fixture
def patched_load_dataset(monkeypatch):
    def fake_load_dataset(dataset_id, split, streaming):
        assert dataset_id == XLAMAdapter.dataset_id
        assert streaming is True
        return iter(_SAMPLE_ROWS)

    import calibration.datasets.xlam as m

    monkeypatch.setattr(m, "load_dataset", fake_load_dataset)


def test_xlam_adapter_implements_protocol():
    assert isinstance(XLAMAdapter(), DatasetAdapter)


def test_xlam_adapter_yields_structured_tools(patched_load_dataset):
    adapter = XLAMAdapter()
    convs = list(adapter.iter_conversations(split="train"))
    assert len(convs) == 2

    first = convs[0]
    assert first.tools is not None
    assert len(first.tools) == 1
    assert first.tools[0]["name"] == "get_weather"
    roles = [m["role"] for m in first.messages]
    assert roles == ["user", "assistant"]
    assert first.messages[0]["content"] == "What is the weather in Paris?"
    assert "get_weather" in first.messages[1]["content"]


def test_xlam_adapter_handles_multiple_tools_and_calls(patched_load_dataset):
    adapter = XLAMAdapter()
    convs = list(adapter.iter_conversations(split="train"))
    second = convs[1]
    assert len(second.tools) == 2
    assert '"name": "a"' in second.messages[1]["content"]
    assert '"name": "b"' in second.messages[1]["content"]
