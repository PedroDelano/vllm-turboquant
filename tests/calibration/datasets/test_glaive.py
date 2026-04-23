"""Tests for GlaiveAdapter.

We monkeypatch `datasets.load_dataset` to avoid network. Fixture rows
preserve the real HF column structure (`system`, `chat`) and the turn
markers (`USER:`, `ASSISTANT:`, `FUNCTION RESPONSE:`) verbatim.
"""

from __future__ import annotations

import pytest

from calibration.datasets.base import DatasetAdapter
from calibration.datasets.glaive import GlaiveAdapter


_SAMPLE_ROWS = [
    {
        "system": (
            "SYSTEM: You are a helpful assistant with access to the following "
            'functions. Use them if required - {"name": "get_weather", '
            '"description": "Get the current weather", "parameters": {}}'
        ),
        "chat": (
            "USER: What is the weather in Paris?\n\n\n"
            "ASSISTANT: <functioncall> {\"name\": \"get_weather\"}\n\n\n"
            "FUNCTION RESPONSE: {\"temp\": 20}\n\n\n"
            "ASSISTANT: It is 20 degrees in Paris."
        ),
    },
    {
        # Missing chat — should be skipped.
        "system": "SYSTEM: ...",
        "chat": "",
    },
    {
        # Missing system — should be skipped.
        "system": "",
        "chat": "USER: hi\n\n\nASSISTANT: hello",
    },
]


@pytest.fixture
def patched_load_dataset(monkeypatch):
    def fake_load_dataset(dataset_id, split, streaming):
        assert dataset_id == GlaiveAdapter.dataset_id
        assert streaming is True
        return iter(_SAMPLE_ROWS)

    import calibration.datasets.glaive as m

    monkeypatch.setattr(m, "load_dataset", fake_load_dataset)


def test_glaive_adapter_implements_protocol():
    assert isinstance(GlaiveAdapter(), DatasetAdapter)


def test_glaive_adapter_parses_multi_turn_conversation(patched_load_dataset):
    adapter = GlaiveAdapter()
    convs = list(adapter.iter_conversations(split="train"))
    assert len(convs) == 1

    conv = convs[0]
    assert conv.tools is None
    roles = [m["role"] for m in conv.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert "get_weather" in conv.messages[0]["content"]
    assert conv.messages[1]["content"] == "What is the weather in Paris?"
    assert conv.messages[3]["content"] == '{"temp": 20}'


def test_glaive_adapter_skips_malformed(patched_load_dataset):
    adapter = GlaiveAdapter()
    convs = list(adapter.iter_conversations(split="train"))
    assert len(convs) == 1
