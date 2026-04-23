"""Tests for ToolACEAdapter.

Fixture covers both supported conversation shapes:
  - dict with `role`/`content`
  - dict with `from`/`value` (sharegpt-style)
Also covers ToolACE's non-standard roles (`function_call`, `observation`)
which the adapter normalizes to standard roles.
"""

from __future__ import annotations

import pytest

from calibration.datasets.base import DatasetAdapter
from calibration.datasets.toolace import ToolACEAdapter


_TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Look up a stock price",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    }
]

_SAMPLE_ROWS = [
    {
        "system": "You are a helpful assistant.",
        "tools": _TOOLS,
        "conversations": [
            {"role": "user", "content": "What is AAPL at?"},
            {
                "role": "function_call",
                "content": '{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}',
            },
            {"role": "observation", "content": '{"price": 175.3}'},
            {"role": "assistant", "content": "AAPL is at 175.3."},
        ],
    },
    {
        # sharegpt-style with from/value keys.
        "system": "",
        "tools": _TOOLS,
        "conversations": [
            {"from": "user", "value": "price of MSFT"},
            {"from": "assistant", "value": "calling tool"},
        ],
    },
    {
        # malformed — no conversations
        "system": "",
        "tools": _TOOLS,
        "conversations": [],
    },
]


@pytest.fixture
def patched_load_dataset(monkeypatch):
    def fake_load_dataset(dataset_id, split, streaming):
        assert dataset_id == ToolACEAdapter.dataset_id
        assert streaming is True
        return iter(_SAMPLE_ROWS)

    import calibration.datasets.toolace as m

    monkeypatch.setattr(m, "load_dataset", fake_load_dataset)


def test_toolace_adapter_implements_protocol():
    assert isinstance(ToolACEAdapter(), DatasetAdapter)


def test_toolace_adapter_normalizes_nonstandard_roles(patched_load_dataset):
    adapter = ToolACEAdapter()
    convs = list(adapter.iter_conversations(split="train"))
    assert len(convs) == 2

    first = convs[0]
    assert first.tools == _TOOLS
    roles = [m["role"] for m in first.messages]
    # system prepended, function_call->assistant, observation->tool.
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_toolace_adapter_accepts_sharegpt_keys(patched_load_dataset):
    adapter = ToolACEAdapter()
    convs = list(adapter.iter_conversations(split="train"))
    second = convs[1]
    assert [m["role"] for m in second.messages] == ["user", "assistant"]
    assert second.messages[0]["content"] == "price of MSFT"
