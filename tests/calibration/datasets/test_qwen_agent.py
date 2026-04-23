"""Tests for QwenAgentAdapter.

Fixture mirrors the real HF schema: messages are already in OpenAI
chat format (role + content), with `tool_calls` populated on
assistant turns that fire tools and `content` None in that case.
`tools` ships as a JSON-encoded string of the OpenAI function-calling
format.
"""

from __future__ import annotations

import json

import pytest

from calibration.datasets.base import DatasetAdapter
from calibration.datasets.qwen_agent import QwenAgentAdapter


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_client_by_email",
            "description": "Look up a client by their email.",
            "parameters": {
                "type": "object",
                "properties": {"email": {"type": "string"}},
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

_SAMPLE_ROWS = [
    {
        "id": "10024_t0",
        "domain": "bank",
        "num_turns": 4,
        "tools": json.dumps(_TOOLS),
        "messages": [
            {
                "role": "system",
                "content": "You are a customer service agent.",
                "reasoning_content": "",
                "tool_call_id": None,
                "tool_calls": None,
            },
            {
                "role": "user",
                "content": "I need to unfreeze my account.",
                "reasoning_content": "",
                "tool_call_id": None,
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I need to look up the client first.",
                "tool_call_id": None,
                "tool_calls": [
                    {
                        "function": {
                            "arguments": '{"email": "user@example.com"}',
                            "name": "find_client_by_email",
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"client_id": "12345"}',
                "reasoning_content": "",
                "tool_call_id": "call_1",
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": "I've located your account.",
                "reasoning_content": "",
                "tool_call_id": None,
                "tool_calls": None,
            },
        ],
    },
    {
        # malformed — empty messages
        "id": "bad",
        "domain": "x",
        "tools": "[]",
        "messages": [],
    },
    {
        # malformed tools JSON — still parseable messages, tools=None
        "id": "nodtools",
        "domain": "x",
        "tools": "not json",
        "messages": [
            {"role": "user", "content": "hi", "tool_calls": None},
            {"role": "assistant", "content": "hello", "tool_calls": None},
        ],
    },
]


@pytest.fixture
def patched_load_dataset(monkeypatch):
    def fake_load_dataset(dataset_id, split, streaming):
        assert dataset_id == QwenAgentAdapter.dataset_id
        assert streaming is True
        return iter(_SAMPLE_ROWS)

    import calibration.datasets.qwen_agent as m

    monkeypatch.setattr(m, "load_dataset", fake_load_dataset)


def test_qwen_agent_adapter_implements_protocol():
    assert isinstance(QwenAgentAdapter(), DatasetAdapter)


def test_qwen_agent_adapter_parses_multi_turn_with_structured_tools(
    patched_load_dataset,
):
    adapter = QwenAgentAdapter()
    convs = list(adapter.iter_conversations(split="train"))
    # Rows 0 and 2 survive; row 1 is empty-messages.
    assert len(convs) == 2

    first = convs[0]
    assert first.tools is not None
    assert len(first.tools) == 2
    assert first.tools[0]["function"]["name"] == "find_client_by_email"

    roles = [m["role"] for m in first.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]

    # Assistant turn with only tool_calls should have the serialized
    # call as content (otherwise the turn would be dropped as empty).
    assert "find_client_by_email" in first.messages[2]["content"]
    assert "user@example.com" in first.messages[2]["content"]


def test_qwen_agent_adapter_handles_malformed_tools_gracefully(
    patched_load_dataset,
):
    adapter = QwenAgentAdapter()
    convs = list(adapter.iter_conversations(split="train"))
    # The third row had invalid tools JSON; the adapter should yield
    # it with tools=None rather than crashing or dropping it.
    last = convs[-1]
    assert last.tools is None
    assert [m["role"] for m in last.messages] == ["user", "assistant"]
