"""Shared fixtures for dataset-builder tests.

MockTokenizer mimics the subset of `transformers.PreTrainedTokenizer`
the builder touches: `chat_template`, `apply_chat_template`, `__call__`,
`decode`. It is fully deterministic — no real templates, no real
tokenization — so tests do not depend on any installed model weights or
network access.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from calibration.datasets.base import Conversation, Message


@dataclass
class _TokenizerOutput:
    input_ids: list[int]


class MockTokenizer:
    """Deterministic stand-in for a real HF tokenizer."""

    def __init__(
        self,
        *,
        chat_template: str | None = "<template>",
        fail_on_tools: bool = False,
        fail_on_tool_role: bool = False,
    ) -> None:
        self.chat_template = chat_template
        self._fail_on_tools = fail_on_tools
        self._fail_on_tool_role = fail_on_tool_role
        self._last_text: str = ""

    def apply_chat_template(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        if self._fail_on_tools and tools is not None:
            raise ValueError("template does not accept tools=")
        if self._fail_on_tool_role and any(
            m["role"] == "tool" for m in messages
        ):
            raise ValueError("template does not accept role=tool")
        parts: list[str] = []
        if tools is not None:
            parts.append(f"[TOOLS]{json.dumps(tools)}[/TOOLS]")
        for m in messages:
            parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
        return "".join(parts)

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> _TokenizerOutput:
        # Tokenize = whitespace-split words; IDs are indices.
        # Remember the text so decode() can reconstruct a prefix.
        self._last_text = text
        words = text.split()
        if truncation and max_length is not None:
            words = words[:max_length]
        return _TokenizerOutput(input_ids=list(range(len(words))))

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        words = self._last_text.split()[: len(ids)]
        return " ".join(words)


class StubAdapter:
    """Test adapter that yields a fixed list of Conversations."""

    name = "stub"
    dataset_id = "stub/stub"
    default_split = "train"

    def __init__(self, conversations: list[Conversation]) -> None:
        self._conversations = conversations

    def iter_conversations(self, *, split: str) -> Iterator[Conversation]:
        yield from self._conversations


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer()


@pytest.fixture
def mock_tokenizer_no_template() -> MockTokenizer:
    return MockTokenizer(chat_template=None)


@pytest.fixture
def mock_tokenizer_rejects_tools() -> MockTokenizer:
    return MockTokenizer(fail_on_tools=True)


@pytest.fixture
def mock_tokenizer_rejects_tool_role() -> MockTokenizer:
    return MockTokenizer(fail_on_tool_role=True)


@pytest.fixture
def simple_conversations() -> list[Conversation]:
    return [
        Conversation(
            messages=[
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "what is the weather in paris"},
                {"role": "assistant", "content": "it is sunny"},
            ],
            tools=None,
        )
    ]


@pytest.fixture
def conversations_with_tools() -> list[Conversation]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "get the weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    return [
        Conversation(
            messages=[
                {"role": "user", "content": "weather in paris"},
                {"role": "assistant", "content": "calling get_weather"},
            ],
            tools=tools,
        )
    ]
