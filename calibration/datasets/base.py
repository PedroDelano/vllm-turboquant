"""Core types for dataset adapters.

An adapter normalizes an arbitrary tool-calling / chat dataset into a
stream of `Conversation` objects. The generic builder then renders each
conversation through a target tokenizer's chat template (with tool
schemas passed via the `tools=` argument) and writes JSONL that the
TurboQuant calibration script and `vllm bench serve` both consume.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, TypedDict, runtime_checkable


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass
class Conversation:
    messages: list[Message]
    tools: list[dict] | None = None


@runtime_checkable
class DatasetAdapter(Protocol):
    name: ClassVar[str]
    dataset_id: ClassVar[str]
    default_split: ClassVar[str]

    def iter_conversations(self, *, split: str) -> Iterator[Conversation]: ...
