"""Adapter for gorilla-llm/Berkeley-Function-Calling-Leaderboard.

BFCL is a config'd dataset (`simple`, `multiple`, `parallel`,
`parallel_multiple`, `live_*` variants, `multi_turn_base`, ...). The
adapter takes a `subset` constructor arg and passes it through to
`load_dataset(..., name=subset, ...)`.

Per-row schema (non-multi-turn categories):
    - id:       str
    - question: list[list[{"role": ..., "content": ...}]]
                outer is 1-element for single-turn; inner is the messages.
    - function: list[dict]  function definitions (== tools list).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

from datasets import load_dataset

from .base import Conversation, Message

_VALID_ROLES = {"system", "user", "assistant", "tool"}


class BFCLAdapter:
    name: ClassVar[str] = "bfcl"
    dataset_id: ClassVar[str] = (
        "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    )
    default_split: ClassVar[str] = "train"

    def __init__(self, *, subset: str = "simple") -> None:
        self.subset = subset

    def iter_conversations(self, *, split: str) -> Iterator[Conversation]:
        ds = load_dataset(
            self.dataset_id,
            name=self.subset,
            split=split,
            streaming=True,
        )
        for row in ds:
            question: list[list[dict[str, Any]]] = row.get("question") or []
            function = row.get("function") or []
            if not question or not isinstance(function, list):
                continue
            inner = question[0] if question else []
            messages: list[Message] = []
            for turn in inner:
                role = turn.get("role")
                content = (turn.get("content") or "").strip()
                if role not in _VALID_ROLES or not content:
                    continue
                messages.append({"role": role, "content": content})  # type: ignore[arg-type]
            if not messages:
                continue
            yield Conversation(messages=messages, tools=function)
