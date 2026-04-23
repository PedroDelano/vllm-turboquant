"""Adapter for Salesforce/xlam-function-calling-60k.

Schema per row (as of 2026-04-23):
    - query:   str                   user's question
    - tools:   str (JSON-encoded)    list of function definitions
    - answers: str (JSON-encoded)    list of function calls

Single-turn: one user message + one synthetic assistant message whose
content is the JSON-serialized call list. Tool schemas are parsed out
of `tools` and placed in Conversation.tools so the chat template can
render them natively via `tools=`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import ClassVar

from datasets import load_dataset

from .base import Conversation, Message


def _parse_json_field(raw: object) -> list | None:
    """Accept both JSON-string and already-decoded-list forms (HF
    sometimes decodes columns eagerly depending on feature spec)."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            v = json.loads(s)
        except json.JSONDecodeError:
            return None
        return v if isinstance(v, list) else None
    return None


class XLAMAdapter:
    name: ClassVar[str] = "xlam"
    dataset_id: ClassVar[str] = "Salesforce/xlam-function-calling-60k"
    default_split: ClassVar[str] = "train"

    def iter_conversations(self, *, split: str) -> Iterator[Conversation]:
        ds = load_dataset(self.dataset_id, split=split, streaming=True)
        for row in ds:
            query = (row.get("query") or "").strip()
            tools = _parse_json_field(row.get("tools"))
            answers = _parse_json_field(row.get("answers"))
            if not query or tools is None or answers is None:
                continue
            messages: list[Message] = [
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "content": json.dumps(answers, ensure_ascii=False),
                },
            ]
            yield Conversation(messages=messages, tools=tools)
