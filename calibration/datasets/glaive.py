"""Adapter for glaiveai/glaive-function-calling-v2.

Rows: {"system": str, "chat": str}. Chat is a flat transcript with
markers USER: / ASSISTANT: / FUNCTION RESPONSE:. Tool schemas are
embedded as JSON inside the system field; we do not extract them into
a structured `tools` list (Conversation.tools stays None for this
adapter). The chat template will see the schemas as part of system
content, which is how the existing calibration artifact was built.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import ClassVar

from datasets import load_dataset

from .base import Conversation, Message

_TURN_RE = re.compile(
    r"^(USER|ASSISTANT|FUNCTION RESPONSE):\s*",
    re.MULTILINE,
)
_ROLE_MAP = {
    "USER": "user",
    "ASSISTANT": "assistant",
    "FUNCTION RESPONSE": "tool",
}


def _parse_chat(chat: str) -> list[Message]:
    parts = _TURN_RE.split(chat)
    messages: list[Message] = []
    for i in range(1, len(parts), 2):
        role = _ROLE_MAP.get(parts[i])
        if role is None:
            continue
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            messages.append({"role": role, "content": content})  # type: ignore[arg-type]
    return messages


class GlaiveAdapter:
    name: ClassVar[str] = "glaive"
    dataset_id: ClassVar[str] = "glaiveai/glaive-function-calling-v2"
    default_split: ClassVar[str] = "train"

    def iter_conversations(self, *, split: str) -> Iterator[Conversation]:
        ds = load_dataset(self.dataset_id, split=split, streaming=True)
        for row in ds:
            system = (row.get("system") or "").strip()
            chat = (row.get("chat") or "").strip()
            if not system or not chat:
                continue
            messages: list[Message] = [{"role": "system", "content": system}]
            messages.extend(_parse_chat(chat))
            if len(messages) < 2:
                continue
            yield Conversation(messages=messages, tools=None)
