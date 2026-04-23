"""Adapter for Team-ACE/ToolACE.

Normalizes ToolACE's non-standard role names to the canonical set:

    function_call -> assistant   (its content is the serialized call)
    observation   -> tool        (its content is the tool's response)
    system / user / assistant    -> unchanged
    tool                         -> unchanged

Accepts both the `role`/`content` shape and the sharegpt `from`/`value`
shape, since ToolACE has shipped both.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

from datasets import load_dataset

from .base import Conversation, Message

_ROLE_MAP = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "function_call": "assistant",
    "tool": "tool",
    "observation": "tool",
    "function_response": "tool",
}


def _turn_role_and_content(turn: dict[str, Any]) -> tuple[str, str] | None:
    if "role" in turn and "content" in turn:
        role_src, content = turn["role"], turn["content"]
    elif "from" in turn and "value" in turn:
        role_src, content = turn["from"], turn["value"]
    else:
        return None
    role = _ROLE_MAP.get(str(role_src).lower())
    if role is None:
        return None
    content = (content or "").strip()
    if not content:
        return None
    return role, content


class ToolACEAdapter:
    name: ClassVar[str] = "toolace"
    dataset_id: ClassVar[str] = "Team-ACE/ToolACE"
    default_split: ClassVar[str] = "train"

    def iter_conversations(self, *, split: str) -> Iterator[Conversation]:
        ds = load_dataset(self.dataset_id, split=split, streaming=True)
        for row in ds:
            conversations = row.get("conversations") or []
            if not conversations:
                continue
            tools = row.get("tools") or None
            if tools is not None and not isinstance(tools, list):
                tools = None

            messages: list[Message] = []
            system = (row.get("system") or "").strip()
            if system:
                messages.append({"role": "system", "content": system})
            for turn in conversations:
                parsed = _turn_role_and_content(turn)
                if parsed is None:
                    continue
                role, content = parsed
                messages.append({"role": role, "content": content})  # type: ignore[arg-type]
            if len(messages) < 2:
                continue
            yield Conversation(messages=messages, tools=tools)
