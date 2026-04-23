"""Adapter for zake7749/Qwen-3.6-plus-agent-tool-calling-trajectory.

Multi-turn agentic trajectories with OpenAI-format chat messages and
a structured tool list per conversation.

Per-row schema:
    - id:        str
    - domain:    str
    - score:     float
    - reward:    float
    - num_turns: int
    - tools:     str (JSON-encoded OpenAI `[{type, function: {...}}]`)
    - messages:  list[dict] with keys {role, content, reasoning_content,
                 tool_call_id, tool_calls}

Assistant turns that fire tools carry `content=None` and populate
`tool_calls`; the adapter serializes the tool_calls list into the
message content so the chat template sees a non-empty string (and so
the calibration pass captures the JSON punctuation that dominates
tool-call decode activations).

`reasoning_content` (internal chain-of-thought) is intentionally
dropped — it is not shown to the model at serve time and would bias
the calibration distribution away from production.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import ClassVar

from datasets import load_dataset

from .base import Conversation, Message

_VALID_ROLES = {"system", "user", "assistant", "tool"}


def _parse_tools(raw: object) -> list | None:
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


class QwenAgentAdapter:
    name: ClassVar[str] = "qwen-agent"
    dataset_id: ClassVar[str] = (
        "zake7749/Qwen-3.6-plus-agent-tool-calling-trajectory"
    )
    default_split: ClassVar[str] = "train"

    def iter_conversations(self, *, split: str) -> Iterator[Conversation]:
        ds = load_dataset(self.dataset_id, split=split, streaming=True)
        for row in ds:
            raw_messages = row.get("messages") or []
            if not raw_messages:
                continue
            tools = _parse_tools(row.get("tools"))

            messages: list[Message] = []
            for m in raw_messages:
                role = m.get("role")
                if role not in _VALID_ROLES:
                    continue
                content = m.get("content")
                if not content:
                    tool_calls = m.get("tool_calls")
                    if role == "assistant" and tool_calls:
                        content = json.dumps(tool_calls, ensure_ascii=False)
                content = (content or "").strip()
                if not content:
                    continue
                messages.append({"role": role, "content": content})  # type: ignore[arg-type]
            if len(messages) < 2:
                continue
            yield Conversation(messages=messages, tools=tools)
