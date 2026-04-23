"""Generic render / filter / write pipeline.

All dataset-specific logic lives in adapters. This module only knows
how to turn a stream of Conversation objects into the canonical
{"prompt": ..., "text": ...} JSONL consumed by
benchmarks/generate_turboquant_metadata.py and
`vllm bench serve --dataset-name custom`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .base import Conversation, DatasetAdapter, Message

ToolsFallback = Literal["error", "render-as-system", "drop"]


@dataclass
class BuildReport:
    kept: int
    skipped: int
    output_path: Path


def build(
    adapter: DatasetAdapter,
    *,
    tokenizer: Any,
    output_path: Path,
    num_prompts: int,
    min_tokens: int = 128,
    max_tokens: int = 1024,
    split: str | None = None,
    tools_fallback: ToolsFallback = "error",
) -> BuildReport:
    if getattr(tokenizer, "chat_template", None) is None:
        raise SystemExit(
            "Tokenizer has no chat_template; point --tokenizer at a "
            "chat-configured model."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    effective_split = split if split is not None else adapter.default_split

    kept = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for conv in adapter.iter_conversations(split=effective_split):
            if kept >= num_prompts:
                break
            rendered = _render(
                tokenizer, conv, tools_fallback=tools_fallback
            )
            if rendered is None:
                skipped += 1
                continue
            ids = tokenizer(
                rendered, truncation=True, max_length=max_tokens
            ).input_ids
            if len(ids) < min_tokens:
                skipped += 1
                continue
            truncated = tokenizer.decode(ids, skip_special_tokens=False)
            f.write(
                json.dumps(
                    {"prompt": truncated, "text": truncated},
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1

    if kept == 0:
        raise RuntimeError(
            f"build(): kept 0 prompts (skipped {skipped}). Check dataset "
            f"id / split / min_tokens threshold."
        )

    return BuildReport(kept=kept, skipped=skipped, output_path=output_path)


def _render(
    tokenizer: Any,
    conv: Conversation,
    *,
    tools_fallback: ToolsFallback,
) -> str | None:
    """Render a Conversation through apply_chat_template, handling
    tool-role rejection and tools= rejection according to the configured
    fallback strategy. Returns None if the conversation should be
    dropped."""
    try:
        return _apply(tokenizer, conv.messages, conv.tools)
    except Exception:
        patched = _patch_tool_role(conv.messages)
        if patched is not conv.messages:
            try:
                return _apply(tokenizer, patched, conv.tools)
            except Exception:
                pass
        if conv.tools is None:
            raise
        if tools_fallback == "error":
            raise
        if tools_fallback == "drop":
            return None
        if tools_fallback == "render-as-system":
            messages_with_embedded = _embed_tools_as_system(
                patched if patched is not conv.messages else conv.messages,
                conv.tools,
            )
            return _apply(tokenizer, messages_with_embedded, tools=None)
        raise ValueError(f"unknown tools_fallback: {tools_fallback!r}")


def _embed_tools_as_system(
    messages: list[Message], tools: list[dict]
) -> list[Message]:
    """Serialize tools as JSON and prepend to the system message (or
    insert one if absent). Used as a last-resort when a chat template
    does not accept the `tools=` argument."""
    tools_block = (
        "You have access to the following tools:\n"
        + json.dumps(tools, ensure_ascii=False, indent=2)
    )
    out: list[Message] = []
    injected = False
    for m in messages:
        if m["role"] == "system" and not injected:
            out.append({
                "role": "system",
                "content": f"{tools_block}\n\n{m['content']}",
            })
            injected = True
        else:
            out.append(m)
    if not injected:
        out.insert(0, {"role": "system", "content": tools_block})
    return out


def _apply(
    tokenizer: Any,
    messages: list[Message],
    tools: list[dict] | None,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )


def _patch_tool_role(messages: list[Message]) -> list[Message]:
    """Rewrite role=tool messages to role=user with a clear prefix, for
    chat templates that reject the tool role."""
    if not any(m["role"] == "tool" for m in messages):
        return messages
    out: list[Message] = []
    for m in messages:
        if m["role"] == "tool":
            out.append({
                "role": "user",
                "content": f"[tool response]\n{m['content']}",
            })
        else:
            out.append(m)
    return out
