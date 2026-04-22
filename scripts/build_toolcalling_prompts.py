#!/usr/bin/env python3
"""
Build a JSONL calibration-prompts file for tool-calling workloads.

Pulls N examples from glaiveai/glaive-function-calling-v2 (Apache-2.0,
public, ungated), parses each record into multi-turn messages, renders
via the target model's chat template, and writes one
``{"text": "..."}`` object per line.

Why: a TurboQuant metadata file derived from unstructured text (e.g. the
8-prompt tests/prompts/example.txt fixture) does not capture the
activation patterns that dominate tool-calling KV cache — chat-template
boundary tokens (``<|im_start|>``, ``<|im_end|>``), JSON punctuation in
tool schemas and arguments, and the long structured system prompts that
list available functions. Rendering real tool-calling conversations
through the model's own chat template gives the calibration script
activations that match production inference.

Usage
-----
::

    /root/vllm-venv-calib/bin/python scripts/build_toolcalling_prompts.py \\
        --tokenizer /workspace/hf-cache/models--Qwen--Qwen3.5-0.8B/snapshots/<sha>/ \\
        --output calibration/prompts/toolcalling_qwen3_5.jsonl \\
        --num-prompts 300 \\
        --max-tokens 1024

The output JSONL is consumed by ``benchmarks/generate_turboquant_metadata.py``
when ``--prompts-file`` has a ``.jsonl`` extension.

The tokenizer path only needs to share the chat template with the model
you are calibrating — Qwen3.5 family members all share one template, so
pointing at 0.8B is fine for generating prompts that will be used to
calibrate 122B.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

TURN_RE = re.compile(
    r"^(USER|ASSISTANT|FUNCTION RESPONSE):\s*",
    re.MULTILINE,
)
_ROLE_MAP = {
    "USER": "user",
    "ASSISTANT": "assistant",
    "FUNCTION RESPONSE": "tool",
}


def _parse_chat(chat: str) -> list[dict[str, str]]:
    """Split glaive-style chat into role/content messages."""
    parts = TURN_RE.split(chat)
    messages: list[dict[str, str]] = []
    for i in range(1, len(parts), 2):
        role = _ROLE_MAP.get(parts[i])
        if role is None:
            continue
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _render_with_fallback(tok, messages: list[dict[str, str]]) -> str | None:
    """Render via chat_template; fall back to role="user" for tool turns
    if the template rejects them."""
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        pass
    patched = []
    for m in messages:
        if m["role"] == "tool":
            patched.append({
                "role": "user",
                "content": f"[tool response]\n{m['content']}",
            })
        else:
            patched.append(m)
    try:
        return tok.apply_chat_template(
            patched, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path or HF id of a tokenizer with the target chat template.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=300,
        help="Number of rendered prompts to emit.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Truncate rendered prompts to this many tokens.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=128,
        help="Skip rendered prompts shorter than this many tokens.",
    )
    parser.add_argument(
        "--dataset",
        default="glaiveai/glaive-function-calling-v2",
        help="HF dataset id.",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to stream from.",
    )
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if getattr(tok, "chat_template", None) is None:
        raise SystemExit(
            f"Tokenizer at {args.tokenizer} has no chat_template; "
            "point --tokenizer at a chat-configured model."
        )

    ds = load_dataset(args.dataset, split=args.dataset_split, streaming=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for example in ds:
            if kept >= args.num_prompts:
                break
            system = (example.get("system") or "").strip()
            chat = (example.get("chat") or "").strip()
            if not system or not chat:
                skipped += 1
                continue
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system}
            ]
            messages.extend(_parse_chat(chat))
            if len(messages) < 2:
                skipped += 1
                continue
            rendered = _render_with_fallback(tok, messages)
            if rendered is None:
                skipped += 1
                continue
            ids = tok(
                rendered, truncation=True, max_length=args.max_tokens
            ).input_ids
            if len(ids) < args.min_tokens:
                skipped += 1
                continue
            truncated = tok.decode(ids, skip_special_tokens=False)
            f.write(
                json.dumps({"text": truncated}, ensure_ascii=False) + "\n"
            )
            kept += 1

    print(
        f"Wrote {kept} prompts to {out_path} "
        f"(skipped {skipped} malformed/too-short)."
    )


if __name__ == "__main__":
    main()
