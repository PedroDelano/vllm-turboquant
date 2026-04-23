# Generic dataset builder for TurboQuant calibration + benchmarking

Status: approved design, pending implementation plan.
Date: 2026-04-23.
AI assistance: this design was brainstormed with Claude (Opus 4.7). The
submitting human owns the design decisions and will review every changed
line during implementation.

## Problem

`scripts/build_toolcalling_prompts.py` is hard-coded to
`glaiveai/glaive-function-calling-v2`: it knows the column names (`system`,
`chat`), the flat-string turn separators (`USER:` / `ASSISTANT:` /
`FUNCTION RESPONSE:`), and emits a JSONL that feeds both the TurboQuant
calibration script and `vllm bench serve --dataset-name custom`.

Swapping the source dataset (e.g. glaive → `Salesforce/xlam-function-calling-60k`
→ `Team-ACE/ToolACE` → Berkeley Function Calling Leaderboard) currently
requires forking the script. We want a generic module that decouples
dataset-specific parsing from the rest of the pipeline, so new datasets
become one small adapter class + one registry-line edit.

## Non-goals

- Evaluating the quality of a calibration (that is `compare_bench_results.py`).
- AST-level BFCL scoring. This module only produces **prompts** from BFCL.
- Runtime tool execution. No sandbox, no real API calls.
- Dataset mixing, sharding, async, caching. Cheap and unnecessary at 300
  prompts/run; add later if needed.

## Users and consumers

The output JSONL is consumed unchanged by:

- `benchmarks/generate_turboquant_metadata.py` — reads the `text` field
  per line.
- `vllm bench serve --dataset-name custom --dataset-path <file>` — reads
  the `prompt` field per line.

Both consumers are already supported by today's builder via dual-key
output. The new module preserves that output shape exactly.

## Output shape (load-bearing)

One JSON object per line:

```json
{"prompt": "<rendered chat-template string>", "text": "<same string>"}
```

The string must be the model's chat template already applied, with special
tokens preserved (so `<|im_start|>` / `<|im_end|>` etc. are literal in the
file, not re-tokenized at serve time). Truncation is done on token ids
and then decoded back with `skip_special_tokens=False`, matching the
current glaive script's behaviour.

## Architecture

### Package layout

```
calibration/
├── prompts/                                          # existing output dir
│   └── toolcalling_qwen3_5.jsonl
├── datasets/                                         # NEW — this module
│   ├── __init__.py                                   # explicit ADAPTERS registry
│   ├── base.py                                       # Message, Conversation, DatasetAdapter protocol
│   ├── builder.py                                    # generic render / filter / write pipeline
│   ├── glaive.py                                     # GlaiveAdapter
│   ├── xlam.py                                       # XLAMAdapter
│   ├── toolace.py                                    # ToolACEAdapter
│   └── bfcl.py                                       # BFCLAdapter
└── Qwen_Qwen3.5-122B-A10B_turboquant35_toolcalling.json
```

Chosen over `scripts/` because `scripts/` is for executables, not a
package; chosen over a vLLM-internal path because this is a pre-serving
offline utility that should not ship inside the `vllm` wheel.

### Core types (`base.py`)

```python
from typing import ClassVar, Iterator, Literal, Protocol, TypedDict
from dataclasses import dataclass

class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

@dataclass
class Conversation:
    messages: list[Message]
    tools: list[dict] | None = None   # JSON-schema function defs

class DatasetAdapter(Protocol):
    name: ClassVar[str]               # CLI key (e.g. "xlam")
    dataset_id: ClassVar[str]         # HF repo id
    default_split: ClassVar[str]

    def iter_conversations(
        self, *, split: str,
    ) -> Iterator[Conversation]: ...
```

Rationale — why this schema, not richer:
- Tool schemas live at `Conversation.tools`, rendered via
  `tokenizer.apply_chat_template(messages, tools=tools, ...)`. This is
  the canonical modern path (Qwen3.5, Llama 3.1+, Mistral, Granite all
  support it). It puts the KV-cache-relevant distribution (long JSON
  tool schemas in the system prompt) through the tokenizer's own code
  path, matching production.
- Messages stay flat (`role` + `content`); assistant tool-calls are
  rendered as plain text inside `content`. The richer HF shape
  (`message.tool_calls: list[{name, arguments}]`, `message.tool_call_id`)
  is deliberately **not** modelled in v1: for **calibration**, what
  dominates the captured K/V distribution is the chat-template boundary
  tokens and the tool schemas, not the exact serialisation of the
  assistant's call. Upgrade path to full HF shape is clean — add
  optional fields to `Message` later; adapters that can populate them
  will, the rest default to `None`.

### Registry (`__init__.py`)

```python
from .base import Conversation, DatasetAdapter, Message
from .builder import BuildReport, build
from .glaive import GlaiveAdapter
from .xlam import XLAMAdapter
from .toolace import ToolACEAdapter
from .bfcl import BFCLAdapter

ADAPTERS: dict[str, type[DatasetAdapter]] = {
    GlaiveAdapter.name: GlaiveAdapter,
    XLAMAdapter.name: XLAMAdapter,
    ToolACEAdapter.name: ToolACEAdapter,
    BFCLAdapter.name: BFCLAdapter,
}
```

Explicit dict beats auto-registration for four in-tree adapters: a
5-line dict is readable top-to-bottom, and the friction of "edit one
line to add a dataset" is lower than the friction of debugging why an
auto-registered adapter did not show up. Revisit if we ever pass ~20
adapters.

### Generic pipeline (`builder.py`)

One public function:

```python
def build(
    adapter: DatasetAdapter,
    *,
    tokenizer,                                 # transformers.PreTrainedTokenizer
    output_path: Path,
    num_prompts: int,
    min_tokens: int = 128,
    max_tokens: int = 1024,
    split: str | None = None,                  # defaults to adapter.default_split
    tools_fallback: Literal[
        "error", "render-as-system", "drop"
    ] = "error",
) -> BuildReport: ...
```

Flow per conversation:

1. Render via `tokenizer.apply_chat_template(conv.messages,
   tools=conv.tools, tokenize=False, add_generation_prompt=False)`.
   - On rejection of the `tool` role: rewrite `tool` turns as `user`
     turns with a `[tool response]\n` prefix and retry once (ported
     from `build_toolcalling_prompts.py:73-96`).
   - On `tools=` rejection: apply `tools_fallback`.
2. Tokenize the rendered string; drop if `len(ids) < min_tokens`;
   truncate to `max_tokens`.
3. Decode back with `skip_special_tokens=False` — preserves
   `<|im_start|>` etc. as literal text.
4. Write `{"prompt": s, "text": s}` as one JSON line.
5. Stop when `kept == num_prompts`.

`BuildReport` is a small dataclass: `kept: int, skipped: int,
output_path: Path`. The CLI prints it.

### CLI (`scripts/build_prompts.py`)

Replaces `scripts/build_toolcalling_prompts.py`. Strict superset — the
same invocation with `--dataset glaive` reproduces today's behaviour.

```
python scripts/build_prompts.py \
    --dataset xlam \
    --tokenizer /workspace/hf-cache/.../Qwen3.5-0.8B/snapshots/<sha>/ \
    --output calibration/prompts/toolcalling_qwen3_5_xlam.jsonl \
    --num-prompts 300 \
    --min-tokens 128 \
    --max-tokens 1024
```

Args:
- `--dataset {glaive,xlam,toolace,bfcl}` (whatever is in `ADAPTERS`).
- `--tokenizer` (HF id or local path; must carry a chat template).
- `--output`, `--num-prompts`, `--min-tokens`, `--max-tokens`, `--split`
  (defaults to adapter's `default_split`).
- `--tools-fallback {error,render-as-system,drop}` (default `error`).

## Error handling

- **Malformed row.** Adapters wrap each row in `try/except` and increment
  a `skipped` counter. Pipeline logs `kept=N, skipped=M` at the end.
- **Tokenizer lacks `chat_template`.** Builder checks up-front and exits
  with an actionable message (matching `build_toolcalling_prompts.py:138`).
- **`apply_chat_template(..., tools=...)` fails.** Controlled by
  `--tools-fallback`:
  - `error` (default) — fail fast, operator investigates.
  - `render-as-system` — serialise `tools` as JSON and prepend to the
    system message content. Rough but lets old templates work.
  - `drop` — skip this conversation.
- **Unknown role.** Builder's built-in fallback rewrites `tool` → `user`
  with a `[tool response]` prefix. Adapters never have to care.
- **Empty output.** If `kept == 0` at the end, raise rather than write
  an empty file — catches "wrong dataset id" / "everything filtered
  out" early.

## Testing

`tests/calibration/datasets/`, fixture-driven, no live HF downloads.

- **`test_adapters.py`** — one test per adapter. Monkeypatches
  `datasets.load_dataset` to return 2–3 hand-crafted rows copied from
  real HF structure. Asserts:
  - Expected role sequence in `Conversation.messages`.
  - `tools` list length and entry `name` fields.
  - Edge cases: missing `system`, malformed tool-call JSON, single-turn
    vs multi-turn.
- **`test_builder.py`** — stub adapter yielding deterministic
  conversations. Verifies:
  - JSONL has both `prompt` and `text` keys.
  - `min_tokens` filtering drops shorts.
  - `max_tokens` truncation at the boundary.
  - `num_prompts` stop condition.
  - `tools_fallback="render-as-system"` produces non-empty system
    content when the template rejects `tools=`.
- **`test_registry.py`** — `ADAPTERS` keys match each class's `name`
  attr; no duplicates; each class is `isinstance`-compatible with the
  `runtime_checkable` Protocol.

Tokenizer in tests: `Qwen/Qwen2.5-0.5B-Instruct` (already validated on
H100 per `CLAUDE.md`; small and fast). Unit-level builder tests use a
mock tokenizer where exact template output does not matter.

## Scope cuts (YAGNI)

Explicitly **out** of v1:

- Async / parallel row fetching.
- Sharding or multi-process rendering.
- Result caching across runs.
- Dataset mixing (`150 glaive + 150 xlam → one JSONL`). Trivial later
  as a `MixAdapter` wrapping `list[DatasetAdapter]` with quotas.
- Splitting output into calibration-only and eval-only JSONLs. Already
  achievable by running the builder twice with disjoint row ranges.
- Auto-inferring the chat template from the calibration model's
  checkpoint. Keep `--tokenizer` explicit — it is the one dial you
  actually want to control.
- BFCL AST-level scoring / executable tests. This module produces
  **prompts** from BFCL only.

## Deletions

- `scripts/build_toolcalling_prompts.py` — removed.
  `scripts/build_prompts.py --dataset glaive` is a strict superset.
- The existing `calibration/prompts/toolcalling_qwen3_5.jsonl` artifact
  is **not** regenerated by this change. Optional follow-up: re-render
  it with `--dataset glaive` in a separate commit and confirm it is
  unchanged (or explicitly note the diff).

## Open follow-ups (not part of this module)

- Extending `benchmark_turboquant_vs_baseline_qwen3_5.sh` to run with
  multiple prompt sources (xlam for calibration, BFCL for evaluation).
- Wiring `scripts/eval_perplexity.py` (discussed earlier in session) to
  measure PPL delta between baseline and TurboQuant arms using these
  same JSONL files.
