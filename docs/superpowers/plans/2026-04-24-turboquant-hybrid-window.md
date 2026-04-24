# TurboQuant Head+Tail Hybrid Window — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a head+tail sliding-window hybrid to TurboQuant so long-context Qwen serving produces coherent output with ~2.7× compression vs bf16.

**Architecture:** Two parallel KV caches per attention layer: the existing tq35 paged cache (stores ALL tokens) plus a new fixed-size bf16 "window" cache that stores only the first `head_size` tokens and the last `tail_size` tokens per sequence. Decode runs two attention passes (tq35 masked to the middle, dense torch-level attention on the window) and merges via LSE. Prefill uses the existing dense path unchanged.

**Tech Stack:** Python 3.12, PyTorch (bf16), Triton, vLLM v1 attention backends. All work in `/workspace/vllm-turboquant/` on `/root/vllm-venv`.

**Commit policy (user override):** NO commits, NO git operations of any kind. All work is left uncommitted in the working tree for the user to review. Every task ends with a `pytest` verification; none of them ends with `git commit`.

---

## File Structure

**Created:**
- `vllm/v1/attention/ops/hybrid_window_cache.py` — `Bf16Window` dataclass, `allocate_bf16_window`, `write_prefill_to_window`, `append_decode_to_window`, `gather_window_kv`.
- `tests/quantization/test_turboquant_hybrid_window.py` — unit + integration tests.

**Modified:**
- `vllm/config/cache.py` — add `hybrid_window_head_size`, `hybrid_window_tail_size` fields to `CacheConfig`, plus startup validation.
- `vllm/engine/arg_utils.py` — surface `--turboquant-head-window-size` and `--turboquant-tail-window-size` flags on the engine args.
- `vllm/v1/attention/backends/triton_attn.py` — extend `do_kv_cache_update` to also write to the bf16 window; extend `_forward_turboquant` with a hybrid 2-pass path behind a config check.

**Untouched:**
- `vllm/v1/attention/ops/turboquant_kv_cache.py` (kernel + packing code unchanged).
- `vllm/v1/attention/ops/triton_turboquant_decode.py` and `triton_turboquant_kv_update.py` (unchanged).
- `benchmarks/generate_turboquant_metadata.py` (unchanged — calibration path doesn't need to know about the window).

---

## Shared fixtures

Tasks below reference these fixtures. Set them up once before starting.

- Local repo root: `/workspace/vllm-turboquant`
- Main venv (has vLLM + pytest + datasets): `/root/vllm-venv/bin/python`, `/root/vllm-venv/bin/pytest`
- Small model for cheap tests: `Qwen/Qwen2.5-0.5B-Instruct` snapshot at `/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- Existing small-model calibration metadata: `/tmp/qwen2_5_turboquant35.json`
- Existing small-model glaive prompts: `/tmp/small_glaive_qwen25.jsonl`

If any of those `/tmp/*` files are missing, regenerate with:

```bash
/root/vllm-venv/bin/python scripts/build_prompts.py \
    --dataset glaive \
    --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
    --output /tmp/small_glaive_qwen25.jsonl \
    --num-prompts 64 --min-tokens 128 --max-tokens 1024
/root/vllm-venv/bin/python benchmarks/generate_turboquant_metadata.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --kv-cache-dtype turboquant35 \
    --prompts-file /tmp/small_glaive_qwen25.jsonl \
    --output /tmp/qwen2_5_turboquant35.json \
    --device cuda --dtype bfloat16 \
    --batch-size 4 --max-seq-len 1024 --max-prompts 64 --trust-remote-code
```

---

## Task 1: Config fields + validation

**Files:**
- Modify: `vllm/config/cache.py`

- [ ] **Step 1.1: Read the file to find `_validate_turboquant`**

Run:
```bash
grep -n "enable_turboquant\|turboquant_metadata_path\|_validate_turboquant" vllm/config/cache.py | head -20
```

Confirm the existing fields (`enable_turboquant`, `turboquant_metadata_path`) live in `CacheConfig` and that `_validate_turboquant` is a `model_validator`.

- [ ] **Step 1.2: Add config fields**

Find the line declaring `turboquant_metadata_path` on `CacheConfig`. Add two fields alongside:

```python
    hybrid_window_head_size: int = 0
    """Number of leading tokens per sequence to keep in bf16 alongside the
    turboquant cache. 0 disables the hybrid window."""

    hybrid_window_tail_size: int = 0
    """Number of trailing tokens per sequence to keep in bf16 alongside the
    turboquant cache. 0 disables the hybrid window."""
```

Defaults of `0/0` keep the current behaviour. Non-zero enables the window.

- [ ] **Step 1.3: Add validator**

Extend `_validate_turboquant` (the `@model_validator(mode="after")` method). Append to its body before `return self`:

```python
        head = self.hybrid_window_head_size
        tail = self.hybrid_window_tail_size
        if head < 0 or tail < 0:
            raise ValueError(
                "hybrid_window_head_size/tail_size must be >= 0; got "
                f"head={head}, tail={tail}."
            )
        if (head == 0) != (tail == 0):
            raise ValueError(
                "hybrid_window_head_size and hybrid_window_tail_size must be "
                "both zero (disabled) or both positive (enabled); got "
                f"head={head}, tail={tail}."
            )
```

This guards the "half-enabled" misconfiguration.

- [ ] **Step 1.4: Verify import/syntax**

Run:
```bash
/root/vllm-venv/bin/python -c "from vllm.config.cache import CacheConfig; print(CacheConfig.model_fields['hybrid_window_head_size'])"
```

Expected: prints a `FieldInfo(...)` line without error.

---

## Task 2: Engine-arg plumbing

**Files:**
- Modify: `vllm/engine/arg_utils.py`

- [ ] **Step 2.1: Locate the turboquant args**

Run:
```bash
grep -n "enable_turboquant\|turboquant_metadata_path\|turboquant-metadata-path" vllm/engine/arg_utils.py | head -10
```

Note the `add_argument` call for `--turboquant-metadata-path`. We'll add two more alongside it.

- [ ] **Step 2.2: Add the two CLI arguments**

Right after the existing `--turboquant-metadata-path` add_argument block in `arg_utils.py`, add:

```python
        parser.add_argument(
            "--turboquant-head-window-size",
            type=int,
            default=0,
            help=(
                "Number of leading tokens per sequence to keep in bf16 "
                "alongside the turboquant cache. 0 disables the hybrid "
                "window (default)."
            ),
        )
        parser.add_argument(
            "--turboquant-tail-window-size",
            type=int,
            default=0,
            help=(
                "Number of trailing tokens per sequence to keep in bf16 "
                "alongside the turboquant cache. 0 disables the hybrid "
                "window (default)."
            ),
        )
```

Match the existing argparse style in that file (tabs/spaces, help wrapping). If the existing style uses explicit `dest=`, match that too.

- [ ] **Step 2.3: Thread values into CacheConfig**

Find where `CacheConfig(...)` is constructed from the parsed args (typically `create_engine_config` or similar). The existing code passes `turboquant_metadata_path=args.turboquant_metadata_path`. Add:

```python
            hybrid_window_head_size=args.turboquant_head_window_size,
            hybrid_window_tail_size=args.turboquant_tail_window_size,
```

Note: argparse converts `--turboquant-head-window-size` to `args.turboquant_head_window_size`. Verify with the surrounding code — if it uses `getattr(args, ...)` style, match.

- [ ] **Step 2.4: Verify**

Run:
```bash
/root/vllm-venv/bin/python -c "
from vllm.engine.arg_utils import EngineArgs
import argparse
p = argparse.ArgumentParser()
EngineArgs.add_cli_args(p)
args = p.parse_args(['--turboquant-head-window-size=128', '--turboquant-tail-window-size=64', '--model=Qwen/Qwen2.5-0.5B-Instruct'])
print('head:', args.turboquant_head_window_size)
print('tail:', args.turboquant_tail_window_size)
"
```

Expected: `head: 128` and `tail: 64`. If `EngineArgs.add_cli_args` is not the correct hook name, use the grep output from Step 2.1 to find the right one.

---

## Task 3: Window cache module skeleton + dataclass

**Files:**
- Create: `vllm/v1/attention/ops/hybrid_window_cache.py`

- [ ] **Step 3.1: Write the module skeleton**

Create `vllm/v1/attention/ops/hybrid_window_cache.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bf16 sliding-window cache that runs alongside the turboquant KV cache.

Stores the first `head_size` tokens and the last `tail_size` tokens of each
sequence in bf16. The middle region lives only in the turboquant cache and
pays quantization error. At decode time, the attention backend runs two
passes (turboquant middle + dense head/tail) and merges via LSE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Bf16Window:
    """Per-(layer, seq) bf16 storage for head+tail regions.

    Shapes (for a single layer, all sequences):
        head_keys:    (num_seqs, head_size, num_kv_heads, head_dim)  bf16
        head_values:  (num_seqs, head_size, num_kv_heads, head_dim)  bf16
        tail_keys:    (num_seqs, tail_size, num_kv_heads, head_dim)  bf16
        tail_values:  (num_seqs, tail_size, num_kv_heads, head_dim)  bf16

        head_token_count: (num_seqs,) int32 — how many tokens of head are populated.
        tail_write_ptr:   (num_seqs,) int32 — next tail slot to overwrite (ring).
        tail_token_count: (num_seqs,) int32 — how many tail slots are populated.
        total_seq_len:    (num_seqs,) int32 — total tokens written (head + middle + tail).
    """

    head_keys: torch.Tensor
    head_values: torch.Tensor
    tail_keys: torch.Tensor
    tail_values: torch.Tensor
    head_token_count: torch.Tensor
    tail_write_ptr: torch.Tensor
    tail_token_count: torch.Tensor
    total_seq_len: torch.Tensor

    @property
    def head_size(self) -> int:
        return self.head_keys.shape[1]

    @property
    def tail_size(self) -> int:
        return self.tail_keys.shape[1]

    @property
    def num_seqs(self) -> int:
        return self.head_keys.shape[0]

    @property
    def num_kv_heads(self) -> int:
        return self.head_keys.shape[2]

    @property
    def head_dim(self) -> int:
        return self.head_keys.shape[3]


def allocate_bf16_window(
    *,
    num_seqs: int,
    head_size: int,
    tail_size: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Bf16Window:
    """Allocate the bf16 head+tail storage for one attention layer."""
    if head_size <= 0 or tail_size <= 0:
        raise ValueError(
            f"head_size and tail_size must be positive; got "
            f"head={head_size}, tail={tail_size}."
        )
    shape_head = (num_seqs, head_size, num_kv_heads, head_dim)
    shape_tail = (num_seqs, tail_size, num_kv_heads, head_dim)
    return Bf16Window(
        head_keys=torch.zeros(shape_head, dtype=dtype, device=device),
        head_values=torch.zeros(shape_head, dtype=dtype, device=device),
        tail_keys=torch.zeros(shape_tail, dtype=dtype, device=device),
        tail_values=torch.zeros(shape_tail, dtype=dtype, device=device),
        head_token_count=torch.zeros(num_seqs, dtype=torch.int32, device=device),
        tail_write_ptr=torch.zeros(num_seqs, dtype=torch.int32, device=device),
        tail_token_count=torch.zeros(num_seqs, dtype=torch.int32, device=device),
        total_seq_len=torch.zeros(num_seqs, dtype=torch.int32, device=device),
    )
```

- [ ] **Step 3.2: Verify import and allocator work**

Run:
```bash
/root/vllm-venv/bin/python -c "
import torch
from vllm.v1.attention.ops.hybrid_window_cache import allocate_bf16_window
w = allocate_bf16_window(
    num_seqs=2, head_size=4, tail_size=3, num_kv_heads=2, head_dim=8,
    device=torch.device('cuda'),
)
print('head shape:', w.head_keys.shape, w.head_keys.dtype)
print('tail shape:', w.tail_keys.shape, w.tail_keys.dtype)
print('head_token_count:', w.head_token_count)
assert w.head_size == 4 and w.tail_size == 3 and w.num_seqs == 2
print('ok')
"
```

Expected: shapes `(2,4,2,8)` and `(2,3,2,8)`, `dtype=bfloat16`, `head_token_count=tensor([0,0])`, `ok`.

---

## Task 4: Prefill write path + unit test

**Files:**
- Modify: `vllm/v1/attention/ops/hybrid_window_cache.py`
- Create: `tests/quantization/test_turboquant_hybrid_window.py`

- [ ] **Step 4.1: Write the failing test first**

Create `tests/quantization/test_turboquant_hybrid_window.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the bf16 hybrid window cache used alongside TurboQuant."""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops.hybrid_window_cache import (
    Bf16Window,
    allocate_bf16_window,
)


def _make_window(num_seqs=2, head=4, tail=3, kv=1, dim=4):
    return allocate_bf16_window(
        num_seqs=num_seqs,
        head_size=head,
        tail_size=tail,
        num_kv_heads=kv,
        head_dim=dim,
        device=torch.device("cuda"),
    )


def test_allocator_shapes_and_dtypes():
    w = _make_window()
    assert w.head_keys.shape == (2, 4, 1, 4)
    assert w.tail_keys.shape == (2, 3, 1, 4)
    assert w.head_keys.dtype == torch.bfloat16
    assert w.head_token_count.dtype == torch.int32
    assert w.head_token_count.tolist() == [0, 0]
    assert w.tail_write_ptr.tolist() == [0, 0]


def test_write_prefill_shorter_than_head():
    from vllm.v1.attention.ops.hybrid_window_cache import write_prefill_to_window
    w = _make_window(num_seqs=1, head=8, tail=4, kv=1, dim=2)
    # Sequence of 3 tokens — all go in the head, tail untouched.
    k = torch.arange(6, dtype=torch.bfloat16, device="cuda").reshape(3, 1, 2)
    v = k.clone() + 100
    write_prefill_to_window(w, seq_id=0, keys=k, values=v)
    assert w.head_token_count.tolist() == [3]
    assert w.tail_token_count.tolist() == [0]
    assert w.total_seq_len.tolist() == [3]
    assert torch.equal(w.head_keys[0, :3], k)
    assert torch.equal(w.head_values[0, :3], v)
    # Untouched head slots remain zero.
    assert torch.all(w.head_keys[0, 3:] == 0)


def test_write_prefill_between_head_and_tail():
    from vllm.v1.attention.ops.hybrid_window_cache import write_prefill_to_window
    w = _make_window(num_seqs=1, head=4, tail=3, kv=1, dim=2)
    # Sequence of 5 tokens — first 4 go to head, last 1 to tail.
    k = torch.arange(10, dtype=torch.bfloat16, device="cuda").reshape(5, 1, 2)
    v = k.clone() + 100
    write_prefill_to_window(w, seq_id=0, keys=k, values=v)
    assert w.head_token_count.tolist() == [4]
    assert w.tail_token_count.tolist() == [1]
    assert w.total_seq_len.tolist() == [5]
    assert torch.equal(w.head_keys[0], k[:4])
    assert torch.equal(w.tail_keys[0, 0], k[4, 0])
    # tail slot 1+ untouched.
    assert torch.all(w.tail_keys[0, 1:] == 0)


def test_write_prefill_long_sequence():
    from vllm.v1.attention.ops.hybrid_window_cache import write_prefill_to_window
    w = _make_window(num_seqs=1, head=4, tail=3, kv=1, dim=2)
    # Sequence of 10 tokens: head=first 4, middle=next 3 (dropped from window),
    # tail=last 3.
    k = torch.arange(20, dtype=torch.bfloat16, device="cuda").reshape(10, 1, 2)
    v = k.clone() + 100
    write_prefill_to_window(w, seq_id=0, keys=k, values=v)
    assert w.head_token_count.tolist() == [4]
    assert w.tail_token_count.tolist() == [3]
    assert w.total_seq_len.tolist() == [10]
    assert torch.equal(w.head_keys[0], k[:4])
    assert torch.equal(w.tail_keys[0], k[7:10])
```

- [ ] **Step 4.2: Run — expect ImportError on `write_prefill_to_window`**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_hybrid_window.py -v 2>&1 | tail -10
```

Expected: first test passes (allocator exists); remaining three fail with ImportError on `write_prefill_to_window`.

- [ ] **Step 4.3: Implement the prefill write**

Append to `vllm/v1/attention/ops/hybrid_window_cache.py`:

```python
def write_prefill_to_window(
    window: Bf16Window,
    *,
    seq_id: int,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Write a prefill of length L to the head+tail regions of one sequence.

    Layout: first min(L, head_size) tokens → head; if L > head_size, the
    last min(L - head_size, tail_size) tokens → tail (tokens beyond the
    head but before the tail end are in the "middle" region — they live
    only in the turboquant cache, not the bf16 window).

    Shapes:
        keys, values: (L, num_kv_heads, head_dim) bf16.
    """
    assert keys.shape == values.shape
    assert keys.dim() == 3
    L = keys.shape[0]
    head_size = window.head_size
    tail_size = window.tail_size

    n_head = min(L, head_size)
    if n_head > 0:
        window.head_keys[seq_id, :n_head].copy_(keys[:n_head])
        window.head_values[seq_id, :n_head].copy_(values[:n_head])
    window.head_token_count[seq_id] = n_head

    remaining = L - head_size
    if remaining > 0:
        n_tail = min(remaining, tail_size)
        tail_slice = slice(L - n_tail, L)
        window.tail_keys[seq_id, :n_tail].copy_(keys[tail_slice])
        window.tail_values[seq_id, :n_tail].copy_(values[tail_slice])
        window.tail_token_count[seq_id] = n_tail
        window.tail_write_ptr[seq_id] = n_tail % tail_size
    else:
        window.tail_token_count[seq_id] = 0
        window.tail_write_ptr[seq_id] = 0

    window.total_seq_len[seq_id] = L
```

- [ ] **Step 4.4: Run — all four tests should pass**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_hybrid_window.py -v 2>&1 | tail -10
```

Expected: 4 passed.

---

## Task 5: Decode append + ring-buffer eviction test

**Files:**
- Modify: `vllm/v1/attention/ops/hybrid_window_cache.py`
- Modify: `tests/quantization/test_turboquant_hybrid_window.py`

- [ ] **Step 5.1: Write failing test**

Append to `tests/quantization/test_turboquant_hybrid_window.py`:

```python
def test_append_decode_fills_tail():
    from vllm.v1.attention.ops.hybrid_window_cache import (
        append_decode_to_window,
        write_prefill_to_window,
    )
    w = _make_window(num_seqs=1, head=4, tail=3, kv=1, dim=2)
    # Prefill seeds first 4 tokens in head, no tail yet.
    k = torch.arange(8, dtype=torch.bfloat16, device="cuda").reshape(4, 1, 2)
    v = k.clone() + 100
    write_prefill_to_window(w, seq_id=0, keys=k, values=v)
    # Decode two tokens — land in tail.
    for i, tok in enumerate([10, 20]):
        dk = torch.full((1, 1, 2), tok, dtype=torch.bfloat16, device="cuda")
        dv = dk.clone() + 1
        append_decode_to_window(w, seq_id=0, key=dk[0], value=dv[0])
    assert w.tail_token_count.tolist() == [2]
    assert w.total_seq_len.tolist() == [6]
    assert w.tail_keys[0, 0, 0, 0].item() == 10.0
    assert w.tail_keys[0, 1, 0, 0].item() == 20.0


def test_append_decode_evicts_oldest_on_wrap():
    from vllm.v1.attention.ops.hybrid_window_cache import (
        append_decode_to_window,
        write_prefill_to_window,
    )
    w = _make_window(num_seqs=1, head=4, tail=3, kv=1, dim=2)
    # Prefill fills head only.
    k = torch.arange(8, dtype=torch.bfloat16, device="cuda").reshape(4, 1, 2)
    v = k.clone() + 100
    write_prefill_to_window(w, seq_id=0, keys=k, values=v)
    # Append 5 decode tokens to a tail of size 3 — last 3 should win.
    for tok in [10, 20, 30, 40, 50]:
        dk = torch.full((1, 1, 2), tok, dtype=torch.bfloat16, device="cuda")
        dv = dk.clone() + 1
        append_decode_to_window(w, seq_id=0, key=dk[0], value=dv[0])
    assert w.tail_token_count.tolist() == [3]
    assert w.total_seq_len.tolist() == [9]
    # The ring state holds 30, 40, 50 logically (order doesn't matter for
    # attention but the set must match).
    stored = sorted(w.tail_keys[0, :, 0, 0].tolist())
    assert stored == [30.0, 40.0, 50.0]
```

- [ ] **Step 5.2: Run — expect ImportError on `append_decode_to_window`**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_hybrid_window.py -v 2>&1 | tail -10
```

Expected: 4 passed, 2 fail.

- [ ] **Step 5.3: Implement append_decode_to_window**

Append to `vllm/v1/attention/ops/hybrid_window_cache.py`:

```python
def append_decode_to_window(
    window: Bf16Window,
    *,
    seq_id: int,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Append one decode token (for one sequence) to the window.

    Goes to the head if head isn't full yet; otherwise to the tail ring
    buffer, overwriting the oldest slot when full.

    Shapes:
        key, value: (num_kv_heads, head_dim) bf16.
    """
    assert key.shape == value.shape
    head_count = int(window.head_token_count[seq_id].item())
    if head_count < window.head_size:
        window.head_keys[seq_id, head_count].copy_(key)
        window.head_values[seq_id, head_count].copy_(value)
        window.head_token_count[seq_id] = head_count + 1
    else:
        ptr = int(window.tail_write_ptr[seq_id].item())
        window.tail_keys[seq_id, ptr].copy_(key)
        window.tail_values[seq_id, ptr].copy_(value)
        window.tail_write_ptr[seq_id] = (ptr + 1) % window.tail_size
        current_count = int(window.tail_token_count[seq_id].item())
        if current_count < window.tail_size:
            window.tail_token_count[seq_id] = current_count + 1
    window.total_seq_len[seq_id] = window.total_seq_len[seq_id] + 1
```

- [ ] **Step 5.4: Run — all 6 tests should pass**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_hybrid_window.py -v 2>&1 | tail -10
```

Expected: 6 passed.

---

## Task 6: Gather API for attention Pass 2

**Files:**
- Modify: `vllm/v1/attention/ops/hybrid_window_cache.py`
- Modify: `tests/quantization/test_turboquant_hybrid_window.py`

- [ ] **Step 6.1: Write failing test**

Append to `tests/quantization/test_turboquant_hybrid_window.py`:

```python
def test_gather_returns_head_plus_tail_in_position_order():
    from vllm.v1.attention.ops.hybrid_window_cache import (
        append_decode_to_window,
        gather_window_kv,
        write_prefill_to_window,
    )
    w = _make_window(num_seqs=1, head=4, tail=3, kv=1, dim=2)
    # Prefill 4 tokens (head only), then 3 decode tokens (tail).
    k = torch.arange(8, dtype=torch.bfloat16, device="cuda").reshape(4, 1, 2)
    v = k.clone() + 100
    write_prefill_to_window(w, seq_id=0, keys=k, values=v)
    for tok in [10, 20, 30]:
        dk = torch.full((1, 1, 2), tok, dtype=torch.bfloat16, device="cuda")
        dv = dk.clone() + 1
        append_decode_to_window(w, seq_id=0, key=dk[0], value=dv[0])

    # Total 7 tokens: 4 head + 3 tail. Positions 0..3 are head; 4..6 are tail.
    keys_out, values_out, positions = gather_window_kv(w, seq_id=0)
    assert keys_out.shape == (7, 1, 2)
    assert positions.tolist() == [0, 1, 2, 3, 4, 5, 6]
    # Head kept in order.
    assert torch.equal(keys_out[:4], k)
    # Tail values in arrival order 10, 20, 30 — match what append wrote.
    assert keys_out[4, 0, 0].item() == 10.0
    assert keys_out[5, 0, 0].item() == 20.0
    assert keys_out[6, 0, 0].item() == 30.0
```

- [ ] **Step 6.2: Run — expect ImportError**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_hybrid_window.py::test_gather_returns_head_plus_tail_in_position_order -v 2>&1 | tail -8
```

Expected: ImportError on `gather_window_kv`.

- [ ] **Step 6.3: Implement gather_window_kv**

Append to `vllm/v1/attention/ops/hybrid_window_cache.py`:

```python
def gather_window_kv(
    window: Bf16Window,
    *,
    seq_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the head+tail K/V for one sequence, along with the position
    indices in the original sequence those tokens correspond to.

    Positions are:
        head:  [0, head_count)
        tail:  [total_seq_len - tail_count, total_seq_len)

    Tail entries are ordered by arrival (oldest first). For a ring that
    hasn't wrapped yet, this is just the first `tail_count` slots. After
    wrapping, the oldest entry sits at `tail_write_ptr` and wraps around.

    Returns:
        keys:      (W, num_kv_heads, head_dim) bf16
        values:    (W, num_kv_heads, head_dim) bf16
        positions: (W,) int32 — position in the sequence (for masking)
    where W = head_count + tail_count.
    """
    head_count = int(window.head_token_count[seq_id].item())
    tail_count = int(window.tail_token_count[seq_id].item())
    total = int(window.total_seq_len[seq_id].item())

    head_keys = window.head_keys[seq_id, :head_count]
    head_values = window.head_values[seq_id, :head_count]

    if tail_count == 0:
        keys = head_keys
        values = head_values
        positions = torch.arange(
            head_count, dtype=torch.int32, device=window.head_keys.device,
        )
        return keys, values, positions

    if tail_count < window.tail_size:
        # Ring hasn't wrapped — entries live at slots [0, tail_count).
        tail_keys_ordered = window.tail_keys[seq_id, :tail_count]
        tail_values_ordered = window.tail_values[seq_id, :tail_count]
    else:
        # Ring is full. Oldest entry is at tail_write_ptr.
        ptr = int(window.tail_write_ptr[seq_id].item())
        if ptr == 0:
            tail_keys_ordered = window.tail_keys[seq_id]
            tail_values_ordered = window.tail_values[seq_id]
        else:
            tail_keys_ordered = torch.cat(
                (
                    window.tail_keys[seq_id, ptr:],
                    window.tail_keys[seq_id, :ptr],
                ),
                dim=0,
            )
            tail_values_ordered = torch.cat(
                (
                    window.tail_values[seq_id, ptr:],
                    window.tail_values[seq_id, :ptr],
                ),
                dim=0,
            )

    keys = torch.cat((head_keys, tail_keys_ordered), dim=0)
    values = torch.cat((head_values, tail_values_ordered), dim=0)
    head_positions = torch.arange(
        head_count, dtype=torch.int32, device=window.head_keys.device,
    )
    tail_positions = torch.arange(
        total - tail_count, total, dtype=torch.int32, device=window.head_keys.device,
    )
    positions = torch.cat((head_positions, tail_positions), dim=0)
    return keys, values, positions
```

- [ ] **Step 6.4: Run — test passes**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_hybrid_window.py -v 2>&1 | tail -10
```

Expected: 7 passed.

---

## Task 7: LSE-aware dense Pass 2 attention helper

**Files:**
- Modify: `vllm/v1/attention/ops/hybrid_window_cache.py`
- Modify: `tests/quantization/test_turboquant_hybrid_window.py`

The Pass 2 (bf16 dense attention over the window) needs to return both the
attention output AND the log-sum-exp of its attention logits, so we can
LSE-merge with Pass 1. `context_attention_fwd` doesn't return LSE, so we
write a small raw-PyTorch attention helper here. Window is ≤1536 tokens so
the cost is negligible (single query per sequence at decode time).

- [ ] **Step 7.1: Write failing unit test**

Append to `tests/quantization/test_turboquant_hybrid_window.py`:

```python
def test_dense_attention_with_lse_matches_reference():
    from vllm.v1.attention.ops.hybrid_window_cache import dense_attention_with_lse
    torch.manual_seed(0)
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    q = torch.randn(1, num_heads, head_dim, dtype=torch.float32, device="cuda")
    k = torch.randn(5, num_kv_heads, head_dim, dtype=torch.float32, device="cuda")
    v = torch.randn(5, num_kv_heads, head_dim, dtype=torch.float32, device="cuda")

    # Reference: upcast Q to per-head, GQA replicates K/V, softmax, @ V.
    q_r = q.squeeze(0)  # (num_heads, head_dim)
    kv_group = num_heads // num_kv_heads
    k_g = k.repeat_interleave(kv_group, dim=1)  # (5, num_heads, head_dim)
    v_g = v.repeat_interleave(kv_group, dim=1)
    scores = (q_r.unsqueeze(0) * k_g).sum(-1) / (head_dim ** 0.5)  # (5, num_heads)
    attn = torch.softmax(scores, dim=0)  # over tokens
    ref_out = (attn.unsqueeze(-1) * v_g).sum(0)  # (num_heads, head_dim)
    ref_lse = torch.logsumexp(scores / 1.0, dim=0)  # (num_heads,)

    out, lse = dense_attention_with_lse(
        query=q, keys=k, values=v,
        softmax_scale=1.0 / (head_dim ** 0.5),
    )
    assert out.shape == (1, num_heads, head_dim)
    assert lse.shape == (1, num_heads)
    torch.testing.assert_close(out.squeeze(0), ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(lse.squeeze(0), ref_lse, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 7.2: Run — expect ImportError**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_hybrid_window.py::test_dense_attention_with_lse_matches_reference -v 2>&1 | tail -6
```

Expected: ImportError.

- [ ] **Step 7.3: Implement dense_attention_with_lse**

Append to `vllm/v1/attention/ops/hybrid_window_cache.py`:

```python
def dense_attention_with_lse(
    *,
    query: torch.Tensor,          # (num_queries, num_heads, head_dim)
    keys: torch.Tensor,           # (num_kv_tokens, num_kv_heads, head_dim)
    values: torch.Tensor,         # (num_kv_tokens, num_kv_heads, head_dim)
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain-PyTorch attention that also returns log-sum-exp of logits.

    Supports grouped-query attention (num_heads divisible by num_kv_heads).
    Upcasts to fp32 internally for numerical stability, returns the output
    in the query's dtype.

    Returns:
        output:     (num_queries, num_heads, head_dim) same dtype as query.
        lse:        (num_queries, num_heads) float32.
    """
    num_queries, num_heads, head_dim = query.shape
    num_kv_tokens, num_kv_heads, _ = keys.shape
    assert num_heads % num_kv_heads == 0, (
        f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}."
    )
    kv_group = num_heads // num_kv_heads

    if num_kv_tokens == 0:
        # No keys to attend to — lse is -inf so the LSE merge treats this pass
        # as contributing nothing.
        out = torch.zeros_like(query)
        lse = torch.full(
            (num_queries, num_heads),
            float("-inf"),
            dtype=torch.float32,
            device=query.device,
        )
        return out, lse

    q_f = query.to(torch.float32)
    k_f = keys.to(torch.float32)
    v_f = values.to(torch.float32)

    # GQA replication.
    if kv_group > 1:
        k_f = k_f.repeat_interleave(kv_group, dim=1)
        v_f = v_f.repeat_interleave(kv_group, dim=1)

    # logits: (num_queries, num_heads, num_kv_tokens)
    logits = torch.einsum("qhd,khd->qhk", q_f, k_f) * softmax_scale
    lse = torch.logsumexp(logits, dim=-1)                # (num_queries, num_heads)
    attn = torch.softmax(logits, dim=-1)                 # (num_queries, num_heads, num_kv_tokens)
    out_f = torch.einsum("qhk,khd->qhd", attn, v_f)      # (num_queries, num_heads, head_dim)

    # The `softmax_scale` inside lse returns `log Σ exp(logits)` where logits
    # already include the scale. That's what the LSE-merge code expects.
    return out_f.to(query.dtype), lse
```

- [ ] **Step 7.4: Run — test passes**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_hybrid_window.py -v 2>&1 | tail -10
```

Expected: 8 passed.

---

## Task 8: LSE-merge smoke test

The attention-backend integration will use `vllm.v1.attention.ops.merge_attn_states.merge_attn_states`. We add a small sanity test here so we know our API understanding is right before wiring it into triton_attn.

**Files:**
- Modify: `tests/quantization/test_turboquant_hybrid_window.py`

- [ ] **Step 8.1: Write the test**

Append:

```python
def test_lse_merge_combines_disjoint_key_partitions_correctly():
    """Given a sequence of K/V split into two partitions, attention over
    the full sequence must equal LSE-merge of attention over each partition."""
    from vllm.v1.attention.ops.hybrid_window_cache import dense_attention_with_lse
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

    torch.manual_seed(42)
    num_queries, num_heads, head_dim = 1, 8, 16
    full_kv = 20
    split = 7

    q = torch.randn(num_queries, num_heads, head_dim, dtype=torch.float32, device="cuda")
    k = torch.randn(full_kv, num_heads, head_dim, dtype=torch.float32, device="cuda")
    v = torch.randn(full_kv, num_heads, head_dim, dtype=torch.float32, device="cuda")

    scale = 1.0 / (head_dim ** 0.5)

    # Ground truth: full attention in one go.
    full_out, _ = dense_attention_with_lse(
        query=q, keys=k, values=v, softmax_scale=scale,
    )

    # Split path: attention on [0, split) + [split, end), then LSE merge.
    a_out, a_lse = dense_attention_with_lse(
        query=q, keys=k[:split], values=v[:split], softmax_scale=scale,
    )
    b_out, b_lse = dense_attention_with_lse(
        query=q, keys=k[split:], values=v[split:], softmax_scale=scale,
    )

    merged = torch.zeros_like(full_out)
    merge_attn_states(
        output=merged,
        prefix_output=a_out,
        prefix_lse=a_lse,
        suffix_output=b_out,
        suffix_lse=b_lse,
    )

    torch.testing.assert_close(merged, full_out, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 8.2: Run**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_hybrid_window.py::test_lse_merge_combines_disjoint_key_partitions_correctly -v 2>&1 | tail -6
```

Expected: **pass** (no new implementation — merge_attn_states already exists).

If the test fails: the LSE convention in `merge_attn_states` may expect a
different layout of `lse` (e.g. `(num_heads, num_queries)` instead of
`(num_queries, num_heads)`). Check the source
`vllm/v1/attention/ops/triton_merge_attn_states.py` and the custom CUDA op at
`vllm/_custom_ops.py`. If the layout is transposed, transpose the LSE tensors
before calling merge_attn_states; adjust `dense_attention_with_lse` to match
the convention. Add a docstring note explaining the convention; re-run.

This is the single part of the design where the exact API shape of vLLM's
internal utility needs to be confirmed empirically. The test above is how we
confirm.

---

## Task 9: Wire the window cache into TurboQuant attention backend — allocation

**Files:**
- Modify: `vllm/v1/attention/backends/triton_attn.py`

This task adds the allocation of the bf16 window in the attention layer's
init path and plumbs through the config.

- [ ] **Step 9.1: Inspect the attention impl init**

Find where the per-layer turboquant attention state is allocated. Search for
the other turboquant caches:

```bash
grep -n "def __init__\|_turboquant_\|enable_turboquant\|hybrid_window" vllm/v1/attention/backends/triton_attn.py | head -30
```

Locate the `__init__` of the attention `TritonAttentionImpl` class (or equivalent). We'll add two new attributes.

- [ ] **Step 9.2: Thread the config**

In the attention impl's `__init__`, after the existing turboquant config wiring (likely inside an `if is_turboquant_kv_cache(self.kv_cache_dtype):` block), add:

```python
        self._hybrid_window_head_size = 0
        self._hybrid_window_tail_size = 0
        self._bf16_window: dict[tuple[str, int | None], "Bf16Window"] = {}

        if is_turboquant_kv_cache(self.kv_cache_dtype):
            cache_config = vllm_config.cache_config  # the same config read above
            self._hybrid_window_head_size = int(
                getattr(cache_config, "hybrid_window_head_size", 0)
            )
            self._hybrid_window_tail_size = int(
                getattr(cache_config, "hybrid_window_tail_size", 0)
            )
            if self.hybrid_window_enabled:
                logger.info_once(
                    "TurboQuant hybrid window enabled: head=%d, tail=%d",
                    self._hybrid_window_head_size,
                    self._hybrid_window_tail_size,
                    scope="local",
                )
```

If the attention impl doesn't already have a `vllm_config` or `cache_config`
reference in scope, walk up to how the existing turboquant init retrieved its
config and mirror that. Don't invent new config plumbing.

- [ ] **Step 9.3: Add the property**

Near the other helper properties on the class:

```python
    @property
    def hybrid_window_enabled(self) -> bool:
        return (
            self._hybrid_window_head_size > 0
            and self._hybrid_window_tail_size > 0
        )
```

- [ ] **Step 9.4: Add the import at file top**

Where the other `hybrid_window_cache` functions would be imported (near the existing turboquant imports):

```python
from vllm.v1.attention.ops.hybrid_window_cache import (
    Bf16Window,
    allocate_bf16_window,
    append_decode_to_window,
    dense_attention_with_lse,
    gather_window_kv,
    write_prefill_to_window,
)
```

- [ ] **Step 9.5: Verify import**

```bash
/root/vllm-venv/bin/python -c "from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl; print('ok')"
```

Expected: `ok`. If the class name differs in this fork, substitute the actual class name from your earlier grep.

---

## Task 10: Dual-write path — populate the window on kv_cache update

**Files:**
- Modify: `vllm/v1/attention/backends/triton_attn.py`

- [ ] **Step 10.1: Locate `do_kv_cache_update`**

```bash
grep -n "def do_kv_cache_update" vllm/v1/attention/backends/triton_attn.py
```

It's around line 1077. After the existing `turboquant_write_packed_kv` calls (lines 1105-1126), add the window write.

- [ ] **Step 10.2: Add window population after the packed write**

Inside `do_kv_cache_update`, in the `is_turboquant_kv_cache` branch, immediately before the `return` on line ~1127, insert:

```python
            if self.hybrid_window_enabled:
                # slot_mapping is (num_tokens,) with one slot per token. We
                # need per-sequence boundaries to split tokens back into
                # sequences. Pull those from the attention metadata.
                self._ensure_bf16_window(
                    device=key.device,
                    dtype=torch.bfloat16,
                    num_seqs=attn_metadata.seq_lens.shape[0],
                )
                window = self._bf16_window[(key.device.type, key.device.index)]
                # Key/value shapes here are (num_tokens, num_kv_heads, head_dim)
                # (bf16 already). Walk the per-sequence groups.
                query_start_loc_cpu = attn_metadata.query_start_loc_cpu
                seq_lens_cpu = attn_metadata.seq_lens_cpu
                for seq_id in range(query_start_loc_cpu.shape[0] - 1):
                    start = int(query_start_loc_cpu[seq_id].item())
                    end = int(query_start_loc_cpu[seq_id + 1].item())
                    n_new = end - start
                    if n_new == 0:
                        continue
                    seq_k = key[start:end]
                    seq_v = value[start:end]
                    if int(window.total_seq_len[seq_id].item()) == 0:
                        write_prefill_to_window(
                            window,
                            seq_id=seq_id,
                            keys=seq_k,
                            values=seq_v,
                        )
                    else:
                        for i in range(n_new):
                            append_decode_to_window(
                                window,
                                seq_id=seq_id,
                                key=seq_k[i],
                                value=seq_v[i],
                            )
```

**Important:** the loop assumes `attn_metadata` exists on `do_kv_cache_update`. If the current signature doesn't receive it, modify the signature (and every call site) to pass it. Search call sites:

```bash
grep -n "do_kv_cache_update" vllm/v1/attention/backends/triton_attn.py vllm/v1/worker/*.py
```

Thread `attn_metadata` through. Do not change any non-turboquant behaviour.

- [ ] **Step 10.3: Helper method to allocate lazily**

Also on the attention impl class:

```python
    def _ensure_bf16_window(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_seqs: int,
    ) -> None:
        cache_key = (device.type, device.index)
        existing = self._bf16_window.get(cache_key)
        if existing is not None and existing.num_seqs >= num_seqs:
            return
        # (Re)allocate to fit num_seqs. First call fits; later grows to
        # accommodate more sequences if scheduler batches more at once.
        new_window = allocate_bf16_window(
            num_seqs=num_seqs,
            head_size=self._hybrid_window_head_size,
            tail_size=self._hybrid_window_tail_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            device=device,
            dtype=dtype,
        )
        if existing is not None:
            # Preserve state for the sequences we already tracked.
            for attr in (
                "head_keys",
                "head_values",
                "tail_keys",
                "tail_values",
            ):
                getattr(new_window, attr)[: existing.num_seqs].copy_(
                    getattr(existing, attr)
                )
            for attr in (
                "head_token_count",
                "tail_write_ptr",
                "tail_token_count",
                "total_seq_len",
            ):
                getattr(new_window, attr)[: existing.num_seqs].copy_(
                    getattr(existing, attr)
                )
        self._bf16_window[cache_key] = new_window
```

- [ ] **Step 10.4: Quick syntactic sanity**

```bash
/root/vllm-venv/bin/python -c "from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl; print('ok')"
```

Expected: `ok`.

- [ ] **Step 10.5: Existing tq tests still pass**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant.py -q 2>&1 | tail -5
```

Expected: still 59 passed, 2 pre-existing failures. If anything else broke, revert and investigate.

---

## Task 11: Two-pass hybrid attention in `_forward_turboquant`

**Files:**
- Modify: `vllm/v1/attention/backends/triton_attn.py`

- [ ] **Step 11.1: Locate the decode branch of `_forward_turboquant`**

At `_forward_turboquant` (starts line ~1338), three paths currently exist:
dense prefill (1351), fallback (1367), and the full turboquant path (1378+).
The hybrid path hooks into the full turboquant path only (decode + chunked
prefill). Read lines 1338-1570 to understand the existing control flow
before editing.

- [ ] **Step 11.2: Add the hybrid dispatch**

In `_forward_turboquant`, after the line `assert self.turboquant_bits is not None` (line ~1379) and before the existing tables setup, branch:

```python
        if self.hybrid_window_enabled:
            return self._forward_turboquant_hybrid(
                query=query,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                output=output,
                attn_metadata=attn_metadata,
            )
```

The dense-prefill fast path above stays untouched because it short-circuits
before this. During pure prefill the existing dense path runs; the hybrid path
only kicks in when the backend is reading from the cache (decode + chunked
prefill).

- [ ] **Step 11.3: Implement `_forward_turboquant_hybrid`**

Append to the same class (below `_forward_turboquant`):

```python
    def _forward_turboquant_hybrid(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> torch.Tensor:
        """Two-pass hybrid attention: tq35 over the middle region, dense
        bf16 over the head+tail window, LSE-merged.

        For sequences where seq_len <= head_size + tail_size there is no
        middle region; we skip Pass 1 entirely and return the dense output
        directly.
        """
        device = query.device
        window = self._bf16_window.get((device.type, device.index))
        if window is None:
            # Allocation hasn't happened yet (first forward without a prior
            # do_kv_cache_update) — fall back to the standard tq35 path.
            return self._forward_turboquant_full(
                query=query,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                output=output,
                attn_metadata=attn_metadata,
            )

        head_size = self._hybrid_window_head_size
        tail_size = self._hybrid_window_tail_size
        query_start_loc_cpu = attn_metadata.query_start_loc_cpu
        seq_lens_cpu = attn_metadata.seq_lens_cpu
        num_seqs = seq_lens_cpu.shape[0]

        # For each sequence in the batch:
        #   1. Assemble head+tail K/V from window.
        #   2. Compute dense attention over the window (with LSE).
        #   3. Compute tq35 attention over the middle region (with LSE).
        #      — If middle region empty, dense result is final.
        #   4. Merge via LSE.
        # We do this per-seq because the middle/window split is sequence-
        # specific. For V1, prefer simple per-seq loops; optimise later.

        for seq_id in range(num_seqs):
            start = int(query_start_loc_cpu[seq_id].item())
            end = int(query_start_loc_cpu[seq_id + 1].item())
            if end == start:
                continue
            seq_len = int(seq_lens_cpu[seq_id].item())
            q_seq = query[start:end]                       # (Lq, num_heads, head_dim)

            window_k, window_v, window_positions = gather_window_kv(
                window, seq_id=seq_id
            )

            # Pass 2: dense bf16 over the window.
            dense_out, dense_lse = dense_attention_with_lse(
                query=q_seq,
                keys=window_k,
                values=window_v,
                softmax_scale=self.scale,
            )

            middle_start = head_size
            middle_end = max(head_size, seq_len - tail_size)
            if middle_end <= middle_start:
                # No middle — attention is purely the window pass.
                output[start:end] = dense_out
                continue

            # Pass 1: tq35 over the middle. Use the existing machinery, but
            # mask seq_lens to the middle range. Reusing the cascade
            # approach: treat `suffix_kv_lens` as the middle length, with
            # `query_positions` shifted so position 0 == start of middle.
            # ... (this is the section that has to match whatever the
            # existing turboquant decode accepts — see note below.)
            raise NotImplementedError(
                "Hybrid middle-pass integration requires calling the existing "
                "turboquant_decode_attention_fwd with per-seq middle offsets. "
                "See implementation notes in the plan and in the existing "
                "cascade branch (_forward_turboquant lines 1396-1517) for "
                "the argument shape required."
            )

        return output
```

- [ ] **Step 11.4: Finish the middle-pass wiring**

This is the load-bearing piece. Look at the existing cascade path
(`_forward_turboquant` lines 1396-1517) — it already demonstrates how to
split a sequence into two key-partitions and LSE-merge them. In cascade
attention, Pass 1 is a common prefix and Pass 2 is a suffix, both using the
turboquant decode kernel. Our hybrid is structurally identical except:

- **Prefix (our Pass 1)** is the **middle** `[head_size, seq_len - tail_size)`,
  using tq35. In the cascade code, this corresponds to
  `common_prefix_len = seq_len - tail_size - head_size` with a shifted
  `prefix_seq_lens` and `token_query_positions` so the kernel interprets
  offsets relative to the middle.
- **Suffix (our Pass 2)** is head+tail using dense — different from cascade
  which uses turboquant for the suffix too.

Replace the `raise NotImplementedError` with the corresponding logic: copy
the cascade-attention section lines 1396-1517 verbatim and **keep only the
Pass 1 part** (the prefix) while swapping the decode's `seq_lens` /
`token_kv_lens` / `token_query_positions` to represent the middle. Replace
the cascade's Pass 2 with our `dense_attention_with_lse` (already computed
as `dense_out`, `dense_lse` above).

The merge call then becomes:

```python
            merged_out = torch.empty_like(dense_out)
            merge_attn_states(
                output=merged_out,
                prefix_output=middle_out,
                prefix_lse=middle_lse,
                suffix_output=dense_out,
                suffix_lse=dense_lse,
            )
            output[start:end] = merged_out
```

This integration is where the most carefully-reviewed vLLM-internal reading
is required. **Treat this task as the place most likely to need a second
pass with fresh eyes.**

- [ ] **Step 11.5: Syntax and smoke check**

```bash
/root/vllm-venv/bin/python -c "from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl; print('ok')"
```

Expected: `ok`.

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant.py -q 2>&1 | tail -5
```

Expected: still 59 passed, 2 pre-existing failures, no new breakage (hybrid
flag defaults to off, so existing paths unchanged).

- [ ] **Step 11.6: Extract `_forward_turboquant_full`**

If there's no existing `_forward_turboquant_full` method to call from the
dispatch in Step 11.2 (there isn't — I use that name aspirationally), extract
the body of the current `_forward_turboquant` (lines 1379-1568) into a new
method called `_forward_turboquant_full`, and have the top-level
`_forward_turboquant` be the dispatch wrapper that picks between dense
prefill / hybrid / full. This is a pure refactor — no behaviour change — done
to keep the hybrid path isolated.

---

## Task 12: Short-sequence equivalence integration test

**Files:**
- Modify: `tests/quantization/test_turboquant_hybrid_window.py`

For `seq_len <= head_size + tail_size`, the hybrid path should be
bit-identical to pure-bf16 baseline (both paths use dense attention over all
tokens — no quantization involved).

- [ ] **Step 12.1: Write the test**

Append:

```python
import pytest

pytestmark = pytest.mark.gpu

# The test marker ensures skipping when CUDA isn't available. If the repo
# doesn't have this marker, remove the line.


def test_hybrid_short_sequence_matches_bf16_baseline():
    """With seq_len <= head+tail, hybrid mode must match bf16 baseline byte-for-byte."""
    from vllm import LLM, SamplingParams

    SNAP = (
        "/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    )
    META = "/tmp/qwen2_5_turboquant35.json"

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nWrite one short sentence about the ocean.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    sp = SamplingParams(temperature=0.0, max_tokens=32, seed=0)

    # Baseline: pure bf16 KV.
    llm_base = LLM(
        model=SNAP, dtype="bfloat16", max_model_len=1024,
        max_num_seqs=1, gpu_memory_utilization=0.70,
        attention_backend="TRITON_ATTN",
        kv_cache_dtype="auto",
    )
    out_base = llm_base.generate([prompt], sp)[0].outputs[0].text
    del llm_base
    import torch; torch.cuda.empty_cache()

    # Hybrid: tq35 + head=256/tail=256 (prompt is <50 tokens so everything is in window).
    llm_hy = LLM(
        model=SNAP, dtype="bfloat16", max_model_len=1024,
        max_num_seqs=1, gpu_memory_utilization=0.70,
        attention_backend="TRITON_ATTN",
        kv_cache_dtype="turboquant35",
        enable_turboquant=True,
        turboquant_metadata_path=META,
        hybrid_window_head_size=256,
        hybrid_window_tail_size=256,
    )
    out_hy = llm_hy.generate([prompt], sp)[0].outputs[0].text
    del llm_hy
    torch.cuda.empty_cache()

    # Expect identical output: every token read from bf16 window — tq35 never
    # used because seq_len (~40 tokens) < head_size (256).
    assert out_hy == out_base, (
        f"Hybrid short-sequence output differs from baseline:\n"
        f"  base: {out_base!r}\n"
        f"  hy  : {out_hy!r}"
    )
```

Note: the test construct `hybrid_window_head_size=256, hybrid_window_tail_size=256`
on `LLM` depends on vLLM surfacing these as kwargs. If the LLM wrapper doesn't
expose cache-config fields directly, use `engine_args=...` or `cache_config=...`
— check an existing test using a similar pattern (search for `enable_turboquant`
in existing tests).

- [ ] **Step 12.2: Run**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_hybrid_window.py::test_hybrid_short_sequence_matches_bf16_baseline -v 2>&1 | tail -10
```

Expected: pass. If it fails with "hybrid_window_head_size not found", fix the
LLM kwarg surfacing by mirroring how `enable_turboquant` flows through
`LLM.__init__` → `EngineArgs` → `CacheConfig`.

---

## Task 13: Long-prompt coherence integration test

**Files:**
- Modify: `tests/quantization/test_turboquant_hybrid_window.py`

The smoking-gun test from the investigation: on the 8 glaive prompts that
currently produce degenerate output with pure tq35, hybrid mode should
produce coherent output.

- [ ] **Step 13.1: Write the test**

Append:

```python
def test_hybrid_long_prompts_produce_coherent_output():
    """Run the smoking-gun prompt set from investigation scripts/tq_diag2.py
    with hybrid enabled. Under greedy decode, outputs must be structurally
    reasonable (unique-token ratio > 0.5, newline fraction < 0.1)."""
    from vllm import LLM, SamplingParams
    import json, statistics

    SNAP = (
        "/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    )
    META = "/tmp/qwen2_5_turboquant35.json"

    prompts: list[str] = []
    with open("/tmp/small_glaive_qwen25.jsonl") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
    prompts = prompts[:8]
    sp = SamplingParams(temperature=0.0, max_tokens=48, seed=0, ignore_eos=True)

    llm = LLM(
        model=SNAP, dtype="bfloat16", max_model_len=2048,
        max_num_seqs=4, gpu_memory_utilization=0.75,
        attention_backend="TRITON_ATTN",
        kv_cache_dtype="turboquant35",
        enable_turboquant=True,
        turboquant_metadata_path=META,
        hybrid_window_head_size=128,
        hybrid_window_tail_size=128,
    )
    texts = [o.outputs[0].text for o in llm.generate(prompts, sp)]
    del llm
    import torch; torch.cuda.empty_cache()

    def uniq_ratio(t: str) -> float:
        toks = t.split()
        return len(set(toks)) / len(toks) if toks else 0.0

    def newline_frac(t: str) -> float:
        return t.count("\n") / max(len(t), 1)

    mean_u = statistics.mean(uniq_ratio(t) for t in texts)
    mean_nl = statistics.mean(newline_frac(t) for t in texts)

    print(f"\nhybrid long-prompt mean_uniq={mean_u:.2f} mean_nl={mean_nl:.2f}")
    for i, t in enumerate(texts):
        print(f"  [{i}] {t[:120]!r}")

    # The acceptance criteria are deliberately loose: we want to see not-
    # degenerate output. Pure-tq35 today gives mean_uniq=0.45, mean_nl=0.37.
    # bf16 gives mean_uniq=0.80, mean_nl=0.02.
    # Hybrid should land much closer to bf16.
    assert mean_u > 0.5, f"unique-token ratio {mean_u} is still degenerate"
    assert mean_nl < 0.1, f"newline fraction {mean_nl} is still degenerate"
```

- [ ] **Step 13.2: Run**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_hybrid_window.py::test_hybrid_long_prompts_produce_coherent_output -v -s 2>&1 | tail -30
```

Expected: pass (mean_uniq > 0.5, mean_nl < 0.1). If the test fails with
degenerate output, the middle-pass wiring in Task 11 has a bug — debug by:

1. Temporarily setting `hybrid_window_head_size=2048, tail_size=0` so the
   entire prompt is in the window. Result should match bf16 (proves dense
   pass correct).
2. Setting `head_size=0, tail_size=2048`. Still dense-only. If broken,
   ring-buffer path is wrong.
3. With a small middle region, test merge layer by layer.

---

## Task 14: Full regression + perf sanity

**Files:** no code changes; verification only.

- [ ] **Step 14.1: Full turboquant unit suite**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant.py tests/quantization/test_turboquant_hybrid_window.py -v 2>&1 | tail -15
```

Expected: the 59 pre-existing passes hold, all new hybrid tests pass, only
the 2 pre-existing `test_turboquant_prefill_reads_quantized_cache` failures
remain.

- [ ] **Step 14.2: Run an end-to-end A/B benchmark against hybrid**

```bash
SNAP=/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
META=/tmp/qwen2_5_turboquant35.json \
PROMPTS=/tmp/small_glaive_qwen25.jsonl \
RESULT_DIR=$PWD/benchmarks/results_qwen2_5_hybrid \
MAX_MODEL_LEN=2048 MAX_NUM_SEQS=4 MAX_CONCURRENCY=4 NUM_PROMPTS=64 OUTPUT_LEN=128 \
GPU_MEMORY_UTILIZATION=0.80 \
TURBOQUANT_HEAD_WINDOW_SIZE=128 TURBOQUANT_TAIL_WINDOW_SIZE=128 \
bash benchmarks/benchmark_turboquant_vs_baseline_qwen3_5.sh
```

Note: the bench script may not pass through `TURBOQUANT_HEAD_WINDOW_SIZE` —
if not, edit `benchmarks/benchmark_turboquant_vs_baseline_qwen3_5.sh:167-170`
to add the flags to the treatment arm's `vllm serve` invocation:

```bash
    --kv-cache-dtype turboquant35 \
    --enable-turboquant \
    --turboquant-metadata-path "$META" \
    --turboquant-head-window-size "${TURBOQUANT_HEAD_WINDOW_SIZE:-1024}" \
    --turboquant-tail-window-size "${TURBOQUANT_TAIL_WINDOW_SIZE:-512}"
```

Expected: `comparison.md` now shows substantially better Jaccard than the
pure-tq35 run (target > 0.5 at same prompt set), confirming end-to-end.

---

## Self-Review

**Spec coverage:**

| Spec section | Covered by |
|---|---|
| Config defaults head=1024 / tail=512 as CLI flags | Tasks 1, 2 |
| Dual-cache architecture | Tasks 3-6, 10 |
| LSE-merged 2-pass attention | Tasks 7, 8, 11 |
| Prefill path: write last-N to tail, no ring-buffer at prefill | Task 4 |
| Decode path: ring-buffer tail + head if not full | Task 5 |
| Short-sequence passthrough identical to bf16 | Task 12 (asserted) |
| Long-prompt coherence | Task 13 |
| Backward compat when hybrid disabled | Task 10 Step 10.5 |
| Tests cover ring eviction, LSE merge, integration | Tasks 5, 8, 13 |
| Memory accounting | Noted in spec; no explicit task (optional ad-hoc) |

**Placeholder scan:** one deliberate "investigate and match the cascade path"
in Task 11 Step 11.4 — this is the single area where I cannot write
byte-exact code without reading more vLLM internals than was reasonable in
the session. The step is concrete (specifies exactly what to read, what to
keep, what to replace) but requires engineer judgment when copying from the
cascade code.

**Type consistency:** `Bf16Window` dataclass fields match across Tasks 3-6.
`dense_attention_with_lse` signature consistent in Tasks 7, 8, 11.
`write_prefill_to_window` / `append_decode_to_window` / `gather_window_kv`
signatures stable across producer and consumer tasks.

**Scope check:** single implementation plan is appropriate. No subsystem
decomposition needed.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-24-turboquant-hybrid-window.md`.

**Two execution options:**

**1. Subagent-Driven (recommended).** Dispatch fresh subagent per task with the full task text + context. Review between tasks. Fast iteration.

**2. Inline Execution.** Execute tasks in this session using executing-plans, with checkpoints after Tasks 8 (all unit tests passing), 11 (attention-backend wiring done), and 14 (full regression).

Task 11 is the single highest-risk task and will need the closest review
regardless of execution mode. Suggest an extra manual pass there.
