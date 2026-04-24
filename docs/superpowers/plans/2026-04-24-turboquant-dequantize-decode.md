# TurboQuant Dequantize-and-Attend Decode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace this fork's broken fused-kernel TurboQuant decode path with a "dequantize then run standard attention" path, matching what 0xSero/turboquant proves works on Qwen3.5. Optionally add a small uncompressed recent-token ring buffer to further reduce noise on high-attention positions.

**Architecture:** Keep the fork's existing tq35 packed cache unchanged. In `_forward_turboquant`, add a new decode path that (a) gathers packed bytes from the paged cache, (b) calls this fork's existing `dequantize_turboquant_vectors` to produce bf16 K/V, (c) optionally overrides the last N positions with uncompressed K/V from a ring buffer, (d) runs SDPA. No LSE-merge, no fused-kernel calls, no changes to the write path or calibration.

**Tech Stack:** Python 3.12, PyTorch (bf16), vLLM v1 attention backends. All work in `/workspace/vllm-turboquant/` on `/root/vllm-venv`.

**Commit policy (user override):** NO commits. NO git ops of any kind. Leave all work uncommitted for the user to review.

**Supersedes:** `docs/superpowers/plans/2026-04-24-turboquant-hybrid-window.md` (the earlier head+tail LSE-merge plan). That plan was on the right track diagnostically but was more complex than the fix actually needs.

---

## File Structure

**Created:**
- `vllm/v1/attention/ops/turboquant_recent_ring.py` — per-layer uncompressed ring buffer for recent K/V. ~120 lines.
- `tests/quantization/test_turboquant_dequant_decode.py` — unit + integration tests.

**Modified:**
- `vllm/config/cache.py` — add `turboquant_recent_ring_capacity: int = 64` (0 disables).
- `vllm/engine/arg_utils.py` — surface `--turboquant-recent-ring-capacity`.
- `vllm/v1/attention/backends/triton_attn.py`:
  - Add `_forward_turboquant_dequant` method (~80 lines).
  - Flip `_forward_turboquant` decode dispatch to call it instead of `turboquant_decode_attention_fwd`.
  - Add `_gather_packed_blocks` helper + ring buffer integration in `do_kv_cache_update`.

**Untouched:**
- `vllm/v1/attention/ops/turboquant_kv_cache.py` (packing math unchanged).
- `vllm/v1/attention/ops/triton_turboquant_kv_update.py` (write kernel unchanged).
- `vllm/v1/attention/ops/triton_turboquant_decode.py` (fused decode kernel; dead code after this change but left for potential future fix).
- `benchmarks/generate_turboquant_metadata.py` (calibration unchanged).
- Existing `calibration/*.json` artifacts (still valid).

---

## Shared fixtures

- Repo root: `/workspace/vllm-turboquant`
- Main venv: `/root/vllm-venv/bin/python`, `/root/vllm-venv/bin/pytest`
- Small model snapshot: `/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- Small-model calibration: `/tmp/qwen2_5_turboquant35.json`
- Small-model prompts: `/tmp/small_glaive_qwen25.jsonl`

If any are missing, regenerate via the recipe in the investigation report's follow-up section.

---

## Task 1: Config flag for the ring buffer

**Files:**
- Modify: `vllm/config/cache.py`

- [ ] **Step 1.1: Read current CacheConfig layout**

```bash
grep -n "enable_turboquant\|turboquant_metadata_path\|class CacheConfig\|_validate_turboquant" vllm/config/cache.py | head -10
```

Note the location of existing turboquant fields and the `_validate_turboquant` model_validator.

- [ ] **Step 1.2: Add the field**

In `CacheConfig`, right after `turboquant_metadata_path`, insert:

```python
    turboquant_recent_ring_capacity: int = 64
    """Number of most-recent tokens per sequence to keep in an uncompressed
    bf16 ring buffer alongside the turboquant packed cache. Used at decode
    time to override the dequantized values for positions where attention
    mass concentrates. Set to 0 to disable the ring buffer (decode still
    works — pure dequant path — just slightly lossier on recent tokens)."""
```

- [ ] **Step 1.3: Validate in `_validate_turboquant`**

Append to the model_validator body, before `return self`:

```python
        if self.turboquant_recent_ring_capacity < 0:
            raise ValueError(
                "turboquant_recent_ring_capacity must be >= 0; got "
                f"{self.turboquant_recent_ring_capacity}."
            )
```

- [ ] **Step 1.4: Verify**

```bash
/root/vllm-venv/bin/python -c "
from vllm.config.cache import CacheConfig
f = CacheConfig.model_fields['turboquant_recent_ring_capacity']
print('default:', f.default)
print('ok')
"
```

Expected: `default: 64` and `ok`.

---

## Task 2: Plumb CLI arg

**Files:**
- Modify: `vllm/engine/arg_utils.py`

- [ ] **Step 2.1: Locate the turboquant arg block**

```bash
grep -n "turboquant-metadata-path\|turboquant_metadata_path" vllm/engine/arg_utils.py | head -5
```

- [ ] **Step 2.2: Add the argparse entry**

After the `--turboquant-metadata-path` `add_argument` block, add:

```python
        parser.add_argument(
            "--turboquant-recent-ring-capacity",
            type=int,
            default=64,
            help=(
                "Number of most-recent tokens per sequence kept uncompressed "
                "alongside the turboquant cache. 0 disables the ring buffer."
            ),
        )
```

Match the surrounding indentation/style exactly.

- [ ] **Step 2.3: Thread to CacheConfig**

Find where `CacheConfig(...)` is constructed from parsed args. After `turboquant_metadata_path=args.turboquant_metadata_path`, add:

```python
            turboquant_recent_ring_capacity=args.turboquant_recent_ring_capacity,
```

- [ ] **Step 2.4: Verify**

```bash
/root/vllm-venv/bin/python -c "
import argparse
from vllm.engine.arg_utils import EngineArgs
p = argparse.ArgumentParser()
EngineArgs.add_cli_args(p)
args = p.parse_args(['--model=Qwen/Qwen2.5-0.5B-Instruct', '--turboquant-recent-ring-capacity=32'])
print('ring:', args.turboquant_recent_ring_capacity)
"
```

Expected: `ring: 32`. If the hook method differs, use the earlier grep output.

---

## Task 3: Recent-token ring buffer module

**Files:**
- Create: `vllm/v1/attention/ops/turboquant_recent_ring.py`

- [ ] **Step 3.1: Create the module**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-(layer, seq) ring buffer of uncompressed recent K/V.

Used alongside the turboquant packed cache so the newest N tokens — where
attention mass concentrates — are available at decode time without going
through the (slightly lossy) dequantize path.

The ring is position-based, not time-based: slot `(count % capacity)` is
the next write target. After the ring fills, writes overwrite the oldest
entry. `gather_recent` returns the entries in sequence-position order,
regardless of the physical slot layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RecentRing:
    """Per-layer uncompressed bf16 K/V ring buffer.

    Shapes (one layer, all sequences):
        keys:        (num_seqs, capacity, num_kv_heads, head_dim)  bf16
        values:      (num_seqs, capacity, num_kv_heads, head_dim)  bf16
        write_ptr:   (num_seqs,) int32 — next slot to overwrite.
        fill_count:  (num_seqs,) int32 — populated slots [0, capacity].
        total_appends: (num_seqs,) int32 — cumulative appends
                                          (= min(seq_len, ring has been populated to)).
    """

    keys: torch.Tensor
    values: torch.Tensor
    write_ptr: torch.Tensor
    fill_count: torch.Tensor
    total_appends: torch.Tensor

    @property
    def capacity(self) -> int:
        return self.keys.shape[1]

    @property
    def num_seqs(self) -> int:
        return self.keys.shape[0]


def allocate_recent_ring(
    *,
    num_seqs: int,
    capacity: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> RecentRing:
    if capacity <= 0:
        raise ValueError(f"ring capacity must be > 0; got {capacity}")
    shape = (num_seqs, capacity, num_kv_heads, head_dim)
    return RecentRing(
        keys=torch.zeros(shape, dtype=dtype, device=device),
        values=torch.zeros(shape, dtype=dtype, device=device),
        write_ptr=torch.zeros(num_seqs, dtype=torch.int32, device=device),
        fill_count=torch.zeros(num_seqs, dtype=torch.int32, device=device),
        total_appends=torch.zeros(num_seqs, dtype=torch.int32, device=device),
    )


def append_recent(
    ring: RecentRing,
    *,
    seq_id: int,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Append one token's K/V (shape: (num_kv_heads, head_dim)) for one seq.

    Overwrites the oldest slot when the ring is full.
    """
    assert key.shape == value.shape
    capacity = ring.capacity
    ptr = int(ring.write_ptr[seq_id].item())
    ring.keys[seq_id, ptr].copy_(key)
    ring.values[seq_id, ptr].copy_(value)
    ring.write_ptr[seq_id] = (ptr + 1) % capacity
    fc = int(ring.fill_count[seq_id].item())
    if fc < capacity:
        ring.fill_count[seq_id] = fc + 1
    ring.total_appends[seq_id] = ring.total_appends[seq_id] + 1


def write_prefill_tail(
    ring: RecentRing,
    *,
    seq_id: int,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Seed the ring from a prefill of length L by keeping the last min(L, capacity) tokens.

    Shapes: (L, num_kv_heads, head_dim) for keys and values.
    """
    assert keys.shape == values.shape
    assert keys.dim() == 3
    L = keys.shape[0]
    capacity = ring.capacity
    n = min(L, capacity)
    if n == 0:
        ring.write_ptr[seq_id] = 0
        ring.fill_count[seq_id] = 0
        ring.total_appends[seq_id] = L
        return
    ring.keys[seq_id, :n].copy_(keys[L - n : L])
    ring.values[seq_id, :n].copy_(values[L - n : L])
    ring.write_ptr[seq_id] = n % capacity
    ring.fill_count[seq_id] = n
    ring.total_appends[seq_id] = L


def gather_recent(
    ring: RecentRing,
    *,
    seq_id: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return the populated ring entries for one seq in insertion order.

    Returns:
        keys:    (n, num_kv_heads, head_dim) bf16
        values:  (n, num_kv_heads, head_dim) bf16
        n:       number of populated entries (0..capacity)
    """
    n = int(ring.fill_count[seq_id].item())
    capacity = ring.capacity
    if n == 0:
        empty = ring.keys.new_zeros((0, ring.keys.shape[2], ring.keys.shape[3]))
        return empty, empty, 0
    if n < capacity:
        return ring.keys[seq_id, :n], ring.values[seq_id, :n], n
    # Ring full — oldest entry is at write_ptr.
    ptr = int(ring.write_ptr[seq_id].item())
    if ptr == 0:
        return ring.keys[seq_id], ring.values[seq_id], capacity
    keys_ordered = torch.cat(
        (ring.keys[seq_id, ptr:], ring.keys[seq_id, :ptr]), dim=0
    )
    values_ordered = torch.cat(
        (ring.values[seq_id, ptr:], ring.values[seq_id, :ptr]), dim=0
    )
    return keys_ordered, values_ordered, capacity
```

- [ ] **Step 3.2: Import sanity**

```bash
/root/vllm-venv/bin/python -c "
import torch
from vllm.v1.attention.ops.turboquant_recent_ring import (
    allocate_recent_ring, append_recent, write_prefill_tail, gather_recent,
)
r = allocate_recent_ring(num_seqs=2, capacity=4, num_kv_heads=1, head_dim=2, device=torch.device('cuda'))
print(r.keys.shape, r.keys.dtype, r.fill_count.tolist())
"
```

Expected: `torch.Size([2, 4, 1, 2]) torch.bfloat16 [0, 0]`.

---

## Task 4: Unit tests for the ring buffer

**Files:**
- Create: `tests/quantization/test_turboquant_dequant_decode.py`

- [ ] **Step 4.1: Initial test file with ring-buffer coverage**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the dequantize-then-attend TurboQuant decode path."""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops.turboquant_recent_ring import (
    allocate_recent_ring,
    append_recent,
    gather_recent,
    write_prefill_tail,
)


def _make_ring(num_seqs=1, cap=4, kv=1, dim=2):
    return allocate_recent_ring(
        num_seqs=num_seqs, capacity=cap, num_kv_heads=kv, head_dim=dim,
        device=torch.device("cuda"),
    )


def test_append_fills_then_wraps_ring_buffer():
    r = _make_ring(cap=3)
    for tok in [10, 20, 30, 40, 50]:
        k = torch.full((1, 2), tok, dtype=torch.bfloat16, device="cuda")
        v = k.clone() + 100
        append_recent(r, seq_id=0, key=k, value=v)
    assert int(r.fill_count[0].item()) == 3
    assert int(r.total_appends[0].item()) == 5
    keys, values, n = gather_recent(r, seq_id=0)
    assert n == 3
    # After wrapping: 30, 40, 50 remain in position order.
    first_coord = [float(x) for x in keys[:, 0, 0].tolist()]
    assert first_coord == [30.0, 40.0, 50.0]


def test_write_prefill_tail_keeps_last_n():
    r = _make_ring(cap=3)
    L = 7
    k = torch.arange(L * 2, dtype=torch.bfloat16, device="cuda").reshape(L, 1, 2)
    v = k.clone() + 100
    write_prefill_tail(r, seq_id=0, keys=k, values=v)
    assert int(r.fill_count[0].item()) == 3
    assert int(r.total_appends[0].item()) == 7
    keys, _, n = gather_recent(r, seq_id=0)
    assert n == 3
    assert torch.equal(keys, k[4:7])


def test_append_after_prefill_tail_wraps_correctly():
    r = _make_ring(cap=3)
    L = 2
    k = torch.arange(L * 2, dtype=torch.bfloat16, device="cuda").reshape(L, 1, 2)
    v = k.clone() + 100
    write_prefill_tail(r, seq_id=0, keys=k, values=v)
    # Ring now has 2 entries. Append two more — second append should start wrapping.
    for tok in [100, 200]:
        dk = torch.full((1, 2), tok, dtype=torch.bfloat16, device="cuda")
        dv = dk.clone() + 1
        append_recent(r, seq_id=0, key=dk, value=dv)
    assert int(r.fill_count[0].item()) == 3
    assert int(r.total_appends[0].item()) == 4
    keys, _, _ = gather_recent(r, seq_id=0)
    # Expected order: original [L-2], original [L-1], then 100, 200 overwriting
    # the oldest. After 4 total appends with cap=3, the ring contains the last
    # 3: [original last, 100, 200].
    first_coord = [float(x) for x in keys[:, 0, 0].tolist()]
    assert first_coord[-1] == 200.0
    assert first_coord[-2] == 100.0
```

- [ ] **Step 4.2: Run**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_dequant_decode.py -v 2>&1 | tail -10
```

Expected: 3 passed.

---

## Task 5: Verify dequantize+SDPA on real K/V matches bf16 attention

**Files:**
- Modify: `tests/quantization/test_turboquant_dequant_decode.py`

This test establishes the ground truth: that `dequantize_turboquant_vectors` + SDPA reproduces bf16 attention to within acceptable tolerance. If this test fails, the dequant path is unfixable and we stop.

- [ ] **Step 5.1: Add the test**

Append:

```python
def test_dequantize_then_sdpa_matches_bf16_attention_closely():
    """Round-trip test: quantize K/V, dequantize, run SDPA, compare to
    running SDPA directly on the original bf16 K/V. Output should be very
    close (cos-sim > 0.95) because dequant preserves direction (cos-sim
    0.99 per vector per the investigation report)."""
    import torch.nn.functional as F
    from vllm.v1.attention.ops.turboquant_kv_cache import (
        build_turboquant_outlier_masks,
        dequantize_turboquant_vectors,
        get_turboquant_centroids,
        get_turboquant_layout,
        get_turboquant_qjl_matrix,
        get_turboquant_rotation,
        quantize_turboquant_vectors,
    )

    torch.manual_seed(0)
    head_dim = 64
    num_kv_heads = 2
    recipe = "turboquant35"
    device = torch.device("cuda")

    # Synthetic K/V with realistic scale.
    seq_len = 128
    num_heads = 8  # 4x GQA
    K = torch.randn(seq_len, num_kv_heads, head_dim, device=device) * 2.0
    V = torch.randn(seq_len, num_kv_heads, head_dim, device=device) * 2.0
    Q = torch.randn(1, num_heads, head_dim, device=device)

    # Build tables and masks.
    layout = get_turboquant_layout(recipe, head_dim)
    rotations = (
        get_turboquant_rotation(device, layout.groups[0].dim, seed_offset=101),
        get_turboquant_rotation(device, layout.groups[1].dim, seed_offset=211),
    )
    qjl_matrices = (
        get_turboquant_qjl_matrix(device, layout.groups[0].dim, seed_offset=307),
        get_turboquant_qjl_matrix(device, layout.groups[1].dim, seed_offset=401),
    )
    centroids = {
        g.mse_bits: get_turboquant_centroids(device, g.dim, g.mse_bits)
        for g in layout.groups if g.mse_bits > 0
    }
    masks_k = build_turboquant_outlier_masks(K.float(), recipe)
    masks_v = build_turboquant_outlier_masks(V.float(), recipe)

    # Quantize + dequantize K (V uses its own mask).
    packed_K = quantize_turboquant_vectors(
        K.float(), recipe, rotations, qjl_matrices, centroids, masks_k,
    )
    packed_V = quantize_turboquant_vectors(
        V.float(), recipe, rotations, qjl_matrices, centroids, masks_v,
    )
    K_dq = dequantize_turboquant_vectors(
        packed_K, recipe, head_dim,
        rotations, qjl_matrices, centroids, masks_k, torch.bfloat16,
    )
    V_dq = dequantize_turboquant_vectors(
        packed_V, recipe, head_dim,
        rotations, qjl_matrices, centroids, masks_v, torch.bfloat16,
    )

    # SDPA helper: (batch=1, heads, tokens, dim) layout with GQA repeat.
    def sdpa(q_, k_, v_):
        # q: (1, Hq, Dq). k,v: (Lk, Hkv, D). Repeat KV heads to Hq.
        k_rep = k_.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_rep = v_.repeat_interleave(num_heads // num_kv_heads, dim=1)
        q_t = q_.transpose(0, 1).unsqueeze(0)           # (1, H, 1, D)
        k_t = k_rep.transpose(0, 1).unsqueeze(0)        # (1, H, L, D)
        v_t = v_rep.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q_t.float(), k_t.float(), v_t.float())
        return out.squeeze(0).transpose(0, 1).to(torch.bfloat16)

    out_ref = sdpa(Q.to(torch.bfloat16), K.to(torch.bfloat16), V.to(torch.bfloat16))
    out_dq = sdpa(Q.to(torch.bfloat16), K_dq, V_dq)

    # Cosine similarity on the attention output per head.
    cos = F.cosine_similarity(
        out_ref.reshape(-1, head_dim).float(),
        out_dq.reshape(-1, head_dim).float(),
        dim=-1,
    )
    mean_cos = float(cos.mean().item())
    print(f"\n  dequant SDPA mean cos-sim vs bf16 SDPA: {mean_cos:.4f}")
    assert mean_cos > 0.95, (
        f"dequantize+SDPA cos-sim vs bf16 SDPA is {mean_cos:.4f} — dequant path "
        "doesn't preserve attention output well enough for decode."
    )
```

- [ ] **Step 5.2: Run**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_dequant_decode.py::test_dequantize_then_sdpa_matches_bf16_attention_closely -v -s 2>&1 | tail -10
```

Expected: pass with cos-sim > 0.95 (typical 0.97-0.99). If it fails with lower cos-sim, the tq35 dequant error is bigger than investigation suggested at the attention level — possibly the cascading attention amplification we identified. In that case, crank ring_capacity to seq_len in later tasks (so the ring overrides everything), effectively making TurboQuant's compression moot but quality-correct.

---

## Task 6: Gather-packed-blocks helper

**Files:**
- Modify: `vllm/v1/attention/backends/triton_attn.py`

We need to walk the paged block table for one sequence and collect its packed-byte K (and V) cache entries into a contiguous tensor. This is a standard indexing pattern but localised for clarity.

- [ ] **Step 6.1: Read the cache layout**

```bash
grep -n "stride_k_cache_0\|stride_k_cache_1\|stride_k_cache_2\|stride_k_cache_3\|block_size\|num_blocks" vllm/v1/attention/ops/triton_turboquant_decode.py | head -15
```

Confirms the cache layout is `(num_blocks, block_size, num_kv_heads, packed_dim)` — `stride_k_cache_3` is the innermost dimension, bytes.

- [ ] **Step 6.2: Add helper near the top of the file (after imports)**

```python
def _gather_packed_kv_for_seq(
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Return a contiguous (seq_len, num_kv_heads, packed_dim) slice.

    kv_cache: (num_blocks, block_size, num_kv_heads, packed_dim) uint8
    block_table_row: (max_blocks_per_seq,) int32
    """
    block_size = kv_cache.shape[1]
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    block_ids = block_table_row[:num_blocks_needed].to(torch.int64)
    blocks = kv_cache[block_ids]  # (num_blocks_needed, block_size, num_kv_heads, packed_dim)
    blocks = blocks.reshape(
        num_blocks_needed * block_size, kv_cache.shape[2], kv_cache.shape[3]
    )
    return blocks[:seq_len].contiguous()
```

- [ ] **Step 6.3: Verify no syntax errors**

```bash
/root/vllm-venv/bin/python -c "from vllm.v1.attention.backends.triton_attn import _gather_packed_kv_for_seq; print('ok')"
```

Expected: `ok`. If the function isn't visible at import time (e.g., it's defined inside a class), move it to module scope.

---

## Task 7: Add `_forward_turboquant_dequant` method

**Files:**
- Modify: `vllm/v1/attention/backends/triton_attn.py`

- [ ] **Step 7.1: Read `_forward_turboquant`**

```bash
sed -n '1338,1400p' vllm/v1/attention/backends/triton_attn.py
```

Confirm the existing tables (`rotations`, `qjl_matrices`, `centroids`) and masks (`key_masks`, `value_masks`) are already assembled — we reuse them.

- [ ] **Step 7.2: Add imports**

Near the top of the file, alongside the other `turboquant_kv_cache` imports:

```python
from vllm.v1.attention.ops.turboquant_kv_cache import (
    # ... existing imports ...
    dequantize_turboquant_vectors,
)
from vllm.v1.attention.ops.turboquant_recent_ring import (
    RecentRing,
    allocate_recent_ring,
    append_recent,
    gather_recent,
    write_prefill_tail,
)
```

- [ ] **Step 7.3: Add the new decode method**

Inside the attention impl class (same class as `_forward_turboquant`):

```python
    def _forward_turboquant_dequant(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        """Decode path that dequantizes the packed cache to bf16 and runs
        standard SDPA. Bypasses the fused turboquant decode kernel. See
        docs/superpowers/specs/2026-04-24-turboquant-dequantize-decode-design.md
        """
        assert self.turboquant_bits is not None
        self._validate_turboquant_device(query.device)

        rotations, qjl_matrices, centroids, _, _ = self._get_turboquant_tables(
            query.device
        )
        key_masks, value_masks = self._ensure_turboquant_masks(query.device)
        recent_ring: RecentRing | None = getattr(self, "_recent_ring", None)

        query_start_loc_cpu = attn_metadata.query_start_loc_cpu
        seq_lens_cpu = attn_metadata.seq_lens_cpu
        num_seqs = seq_lens_cpu.shape[0]

        import torch.nn.functional as F

        for seq_id in range(num_seqs):
            start = int(query_start_loc_cpu[seq_id].item())
            end = int(query_start_loc_cpu[seq_id + 1].item())
            if end == start:
                continue
            seq_len = int(seq_lens_cpu[seq_id].item())

            block_table_row = attn_metadata.block_table[seq_id]

            packed_k = _gather_packed_kv_for_seq(
                key_cache, block_table_row, seq_len
            )
            packed_v = _gather_packed_kv_for_seq(
                value_cache, block_table_row, seq_len
            )

            dequant_k = dequantize_turboquant_vectors(
                packed_k, self.kv_cache_dtype, self.head_size,
                rotations, qjl_matrices, centroids, key_masks,
                torch.bfloat16,
            )
            dequant_v = dequantize_turboquant_vectors(
                packed_v, self.kv_cache_dtype, self.head_size,
                rotations, qjl_matrices, centroids, value_masks,
                torch.bfloat16,
            )

            if recent_ring is not None:
                recent_k, recent_v, n_recent = gather_recent(
                    recent_ring, seq_id=seq_id
                )
                if n_recent > 0:
                    dequant_k[-n_recent:] = recent_k
                    dequant_v[-n_recent:] = recent_v

            q_3d = query[start:end].view(
                end - start, self.num_heads, self.head_size
            )

            # SDPA expects (batch, heads, tokens, dim).
            kv_repeat = self.num_heads // self.num_kv_heads
            k_rep = dequant_k.repeat_interleave(kv_repeat, dim=1) \
                if kv_repeat > 1 else dequant_k
            v_rep = dequant_v.repeat_interleave(kv_repeat, dim=1) \
                if kv_repeat > 1 else dequant_v

            q_t = q_3d.transpose(0, 1).unsqueeze(0)
            k_t = k_rep.transpose(0, 1).unsqueeze(0)
            v_t = v_rep.transpose(0, 1).unsqueeze(0)

            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                is_causal=(end - start) > 1,
                scale=self.scale,
            )
            output[start:end] = out.squeeze(0).transpose(0, 1)

        return output
```

- [ ] **Step 7.4: Verify**

```bash
/root/vllm-venv/bin/python -c "from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl; print('ok')"
```

Expected: `ok`.

---

## Task 8: Dispatch + ring buffer lifecycle

**Files:**
- Modify: `vllm/v1/attention/backends/triton_attn.py`

- [ ] **Step 8.1: Thread config into the attention impl init**

In the attention impl `__init__`, after the existing turboquant init (where `self.turboquant_bits` etc. are set), add:

```python
        self._recent_ring_capacity = 0
        self._recent_ring: RecentRing | None = None
        if is_turboquant_kv_cache(self.kv_cache_dtype):
            cache_config = vllm_config.cache_config  # match how existing tq init reads config
            self._recent_ring_capacity = int(
                getattr(cache_config, "turboquant_recent_ring_capacity", 0)
            )
            if self._recent_ring_capacity > 0:
                logger.info_once(
                    "TurboQuant recent-ring enabled: capacity=%d",
                    self._recent_ring_capacity,
                    scope="local",
                )
```

(Use the same pattern the existing turboquant init uses to fetch `cache_config`.)

- [ ] **Step 8.2: Lazy ring allocation helper**

Add a method:

```python
    def _ensure_recent_ring(
        self,
        *,
        device: torch.device,
        num_seqs: int,
    ) -> None:
        if self._recent_ring_capacity <= 0:
            return
        if self._recent_ring is not None and self._recent_ring.num_seqs >= num_seqs:
            return
        ring = allocate_recent_ring(
            num_seqs=num_seqs,
            capacity=self._recent_ring_capacity,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            device=device,
        )
        if self._recent_ring is not None:
            # Preserve state for already-tracked sequences.
            old = self._recent_ring
            n = old.num_seqs
            for attr in ("keys", "values", "write_ptr", "fill_count", "total_appends"):
                getattr(ring, attr)[:n].copy_(getattr(old, attr))
        self._recent_ring = ring
```

- [ ] **Step 8.3: Hook ring writes into `do_kv_cache_update`**

Inside `do_kv_cache_update`, after the existing tq35 `turboquant_write_packed_kv` calls and before `return`, add:

```python
            if self._recent_ring_capacity > 0:
                self._ensure_recent_ring(
                    device=key.device,
                    num_seqs=attn_metadata.seq_lens.shape[0],
                )
                query_start_loc_cpu = attn_metadata.query_start_loc_cpu
                for seq_id in range(query_start_loc_cpu.shape[0] - 1):
                    s = int(query_start_loc_cpu[seq_id].item())
                    e = int(query_start_loc_cpu[seq_id + 1].item())
                    n = e - s
                    if n == 0:
                        continue
                    if int(self._recent_ring.total_appends[seq_id].item()) == 0:
                        write_prefill_tail(
                            self._recent_ring,
                            seq_id=seq_id,
                            keys=key[s:e],
                            values=value[s:e],
                        )
                    else:
                        for i in range(n):
                            append_recent(
                                self._recent_ring,
                                seq_id=seq_id,
                                key=key[s + i],
                                value=value[s + i],
                            )
```

Note: `do_kv_cache_update` may not currently receive `attn_metadata` as an argument. Search and thread it through if so:

```bash
grep -n "do_kv_cache_update" vllm/v1/attention/backends/triton_attn.py vllm/v1/worker/*.py
```

- [ ] **Step 8.4: Flip the decode dispatch**

In `_forward_turboquant`, locate the block that calls `turboquant_decode_attention_fwd` non-cascade (around line 1520). Replace that entire invocation (and any cascade-specific branch) with a call to the new method:

```python
        # NEW dispatch: dequantize-then-attend decode.
        return self._forward_turboquant_dequant(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            attn_metadata=attn_metadata,
        )
```

The dense-prefill fast path above this stays unchanged.

- [ ] **Step 8.5: Verify the existing suite still imports**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant.py --collect-only -q 2>&1 | tail -5
```

Expected: collection passes (no import errors).

---

## Task 9: Integration test — long-prompt coherence

**Files:**
- Modify: `tests/quantization/test_turboquant_dequant_decode.py`

The smoking-gun test from the investigation. Under greedy decoding, the current fused-kernel decode collapses to newlines on long prompts. With dequant decode + ring buffer, this should produce coherent output.

- [ ] **Step 9.1: Add the test**

Append:

```python
def test_long_prompts_coherent_with_dequant_decode():
    """Runs the 8 glaive-templated prompts (200-500 tokens) that currently
    collapse to '\\n'*48 under greedy decoding with the fused kernel path.
    With the new dequant path, output must be structurally reasonable."""
    import json, statistics

    import torch
    from vllm import LLM, SamplingParams

    SNAP = (
        "/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    )
    META = "/tmp/qwen2_5_turboquant35.json"

    with open("/tmp/small_glaive_qwen25.jsonl") as f:
        prompts = [json.loads(l)["prompt"] for l in f][:8]
    sp = SamplingParams(temperature=0.0, max_tokens=48, seed=0, ignore_eos=True)

    llm = LLM(
        model=SNAP, dtype="bfloat16", max_model_len=2048,
        max_num_seqs=4, gpu_memory_utilization=0.75,
        attention_backend="TRITON_ATTN",
        kv_cache_dtype="turboquant35",
        enable_turboquant=True,
        turboquant_metadata_path=META,
        turboquant_recent_ring_capacity=64,
    )
    texts = [o.outputs[0].text for o in llm.generate(prompts, sp)]
    del llm
    torch.cuda.empty_cache()

    def uniq_ratio(t): return len(set(t.split())) / len(t.split()) if t.split() else 0.0
    def newline_frac(t): return t.count("\n") / max(len(t), 1)

    mean_u = statistics.mean(uniq_ratio(t) for t in texts)
    mean_nl = statistics.mean(newline_frac(t) for t in texts)

    print(f"\n  dequant-decode long-prompt mean_uniq={mean_u:.2f} mean_nl={mean_nl:.2f}")
    for i, t in enumerate(texts):
        print(f"  [{i}] {t[:120]!r}")

    # Pre-fix baseline (fused kernel): mean_uniq=0.45, mean_nl=0.37.
    # Target with dequant decode: mean_uniq > 0.5, mean_nl < 0.1.
    assert mean_u > 0.5, f"unique-token ratio {mean_u} still degenerate"
    assert mean_nl < 0.1, f"newline fraction {mean_nl} still degenerate"
```

- [ ] **Step 9.2: Run**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant_dequant_decode.py::test_long_prompts_coherent_with_dequant_decode -v -s 2>&1 | tail -25
```

Expected: pass. If it fails (output still degenerate), likely causes in order of likelihood:

1. `do_kv_cache_update` isn't receiving `attn_metadata` — the ring never gets populated and the dequant history is still slightly off. Thread the arg through.
2. `_gather_packed_kv_for_seq` is grabbing the wrong slice. Verify by printing the shape and a few bytes.
3. `dequantize_turboquant_vectors` call is missing a parameter. Compare signature to how the calibration script uses it.
4. Ring state is getting reset between decode steps (check that `self._recent_ring` is shared across calls to the same layer).

Debug by setting `turboquant_recent_ring_capacity=9999` so the ring captures everything — then the dequant path is effectively never used and the test tells you if the ring-buffer write path is correct.

---

## Task 10: Regression + real-model smoke test

- [ ] **Step 10.1: Existing tq suite unchanged**

```bash
/root/vllm-venv/bin/pytest tests/quantization/test_turboquant.py -q 2>&1 | tail -5
```

Expected: 59 passed, 2 pre-existing failures. No new failures.

- [ ] **Step 10.2: End-to-end serving smoke**

```bash
SNAP=/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
META=/tmp/qwen2_5_turboquant35.json \
PROMPTS=/tmp/small_glaive_qwen25.jsonl \
RESULT_DIR=$PWD/benchmarks/results_qwen2_5_dequant \
MAX_MODEL_LEN=2048 MAX_NUM_SEQS=4 MAX_CONCURRENCY=4 NUM_PROMPTS=64 OUTPUT_LEN=128 \
GPU_MEMORY_UTILIZATION=0.80 \
bash benchmarks/benchmark_turboquant_vs_baseline_qwen3_5.sh
```

Edit `benchmarks/benchmark_turboquant_vs_baseline_qwen3_5.sh` turboquant35 arm's `vllm serve` invocation to also pass `--turboquant-recent-ring-capacity 64` (the default is 64 already if you accepted it in Task 1, so this is only needed if you want to override).

Expected: the treatment arm's `comparison.md` now shows meaningfully higher Jaccard vs the pre-fix pure-tq35 bench. Pre-fix Qwen2.5-0.5B sanity run showed Jaccard 3.7%; post-fix, expect >40% even at T=0.7 (and higher at T=0 if you rerun `tq_diag2.py`). The speed will likely be slower than bf16 — expected trade-off, documented in the spec.

---

## Self-Review

**Spec coverage:**

| Spec section | Covered by |
|---|---|
| Ring buffer config flag, CLI | Tasks 1, 2 |
| RecentRing module + allocator + append/gather | Task 3 |
| Ring buffer unit tests | Task 4 |
| Dequantize + SDPA reproduces bf16 attention | Task 5 |
| Gather packed bytes from paged cache | Task 6 |
| New `_forward_turboquant_dequant` method | Task 7 |
| Wire ring writes + flip decode dispatch | Task 8 |
| Long-prompt coherence end-to-end | Task 9 |
| No regression of existing tq suite | Task 10 Step 10.1 |
| Smoke on real model | Task 10 Step 10.2 |

**Placeholder scan:** Task 8 Step 8.1 says "match how existing tq init reads config" — concrete (read the existing turboquant init in the same file and mirror the pattern). Task 9 Step 9.2 has diagnostic guidance for failure modes rather than guaranteed code. Both are acceptable because they describe specific, verifiable actions.

**Type consistency:** `RecentRing` fields used consistently. `dequantize_turboquant_vectors` signature matches how the calibration script already uses it.

**Scope:** single plan, 10 tasks, mostly additive (new method + new module), reversible by changing one line of the dispatch back.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-04-24-turboquant-dequantize-decode.md`. Supersedes the earlier head+tail-window plan.

Both the earlier spec/plan and this one remain on disk — the earlier ones document the sliding-window alternative in case the dequant path turns out to be too slow for your deployment and a more complex but fused approach becomes necessary.

**Two execution options per the writing-plans skill:**

1. **Subagent-Driven (recommended).** Fresh subagent per task + two-stage review.
2. **Inline Execution.** Execute tasks in this session using executing-plans, with a checkpoint after Task 5 (dequant+SDPA proves correct on synthetic data) and after Task 8 (integration dispatch done).

Task 7 and 8 together are the highest-risk pair — the integration into `_forward_turboquant` and `do_kv_cache_update`. Extra attention there regardless of execution mode.
