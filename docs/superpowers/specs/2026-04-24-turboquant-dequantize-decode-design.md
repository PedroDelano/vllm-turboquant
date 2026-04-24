# TurboQuant Dequantize-and-Attend Decode — Design

**Status:** approved design, pending implementation plan.
**Date:** 2026-04-24 (revises the earlier head+tail sliding-window design
from the same date; this is the version informed by the 0xSero/turboquant
reference repository).
**AI assistance:** design arrived at by comparing this fork's
implementation against 0xSero/turboquant, which successfully runs on
Qwen3.5 models. Submitting human owns the decisions.

## Supersedes

- `docs/superpowers/specs/2026-04-24-turboquant-hybrid-window-design.md`
  (head+tail sliding window with LSE-merge). The earlier design is still
  valid in its diagnosis of the problem, but the fix is more complex than
  needed. The reference repository demonstrates a simpler pattern that
  also works; we adopt that.

## Problem

TurboQuant35 decode on this fork produces degenerate output (newline
spam, repetition loops) on prompts longer than ~150 tokens across every
Qwen model tested. See `docs/investigations/qwen3_5_turboquant_failure.md`
for the full evidence. The fork's decode path calls
`turboquant_decode_attention_fwd` — a fused Triton kernel that computes
attention scores directly on packed bytes, never materialising the
dequantized K/V vectors. That fused math accumulates enough error over a
long sequence to make the softmax noise-dominated.

## What the reference teaches us

0xSero/turboquant proves the TurboQuant *method* works on Qwen3.5. Its
production decode path (`turboquant/score.py`):

```python
k_hist = quantizer.dequantize(flat.prod_q)         # unpack and reconstruct fp32
v_hist = dequantize_values(flat.value_q, 32)
k_recent = recent_k                                # uncompressed ring buffer
v_recent = recent_v
k_all = torch.cat([k_hist.float(), k_recent.float()], dim=1)
v_all = torch.cat([v_hist.float(), v_recent.float()], dim=1)
# ... standard einsum attention with softmax ...
```

It has the same fused Triton kernels this fork does, but the production
integration bypasses them. The authors trust the slower dequant-then-
attend path more than their own fused kernel. So do we, now.

## Goal

Get TurboQuant35 producing coherent output on long prompts on this fork
by replacing the decode path's fused-kernel call with a
dequantize-then-dense-attention path that reuses this fork's existing
`dequantize_turboquant_vectors` helper (which our reconstruction test
already verified produces cos-sim=0.99 output).

## Non-goals

- Fixing the fused Triton kernel. Leaving it in place as a future
  optimization target; new decode path doesn't call it.
- Replicating the reference's architecture verbatim
  (`CompressedKVStore` + monkey-patching). We keep this fork's existing
  packed-byte cache and just swap the decode math.
- Supporting the MLA / GDN layer variants. Those aren't affected by
  TurboQuant on this fork anyway.
- Rewriting calibration. Existing calibration output is still valid.

## Design

### Core change: dequantize-then-attend decode

In `_forward_turboquant` in `vllm/v1/attention/backends/triton_attn.py`,
replace the call to `turboquant_decode_attention_fwd` with:

1. For each sequence in the batch, gather the relevant packed-byte K/V
   blocks from the paged cache into a contiguous tensor.
2. Call the existing `dequantize_turboquant_vectors` on those blocks to
   get bf16 K and V.
3. Assemble into per-sequence `(seq_len, num_kv_heads, head_dim)`
   tensors (optionally concatenated with the recent-uncompressed ring
   buffer if used — see below).
4. Run standard attention. Options in priority order:
   - `F.scaled_dot_product_attention` (simplest, correct).
   - `context_attention_fwd` from
     `vllm/v1/attention/ops/triton_prefill_attention.py` (the same
     kernel prefill uses — already known-good on this fork).
5. Write the result to `output`.

No LSE merge needed — attention runs over a single, fully-materialised
K/V sequence per request, matching how the reference does it.

### Optional: recent-token ring buffer

To reduce reconstruction noise on the tokens that carry most attention
weight (the last ~32-128 positions), we optionally keep a small ring
buffer of **uncompressed** bf16 K/V per layer per sequence. These tokens
are still written to the packed tq35 cache (so fallback paths work), but
at decode time the ring buffer is the authoritative source. The
dequantized history is only used for tokens older than the ring
capacity.

This matches the reference's `ring_capacity=128` pattern. A size of
32-64 is likely enough on this fork since our dequant round-trip is
already cos-sim=0.99 — the ring buffer is belt-and-braces, not load-
bearing.

V1 default: `ring_capacity=64`. Configurable via a new
`--turboquant-recent-ring-capacity` flag on `vllm serve`. `0` disables
the ring buffer and falls back to pure dequant-history attention
(probably still works based on the reference's own evidence).

### Compression math

Unchanged from the existing tq35 packed-byte cache. The ring buffer adds
a small overhead: `ring_capacity × num_layers × 2 × num_kv_heads ×
head_dim × 2` bytes per sequence. For Qwen3.5-35B-A3B (10 attention
layers, 2 KV heads, head_dim=256) at ring_capacity=64:

```
64 × 10 × 2 × 2 × 256 × 2 = 1.3 MB per sequence
```

Negligible. Headline compression of tq35 is preserved.

## Architecture

### Components affected

- `vllm/v1/attention/backends/triton_attn.py`
  - Add `_forward_turboquant_dequant` method (the new decode path).
  - Modify `_forward_turboquant` to dispatch to the new method for
    decode (but keep the dense-prefill fast path unchanged).
  - Optional: extend `do_kv_cache_update` to also write to the ring
    buffer if enabled.
  - Optional: thread ring_capacity into the attention impl init from
    `CacheConfig`.

### Components NOT changed

- `vllm/v1/attention/ops/turboquant_kv_cache.py` — packing/unpacking
  math unchanged.
- `vllm/v1/attention/ops/triton_turboquant_kv_update.py` — write kernel
  unchanged (tq35 cache still populated same as today).
- `vllm/v1/attention/ops/triton_turboquant_decode.py` — the broken
  fused decode kernel. Leave the file alone; just stop calling it from
  the production decode path.
- `benchmarks/generate_turboquant_metadata.py` — calibration unchanged.
- `calibration/*.json` — existing metadata still works.

### New module (small)

- `vllm/v1/attention/ops/turboquant_recent_ring.py` — a tiny per-layer
  per-sequence ring buffer for uncompressed recent K/V. Optional;
  v1 default is enabled.
  - `RecentRing` dataclass: fixed-capacity bf16 storage, ring pointer.
  - `append_recent(ring, key, value, seq_id)` — decode-time append.
  - `write_prefill_tail(ring, keys, values, seq_id, tail_len)` —
    prefill-time initial fill with last-N tokens.
  - `gather_recent(ring, seq_id) -> (keys, values)` — decode-time read
    in insertion order.

### New code in `_forward_turboquant_dequant`

Pseudocode (~80 lines):

```python
def _forward_turboquant_dequant(self, query, key, value, key_cache,
                                value_cache, output, attn_metadata):
    recent_ring = getattr(self, "_recent_ring", None)
    for seq_id in range(attn_metadata.num_seqs_batch):
        start, end = query_start_loc_cpu[seq_id], query_start_loc_cpu[seq_id + 1]
        seq_len = int(seq_lens_cpu[seq_id])
        q_seq = query[start:end]

        # Gather packed bytes from paged cache for all prior tokens.
        block_ids = attn_metadata.block_table[seq_id, : ceil(seq_len / block_size)]
        packed_k = _gather_packed_blocks(key_cache, block_ids, seq_len)
        packed_v = _gather_packed_blocks(value_cache, block_ids, seq_len)

        # Dequantize via existing helper.
        dequant_k = dequantize_turboquant_vectors(
            packed_k, self.kv_cache_dtype, self.head_size,
            rotations, qjl_matrices, centroids, group_indices,
            torch.bfloat16,
        )
        dequant_v = dequantize_turboquant_vectors(packed_v, ...)

        # Optionally override last ring_capacity tokens with uncompressed
        # recent K/V.
        if recent_ring is not None:
            recent_k, recent_v = gather_recent(recent_ring, seq_id)
            n_recent = recent_k.shape[0]
            dequant_k[-n_recent:] = recent_k
            dequant_v[-n_recent:] = recent_v

        # Standard attention.
        q_3d = q_seq.view(end - start, self.num_heads, self.head_size)
        out = F.scaled_dot_product_attention(
            q_3d.transpose(0, 1).unsqueeze(0),
            dequant_k.transpose(0, 1).unsqueeze(0),
            dequant_v.transpose(0, 1).unsqueeze(0),
            is_causal=True, scale=self.scale,
        )
        output[start:end] = out.squeeze(0).transpose(0, 1)
    return output
```

`_gather_packed_blocks` is a small utility that walks the block_table
and pulls the right bytes out of the paged cache. Standard vLLM
indexing pattern.

### Dispatch

In `_forward_turboquant`:

```python
def _forward_turboquant(self, ...):
    if self._can_use_turboquant_dense_prefill(query, attn_metadata):
        # Existing dense prefill — unchanged.
        return context_attention_fwd(...)

    if key_cache.numel() == 0 or value_cache.numel() == 0:
        # Existing fallback — unchanged.
        return self._fallback_turboquant_attention(...)

    # NEW: dequantize-then-attend. Was: turboquant_decode_attention_fwd(...).
    return self._forward_turboquant_dequant(
        query, key, value, key_cache, value_cache, output, attn_metadata,
    )
```

## Data flow

### Prefill
1. Existing `context_attention_fwd` dense path — unchanged.
2. Existing `turboquant_write_packed_kv` writes all tokens to the tq35
   cache — unchanged.
3. If ring buffer enabled: additionally write the last `ring_capacity`
   tokens of each sequence's prefill output into the ring.

### Decode (one new token per sequence)
1. Write new token to packed cache via existing `turboquant_write_packed_kv`.
2. If ring buffer enabled: append the new token to the ring (overwriting
   oldest slot if ring full).
3. `_forward_turboquant_dequant`:
   a. Gather packed bytes from cache for all prior tokens.
   b. Dequantize to bf16.
   c. If ring enabled, overwrite last `n_recent` positions of K/V with
      the uncompressed values from the ring.
   d. Standard attention via SDPA.
4. Return output.

## Error handling

- `dequantize_turboquant_vectors` failing (e.g., bad metadata) → raise,
  same as today.
- Ring buffer allocation failing → log a warning, disable ring buffer,
  continue with pure dequant path (still correct, just slightly lossier
  on recent tokens).
- Paged cache empty for a sequence (very first decode step) → handled
  by the existing `kv_cache.numel() == 0` fallback.
- Mismatch between cached seq_len and attn_metadata seq_len → assert and
  fail; same as today.

## Testing

- **Unit test: dequantize round-trip still matches cos-sim > 0.98** on
  real Qwen K/V. Reuse `docs/investigations/scripts/tq_reconstruct.py`
  as-is; it already proves the dequant helper is sound.
- **Unit test: SDPA on dequantized K/V matches SDPA on original bf16
  K/V within tolerance** (cos-sim > 0.98 on the attention output). New
  test.
- **Unit test: ring buffer eviction**. Write ring_capacity + N tokens,
  assert only the last ring_capacity are present and ordering is
  correct. New test.
- **Integration test: long-prompt coherence.** Re-run
  `docs/investigations/scripts/tq_diag2.py` with the new decode path.
  Assert mean_uniq > 0.5, mean_nl < 0.1 on the 8 glaive prompts that
  currently collapse to newlines. Target: mean Jaccard > 0.4 vs bf16
  baseline at T=0.
- **No regression.** `pytest tests/quantization/test_turboquant.py` still
  shows 59 passed / 2 pre-existing failures.

## Scope cuts (YAGNI)

- Not porting the reference's `CompressedKVStore` — this fork's packed
  cache is structurally fine, just the decode math was wrong.
- Not attempting to fix the fused Triton decode kernel. If someone
  wants to fix it later, the unit tests and the dequant reference
  implementation provide a ground truth to test against.
- Not introducing modes (off/capture/hybrid/full_tq). The fork's behavior
  is binary: TurboQuant is enabled or not. If enabled, use the new
  dequant path. If disabled, vLLM's normal bf16 path runs as today.
- Not changing calibration, metadata format, or the write path.
- Not head-window (first N tokens uncompressed). Reference skips this
  too — dequant is accurate enough on history.
- Not per-layer ring capacity (one config for all attention layers).

## Risks I'm flagging

1. **Decode throughput drop.** Dequantizing the full history on every
   decode step is O(context × attention_layers × head_dim) of fp32
   work per step. On a 16K context with Qwen3.5's 10 attention layers,
   that's ~160 MB of fp32 K/V reconstructed per step per sequence. At
   high concurrency this eats memory bandwidth. The reference sees
   this in its numbers ("~1.45× context extension on MoE models, vs
   2× on dense") — compression is there, but speed suffers. Acceptable
   for V1; fused-kernel decode is a future optimisation target once we
   have a coherent baseline to compare against.
2. **SDPA in the loop.** `F.scaled_dot_product_attention` with
   `is_causal=True` needs the Q tokens to be consecutive in position.
   For decode, Q length is 1 so this is trivially fine. For chunked
   prefill, the Q chunk may have positions offset into the sequence;
   SDPA still handles this correctly because the causal mask is within
   the Q chunk plus the preceding K. If a specific chunked-prefill
   configuration trips this up, fall back to `context_attention_fwd`
   which explicitly takes per-batch offsets.
3. **Memory during dequant.** For very long contexts, materialising all
   K, V as bf16 tensors per forward pass is expensive in memory too.
   At 32K context × 10 layers × 2 KV heads × 256 head_dim × 2 bytes, we
   need ~327 MB of transient tensors per layer per step. Workable on
   H100. For even longer contexts (128K+), a chunked-dequant pass would
   be needed — out of scope for V1.

## Open follow-ups

- Fix the fused Triton decode kernel for real. If the dequant path
  works and `turboquant_decode_attention_fwd` can be shown to differ
  from the dequant-then-attend output on real inputs, we have a
  concrete regression test for the kernel bug.
- Study whether calibration even matters on the dequant path. Our
  earlier experiment showed identity-masked metadata produces equally
  broken output via the fused kernel. The dequant path might behave
  the same way regardless of outlier selection, which would suggest
  calibration is currently doing nothing useful for end-to-end quality.
