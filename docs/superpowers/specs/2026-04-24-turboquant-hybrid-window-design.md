# TurboQuant Head+Tail Sliding-Window Hybrid — Design

**Status:** approved design, pending implementation plan.
**Date:** 2026-04-24.
**AI assistance:** brainstormed with Claude (Opus 4.7); the submitting human
owns the decisions and will review every line at implementation.

## Problem

TurboQuant35 KV-cache quantization on this fork produces degenerate generations
on long prompts (> ~150 tokens) across every Qwen model tested, under greedy
decoding. Root cause, per the `docs/investigations/qwen3_5_turboquant_failure.md`
report: quantization preserves K/V direction (cos-sim 0.99) but loses ~12-17%
magnitude per token; errors accumulate in the attention sum over long sequences;
the softmax distribution becomes noise-dominated; output collapses to newline
spam or repetition loops. Unit tests (59/61 passing) verify only
kernel-vs-reference equality on synthetic inputs — they miss the compounding.

fp8 KV cache (2× compression) works cleanly, but the whole reason to carry
TurboQuant's complexity is the ~3× compression it targets. We want to preserve
that compression on long context without losing generation quality.

## Goal

Make TurboQuant viable for Qwen serving at **~2.7–2.8× compression on 16K
context**, producing coherent outputs under greedy decoding on the same long
prompts that currently fail.

## Non-goals

- Supporting context lengths that drive the middle region past ~64K tokens
  (the same accumulation problem would eventually reappear; we punt).
- Unifying the quantized and dense caches into a single mixed-precision paged
  store. Two parallel caches are simpler and enough for this problem.
- Revisiting the TurboQuant kernel math or codebooks. This spec is purely
  about which TOKENS get quantized, not HOW they are quantized.
- Per-layer window sizes or adaptive window sizing.

## Core insight

Attention weight in causal LMs concentrates on two regions: the **head** (system
prompt and any instruction-shaped content at the very start) and the **tail**
(the current turn and recent decode steps). The **middle** typically contributes
the long-tail of attention probability mass.

If we keep only the head and tail in bf16 and let the middle tolerate the
TurboQuant quantization error, we keep the high-attention tokens clean while
still getting ~3× compression on most of the cache. The middle's quantization
noise matters much less because those tokens rarely dominate the softmax.

This is a routing fix, not a quantization fix. TurboQuant's math doesn't
change; we just stop trusting it for the tokens whose precision we actually
need.

## Configuration (defaults)

- **`head_size` = 1024 tokens** of bf16 (covers most system prompts / tool
  schemas in typical agent workloads).
- **`tail_size` = 512 tokens** of bf16 (recent turns + decode steps).
- **Middle region** = `[head_size, seq_len - tail_size)` in TurboQuant35.
- Both sizes are server-level CLI flags (`--turboquant-head-window-size`,
  `--turboquant-tail-window-size`), overridable at launch.

### Compression math at 16K context

Qwen3.5-35B-A3B (head_dim=256, 10 attention layers, 2 KV heads):

- bf16 per token per layer: 2048 bytes
- tq35 per token per layer: ~608 bytes
- All bf16 @ 16K seq, 1 seq: **327 MB**
- Hybrid @ 16K seq, 1 seq: 1536 × 10 × 2048 + 14464 × 10 × 608 = **118 MB**
- Compression: **2.78×**

At lower concurrency this saves ~200 MB/seq; at 4-way concurrent 16K
sequences, ~830 MB total. KV cache is not the main memory pressure on
hybrid-attention Qwen3.5 (weights dominate), but compression is real and
compounds with max_num_seqs.

## Architecture

Two parallel KV caches per attention layer:

1. **`kv_cache_tq`** (existing) — packed tq35 bytes. Stores ALL tokens,
   unchanged from today.
2. **`kv_cache_bf16_window`** (NEW) — fixed-size bf16 storage, with a head
   region (contiguous array) and a tail region (ring buffer). Stores only
   the head + tail tokens for each sequence.

Both caches are populated on write. Decode reads both and runs two attention
passes, merged via log-sum-exp (LSE), so the combined output mathematically
equals full-sequence attention:

- **Pass 1 — tq35**: existing `turboquant_decode_attention_fwd`, masked to
  the middle positions `[head_size, seq_len - tail_size)`.
- **Pass 2 — bf16**: standard Triton dense attention, over head + tail
  positions.
- **Merge**: `merge_attn_states` (already used by the cascade path in
  `_forward_turboquant` at triton_attn.py:1515).

### Why LSE merge is the right primitive

Softmax attention is distributive over disjoint keyset partitions via
log-sum-exp. If you compute attention over tokens A and tokens B separately
and get outputs `(out_A, lse_A)` and `(out_B, lse_B)`, then:

- `weight_A = exp(lse_A - max(lse_A, lse_B))`
- `weight_B = exp(lse_B - max(lse_A, lse_B))`
- `out = (weight_A * out_A + weight_B * out_B) / (weight_A + weight_B)`

is exactly what full-batch softmax would produce. vLLM already has the utility
for this (`vllm.v1.attention.ops.merge_attn_states.merge_attn_states`).
Reusing it keeps new code surface small.

## Components to build

- **`vllm/v1/attention/ops/hybrid_window_cache.py`** (new). Allocator + write
  path for the bf16 window. Contains:
  - `allocate_bf16_window(num_seqs, head_size, tail_size, num_kv_heads, head_dim, dtype, device)`
  - `write_bf16_window(cache, seq_id, position, key, value, head_size, tail_size, total_seq_len)`
  - `get_active_window_positions(seq_id, seq_len, head_size, tail_size) -> tuple[LongTensor, LongTensor]` → positions currently held in bf16.

- **`vllm/v1/attention/backends/triton_attn.py`**. Extend `_forward_turboquant`:
  - New branch when `is_hybrid_window_enabled()`.
  - Runs tq35 pass with a middle-only mask, runs bf16 dense pass over the
    window, LSE-merges outputs.
  - ~100 lines net.

- **`vllm/config/cache.py`**. Add the two size flags; validate
  `head_size + tail_size <= max_model_len`.

- **Tests**: `tests/quantization/test_turboquant_hybrid_window.py` — see
  Testing below.

No change to `calibration/datasets/`, `benchmarks/generate_turboquant_metadata.py`,
or `scripts/build_prompts.py`. Calibration metadata still applies to the
middle-region tq35 cache exactly as today.

## Data flow

### Prefill

For a sequence of length L at initial prefill:

1. Existing dense `context_attention_fwd` path computes attention output from
   fresh K/V (unchanged; this is what prefill already does).
2. Existing `turboquant_write_packed_kv` writes all L tokens to `kv_cache_tq`
   (unchanged).
3. NEW: `write_bf16_window` also writes:
   - If `L <= head_size`: all L tokens to the head region; tail empty.
   - Else: first `head_size` tokens to the head region; last
     `min(L - head_size, tail_size)` tokens to the tail region.
   - Middle tokens (`head_size ≤ pos < L - tail_size`): only in tq35 cache.

Short prompts (L ≤ head_size + tail_size) fit entirely in bf16. The tq35 cache
still has copies but the decode path will detect this case and skip Pass 1 —
short-prompt quality matches pure bf16 (verified to work in `tq_diag.py`).

### Decode

Each new generated token:

1. Write to `kv_cache_tq` (unchanged).
2. Write to tail ring buffer:
   - If ring full, the slot for the oldest tail token is overwritten. That
     older token is no longer in bf16 — but it is still in tq35, which is
     where Pass 1 will read it.
3. Forward pass:
   - Pass 1 (tq35 decode): mask tokens outside `[head_size, seq_len - tail_size)`.
   - Pass 2 (dense Triton): use the head array + tail ring as the K/V source.
   - Merge via LSE.

### Boundary conditions

- `seq_len <= head_size`: only head region active. Pass 1 output is empty
  (no valid positions); skip it. Attention is pure bf16 — identical to
  baseline.
- `head_size < seq_len <= head_size + tail_size`: head full, tail growing.
  Still no middle region; skip Pass 1.
- `seq_len > head_size + tail_size`: full regime. Both passes + merge.

The dispatch logic to pick which passes to run is the single decision point
that makes the correctness of the design easy to audit.

## Error handling

- Window allocation failure at startup → hard error, clear message. The user
  reduces `head_size` or `tail_size`.
- `head_size + tail_size > max_model_len` → startup validation error.
- `head_size == 0` (`tail_size == 0`) → enabled path collapses to "all-bf16
  on the non-zero window side"; useful for ablation, no crash.
- Tail ring-buffer slot races during multi-sequence batching → the write path
  is per-slot and indexed by `slot_mapping`, so sequences don't share slots.
  Same safety story as today's tq35 write.

## Testing

- **Correctness on short sequences.** Integration test: with
  `seq_len <= head_size + tail_size`, hybrid mode must produce byte-identical
  output to pure-bf16 baseline at T=0. This proves the routing doesn't
  silently corrupt anything in the "simple" regime.
- **LSE-merge unit test.** Build a synthetic attention scenario (known K/V,
  known Q). Run Pass 1 alone, Pass 2 alone, hybrid with LSE merge. Hybrid
  output must match full-sequence attention with known tolerance.
- **Long-prompt greedy quality.** Port `docs/investigations/scripts/tq_diag2.py`
  and `tq_diag6.py` to re-run with hybrid mode; assert outputs are
  structurally reasonable (unique-token ratio > 0.5, newline fraction < 0.1,
  Jaccard vs bf16 baseline > 0.4).
- **Ring-buffer eviction.** Unit test: generate `tail_size + 100` tokens
  sequentially; assert the bf16 tail region contains exactly the last
  `tail_size` tokens in order, and that older generations were evicted
  cleanly (no stale data retrieved by the bf16 pass).
- **Memory accounting.** At `max_model_len=16384` on Qwen3.5-35B-A3B with
  `max_num_seqs=4`, measure `gpu_memory_usage` at allocation time. Compare
  to both pure bf16 and pure tq35 baselines. Expect ~2.7× compression vs
  bf16.
- **No regression.** Existing `pytest tests/quantization/test_turboquant.py`
  still passes (59/61, with the 2 pre-existing failures unchanged).

## Scope cuts (YAGNI)

- No per-layer window sizes.
- No adaptive / per-request window sizes.
- No background re-quantization on eviction (dual-write covers this).
- No extension of the turboquant recipes (turboquant25 lives or dies by the
  same design; no new recipe needed).
- No unified mixed-precision paged cache.
- No re-benchmarking on the A6000 or GB10 platforms (H100 / SM90 only — that
  is where our validation lives).

## Risks I am accepting

1. **Head of 1024 may not cover long system prompts.** Real Qwen tool-calling
   system prompts go up to ~3K tokens. If the tail of the system prompt falls
   into the middle region, attention on it during decode will be quantized.
   Mitigation: expose `head_size` so deployments with large system prompts
   can bump to 2048 or 4096.
2. **Two-pass attention has per-decode overhead.** Expect ITL to be 1.2-1.5×
   vs pure bf16. If a deployment is ITL-bound, this is a tax to weigh.
3. **Doesn't help at context lengths far past max_window coverage.** If future
   use cases push to 64K+, the middle region grows to dominate and quantization
   noise returns. The design accepts this — the answer at that scale is a
   different quantization method, not more window.

## Open follow-ups (not in scope for this spec)

- Auto-detect system-prompt boundary and set head dynamically per request.
- Support 2-group windows (e.g. head + middle-pinned + tail) if some attention
  workload has a third high-weight region we discover.
- Add an integration-level quality regression test in CI so future TurboQuant
  changes can't silently break long-prompt generation again.
