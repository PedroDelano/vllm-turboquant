# TurboQuant Dequant Perf — Batch + Compile Design

**Status:** approved design, pending implementation plan.
**Date:** 2026-04-24.
**Branch:** `turboquant-dequant-perf` (split off from `main` after commit
`1216691`). All work remains uncommitted per user instruction until they OK.

## Context

The dequant-then-attend decode path shipped on `main` (commit `bb33f10`)
restores TurboQuant output coherence on long prompts, but regresses
wall-time to 12.5× bf16 on a Qwen3.5-35B-A3B IID bench (32.4 s → 404.4 s).
Profiling (see `docs/investigations/qwen3_5_turboquant_failure.md`
addendum and the in-code `TQ_PROFILE=1` instrumentation at
`vllm/v1/attention/backends/triton_attn.py::_forward_turboquant_dequant`)
shows `dequantize_turboquant_vectors` is **91% of decode time**, split
roughly 47/44 between the K-side and V-side calls. Everything else —
gather, ring overlay, SDPA — is under 10% combined.

Root causes of the dequant cost:

1. **Per-sequence loop.** `_forward_turboquant_dequant` iterates over
   sequences in the batch, calling `dequantize_turboquant_vectors` once
   per side per sequence per layer. At `num_seqs=4`, 24 layers, that's
   192 dequant calls per decode step. Each is a short call (~1 ms), so
   kernel-launch overhead dominates.
2. **Op-at-a-time dispatch.** `dequantize_turboquant_vectors` itself
   runs ~8 PyTorch ops (bit unpack, centroid gather, structured
   Hadamard, QJL unpack, QJL matmul, norm rescale). Each is a separate
   CUDA launch reading/writing intermediate tensors. Memory traffic is
   high relative to the actual compute.

## Goal

Reduce decode wall-time on the 35B-A3B IID bench from ~404 s toward
bf16's 32 s, with no changes to the math or correctness. Target bands:

- After step 1 (batch): ~5-7× bf16 overhead (~160-220 s).
- After step 2 (+ `torch.compile`): ~2-3× bf16 overhead (~65-100 s).

Decide whether to continue after each step based on measured profile.

## Non-goals

- Fixing the fused Triton decode kernel's numerical bias. Out of scope;
  separate investigation.
- Writing a hand-tuned Triton kernel for dequant. Considered earlier;
  deferred unless these cheaper changes don't suffice.
- Caching dequantized K/V between decode steps. Effectively doubles
  KV-cache memory; defeats the compression story.
- Changing `dequantize_turboquant_vectors`' math, signature, or output
  dtype.

## Architecture

Two independent, sequentially-applied changes:

### Step 1 — batch across sequences

Restructure `_forward_turboquant_dequant` to gather all sequences'
packed K/V into single contiguous tensors, dequantize once per side,
then split per-seq for ring overlay and SDPA.

```python
# Inside _forward_turboquant_dequant, replace the per-seq loop's dequant
# stanza with:

packed_k_list: list[torch.Tensor] = []
packed_v_list: list[torch.Tensor] = []
seq_offsets: list[int] = [0]
seq_lens_list: list[int] = []
seq_ranges: list[tuple[int, int]] = []  # (start, end) in the batched dequant output

for seq_id in range(num_seqs):
    start = int(query_start_loc_cpu[seq_id].item())
    end = int(query_start_loc_cpu[seq_id + 1].item())
    if end == start:
        seq_ranges.append((seq_offsets[-1], seq_offsets[-1]))
        continue
    seq_len = int(seq_lens_cpu[seq_id].item())
    block_table_row = attn_metadata.block_table[seq_id]
    packed_k_list.append(
        _gather_packed_kv_for_seq(key_cache, block_table_row, seq_len)
    )
    packed_v_list.append(
        _gather_packed_kv_for_seq(value_cache, block_table_row, seq_len)
    )
    seq_lens_list.append(seq_len)
    seq_offsets.append(seq_offsets[-1] + seq_len)
    seq_ranges.append((seq_offsets[-2], seq_offsets[-1]))

if not packed_k_list:
    return output  # nothing to process

packed_k_all = torch.cat(packed_k_list, dim=0)
packed_v_all = torch.cat(packed_v_list, dim=0)

dequant_k_all = dequantize_turboquant_vectors(
    packed_k_all, self.kv_cache_dtype, self.head_size,
    rotations, qjl_matrices, centroids, key_masks, torch.bfloat16,
)
dequant_v_all = dequantize_turboquant_vectors(
    packed_v_all, self.kv_cache_dtype, self.head_size,
    rotations, qjl_matrices, centroids, value_masks, torch.bfloat16,
)

# Then the existing per-seq loop continues, but uses pre-computed slices:
for seq_id in range(num_seqs):
    ds, de = seq_ranges[seq_id]
    if de == ds:
        continue
    dequant_k = dequant_k_all[ds:de]
    dequant_v = dequant_v_all[ds:de]
    # ... existing ring overlay + SDPA unchanged ...
```

Correctness invariant: each seq's dequant output must be bit-identical
to what the per-seq loop produced. That's true because
`dequantize_turboquant_vectors` is pure (no cross-token state) and
concatenating inputs before a pure function equals concatenating its
per-element outputs.

### Step 2 — `torch.compile` the dequant function

Wrap `dequantize_turboquant_vectors` with `torch.compile`:

```python
# Add to a module near dequantize_turboquant_vectors (e.g. same file):

_compiled_dequant_turboquant_vectors = None

def _get_compiled_dequant() -> Callable:
    """Return a cached, torch.compile'd wrapper around
    dequantize_turboquant_vectors.

    Compiled lazily so that test code that doesn't exercise the serving
    path doesn't pay compile cost.
    """
    global _compiled_dequant_turboquant_vectors
    if _compiled_dequant_turboquant_vectors is None:
        _compiled_dequant_turboquant_vectors = torch.compile(
            dequantize_turboquant_vectors,
            mode="reduce-overhead",
            dynamic=True,
            fullgraph=False,
        )
    return _compiled_dequant_turboquant_vectors
```

`_forward_turboquant_dequant` calls `_get_compiled_dequant()` instead
of `dequantize_turboquant_vectors` directly.

`dynamic=True` keeps one compiled graph across varying seq_len.
`fullgraph=False` lets graph breaks at non-compile-friendly ops
(e.g. `.item()`) fall through gracefully. `mode="reduce-overhead"`
uses CUDA graphs for captured regions.

Fallback: environment variable `TQ_DEQUANT=python` forces the
non-compiled path, for correctness bisection.

## Components

- **Modify**: `vllm/v1/attention/backends/triton_attn.py`
  - Restructure `_forward_turboquant_dequant`'s per-seq loop per
    Step 1's pseudocode.
  - Import `_get_compiled_dequant` and use it per Step 2.
  - Wire `TQ_DEQUANT=python` fallback.
  - Keep the `TQ_PROFILE=1` instrumentation intact so we can re-measure.

- **Modify**: `vllm/v1/attention/ops/turboquant_kv_cache.py`
  - Add `_get_compiled_dequant` accessor alongside the existing
    `dequantize_turboquant_vectors`.

- **No new files.**

## Testing

1. **Correctness regression.**
   - `tests/quantization/test_turboquant_dequant_decode.py` — all 5
     existing tests still pass.
   - `test_long_prompts_coherent_with_dequant_decode` — mean_uniq >
     0.5, mean_nl < 0.1 (same numeric floor as today).

2. **Unit: batched dequant matches per-seq dequant.** New test that
   takes 3 sequences with different lengths, runs the batched path and
   the per-seq path on the same inputs, asserts per-seq outputs are
   bit-identical. Catches regressions in Step 1's slicing logic.

3. **Perf micro-bench.** Re-enable `TQ_PROFILE=1` and run the
   long-prompt test. Collect before/after numbers for:
   - Total decode phase time.
   - `dequant_k` + `dequant_v` share.
   - SDPA share.
   Record in `benchmarks/results_tq_dequant_perf/phase_breakdown.md`.

4. **End-to-end A/B.** Re-run the 35B-A3B bench
   (`benchmarks/benchmark_turboquant_vs_baseline_qwen3_5.sh` with
   `TURBOQUANT_RING_CAPACITY=1024`). Expect wall-time drop; record in
   a new `benchmarks/results_qwen3_5_35b_dequant_perf/`. Compare to
   the pre-change baseline in `benchmarks/results_qwen3_5_35b_dequant_fix/`.

5. **`torch.compile` sanity.** Log the number of recompilations during
   the long-prompt test. If >1-2 recompilations happen during normal
   operation (after the initial compile), the `dynamic=True` hint
   isn't doing its job; file a follow-up.

## Error handling

- Step 1: if `packed_k_list` is empty (all seqs have 0 query tokens),
  return `output` unchanged — no dequant call. Matches current
  behavior (the per-seq loop simply skips all seqs).
- Step 2: if compilation fails or `torch.compile` raises, fall back to
  the non-compiled `dequantize_turboquant_vectors` and log a warning.
  Same fallback path as the `TQ_DEQUANT=python` env var.
- The ring-overlay indexing (`dequant_k[-n_use:] = recent_k[-n_use:]`)
  was already per-seq; no change.

## Scope cuts (YAGNI)

- No Triton kernel.
- No caching dequantized K/V across decode steps.
- No rewrite of `dequantize_turboquant_vectors` internals.
- No batching of the ring overlay or SDPA path (not on the critical
  path per the profile).
- No `dynamic_shapes` bucketing to prevent `torch.compile` recompiles —
  we try `dynamic=True` first and file a follow-up only if it falls
  over.

## Risks

1. **`torch.compile` + vLLM's existing compilation can conflict.** vLLM
   wraps the model forward in its own `torch.compile`. Calling another
   `torch.compile`'d function from inside could confuse the compiler or
   cause repeated recompilation. If we see this in practice, fall back
   to plain `torch.compile` without vLLM's wrapping (measure inside a
   small standalone script) or defer Step 2.

2. **Batched dequant needs contiguous packed inputs.** `torch.cat` on
   the per-seq packed-byte tensors creates a new allocation each call.
   That allocation + copy is itself a cost; worth measuring. If
   significant, switch to pre-allocated scratch buffers.

3. **Step 1 correctness failure at batch boundaries.** A bug in
   `seq_ranges` slicing could mix one seq's dequantized K into
   another's attention. The "batched dequant == per-seq dequant" test
   in `Testing #2` is explicitly designed to catch this.

## Decision points

- After Step 1 measurement: if wall-time is already acceptable
  (say <3× bf16), stop. Don't do Step 2.
- After Step 2 measurement: if we hit the target, done. If we're
  still at 5× bf16 or worse, reopen the question of writing a Triton
  dequant kernel, or switch to caching strategy.

## Files that will change in this work

- `vllm/v1/attention/backends/triton_attn.py` (modify)
- `vllm/v1/attention/ops/turboquant_kv_cache.py` (modify, small)

## Files that will not change

- `vllm/v1/attention/ops/turboquant_recent_ring.py`
- `vllm/v1/attention/ops/triton_turboquant_decode.py`
- `vllm/v1/attention/ops/triton_turboquant_kv_update.py`
- `vllm/config/cache.py`
- `vllm/engine/arg_utils.py`
- `tests/quantization/test_turboquant_dequant_decode.py` (except
  optionally adding the new batched-equivalence test)
