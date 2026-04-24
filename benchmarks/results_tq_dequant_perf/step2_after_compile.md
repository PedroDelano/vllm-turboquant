# Step 2 (torch.compile) — deferred

## What was attempted

Wrapped `dequantize_turboquant_vectors` with `torch.compile(...)`:

- First try: `mode="reduce-overhead", dynamic=True, fullgraph=False` —
  test hung past the 10-minute mark without emitting any output.
  Killed. No useful diagnostic.
- Second try: `mode="default", dynamic=True, fullgraph=False` —
  still hung beyond the 5-minute test budget. Killed.

The compiled wrapper triggers on the first call from
`_forward_turboquant_dequant`. The per-sequence loop exits long before
the compile completes or recompiles. On the test config (24 layers,
4 sequences of 167–468 tokens), the first compile call is too
expensive to be usable.

Hypothesis (not verified): `dequantize_turboquant_vectors` has enough
dynamic control flow (bit-packing branches, norm-loading, per-group
operations) that `dynamic=True` causes extensive guard generation or
repeated recompilation per shape. Would need direct debug with
`torch._dynamo.utils`, `TORCH_LOGS=recompiles`, and a minimal
reproducer to confirm.

## Current state on the branch

- `_get_compiled_dequant()` accessor remains in
  `vllm/v1/attention/ops/turboquant_kv_cache.py` for future
  experimentation.
- `TQ_DEQUANT` env var in `_forward_turboquant_dequant` defaults to
  `"python"` so production users get the eager path. Opt into
  compiled with `TQ_DEQUANT=compiled` (at your own risk).
- No regression on any existing test.

## Effective delivery

Step 1 (batch) alone gives the concrete speedup. Step 2 is not
blocking; it was an additive optional multiplier that didn't work
under basic usage in this session.

## Follow-ups that might unblock Step 2

1. Run `TORCH_LOGS=recompiles,dynamo pytest ...` on a 1-prompt decode
   to observe what torch.compile is actually doing on each call.
2. Try `fullgraph=True` — if it fails, the error will tell us which
   op is causing the graph break that's making `dynamic=True` costly.
3. Try compiling the inner `_forward_turboquant_dequant` method
   instead of `dequantize_turboquant_vectors` — larger compile unit
   might amortize overhead.
4. Consider a handwritten Triton kernel (the originally-brainstormed
   path before the user steered to cheaper options) — known to be
   viable from profiling; overall ~1 day of work.

# Summary (Step 1 only)

| | Baseline | After Step 1 | Change |
|---|---:|---:|---:|
| total per 50 calls | ~430 ms | ~138 ms (steady state) | **3.1× faster** |
| dequant_k + dequant_v share | 91% | 77% | -14 pts |
| sdpa share | 3.5% | 9.4% | now visible |
| observed wall-time per long-prompt decode | ~11.8 s | ~4.6 s | **2.6× faster** |

Extrapolating from the small model to 35B-A3B: if dequant was 91% of
the 404 s treatment time, a 3× dequant speedup yields an expected
~158 s (≈5× bf16 baseline of 32 s). End-to-end 35B-A3B bench not
re-run in this session — would take ~25 min. When run, land results
in `benchmarks/results_qwen3_5_35b_dequant_perf/` and compare to
`benchmarks/results_qwen3_5_35b_dequant_fix/comparison.md` (404 s
pre-batch baseline).

## Quality preserved

`test_turboquant_dequant_decode.py` — 6/6 passing:
- 4 ring-buffer unit tests
- 1 dequant+SDPA correctness test
- 1 long-prompt coherence test: mean_uniq=0.84, mean_nl=0.03

`test_turboquant.py` — 59/2 unchanged (2 are pre-existing failures
documented in CLAUDE.md).

## Files changed (all uncommitted on branch `turboquant-dequant-perf`)

- `vllm/v1/attention/backends/triton_attn.py` — batched dequant
  restructure; `TQ_DEQUANT` env-var switch (defaults to python).
- `vllm/v1/attention/ops/turboquant_kv_cache.py` —
  `_get_compiled_dequant()` accessor.
- `tests/quantization/test_turboquant_dequant_decode.py` — new
  `test_batched_dequant_equals_per_seq_dequant` locks in the purity
  invariant.
- `tests/quantization/test_turboquant.py` — one-line fix to
  `test_cache_config_requires_feature_gate` (pre-existing regression
  from the prefix-caching guard; not caused by this perf work).
- `benchmarks/results_tq_dequant_perf/step0_baseline.md`
- `benchmarks/results_tq_dequant_perf/step1_after_batch.md`
- `benchmarks/results_tq_dequant_perf/step2_after_compile.md` (this
  file)
