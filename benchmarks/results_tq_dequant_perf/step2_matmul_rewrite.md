# Step 2 (matmul rewrite) — replacing FWHT butterfly loops with dense matmul

## What changed

`dequantize_turboquant_vectors` at
`vllm/v1/attention/ops/turboquant_kv_cache.py` previously called
`_apply_mse_inverse_transform` / `_apply_qjl_inverse_transform`, which each
run a log2(dim) stack of butterfly passes via `_fwht_pow2` plus per-block
sign multiplies and per-block sqrt normalisation. For the head-dim splits
actually in use (32 + 96 on Qwen 3.5), this costs ~6 separate CUDA
launches per group per side per layer.

The inverse block-Hadamard transform is linear, so it has a dense matrix
form. The rest of the codebase already materialises it via
`get_turboquant_mse_inverse_transform_matrix` / `..._qjl_..._matrix` for
the fused postprocess kernel — proof that the matrix form gives the right
answer. We now use the same matrix inside the Python decode path.

The function signature stays the same (caller still passes sign vectors).
Inside the hot loop we look up the dense `[dim, dim]` inverse matrix via
a new `_inverse_matrix_from_signs(signs, normalized=...)` helper, keyed
by `signs.data_ptr()` (the signs come from a `@cache`'d producer, so the
pointer is stable across requests). One matmul replaces one butterfly
stack.

## Correctness

- `tests/quantization/test_turboquant_dequant_decode.py` → **6/6 passing**
  (includes the long-prompt coherence test and the batched-vs-per-seq
  purity test).
- `tests/quantization/test_turboquant.py` → **59 passed / 2 failed**
  (same 2 `test_turboquant_prefill_reads_quantized_cache` failures that
  are pre-existing on main and documented in `CLAUDE.md`).
- Direct numerical check: `_apply_*_inverse_transform(x, signs)` vs
  `x @ inverse_matrix_from_signs(...)` agree to `|Δ| ≈ 1e-6` on dims
  32/96/128 — float32 rounding only.

## End-to-end wall-time — Qwen3.5-35B-A3B, qwen-agent, 64 prompts, MC=2

Same harness as `results_qwen3_5_35b_dequant_perf` (pre-rewrite step1).

| metric | bf16 KV | step1 (batch) | step2 (matmul) | Δ vs step1 |
|---|---:|---:|---:|---:|
| total wall time | 32.59 s | 348.33 s | **301.21 s** | **-13.5%** |
| req throughput | 1.96 req/s | 0.184 req/s | 0.212 req/s | +15.2% |
| output throughput | 251.38 tok/s | 23.52 tok/s | 27.20 tok/s | +15.6% |
| mean ITL | 6.26 ms | 50.73 ms | **39.13 ms** | **-22.9%** |
| median ITL | 6.18 ms | 50.77 ms | **39.10 ms** | **-23.0%** |
| p99 ITL | 7.30 ms | 53.89 ms | 41.91 ms | -22.2% |
| mean TTFT | 222.87 ms | 4,459.2 ms | 4,458.2 ms | ≈0 |

TurboQuant is still 9.2× bf16 wall-time (down from 10.7×). The remaining
cost is not attention — attention is ~40% of decode; the other 60%
(MoE expert routing, KV update, RoPE, non-attention compute) is
untouched.

## Quality

Token-overlap Jaccard at T=0.7 (harness default):
- step1: 27.1% (exact-match 0%)
- step2: 26.5% (exact-match 0%)

Within measurement noise — the quantisation math is unchanged, only
the kernel layout moved. Long-prompt coherence test at T=0 still
passes (mean_uniq > 0.5, mean_nl < 0.1).

## Profile — Qwen3.5-35B-A3B (same config as step1)

Per-call attention cost on `/tmp/profile_35b.py` at steady state:

| phase | step1 (batch) | step2 (matmul) | Δ |
|---|---:|---:|---:|
| total / call | 3.33 ms | 2.23 ms | **×1.49 faster** |
| dequant_k+v | 2.79 ms | 1.67 ms | **×1.67 faster** |
| dequant share | 83.9% | 74.6% | — |
| sdpa share | 8.5% | 12.7% | — |
| ring_overlay share | 4.3% | 7.3% | — |
| gather share | 3.3% | 5.3% | — |

Dequant is still the largest phase but it's now < 75% of attention time
instead of ~84%. sdpa is the next lever if we want to push further.

## Why this works

`_apply_block_hadamard(identity, signs, inverse=True)` has shape
`[dim, dim]`: row i is `T(e_i)` where T is the block-Hadamard inverse
transform. Since T is linear, `T(x) = sum_j x_j · T(e_j) = x @ M`. The
matrix is built once per (device, dim, sign-seed) and reused forever.

## What was **not** done

- No Triton kernel. The user asked explicitly whether Triton is
  necessary; this rewrite proves it is not for this gap size.
- No `torch.compile`. That path hung in prior attempts (see
  `step2_after_compile.md`). The matmul rewrite gives a structural
  speedup without compilation fragility.

## Remaining levers (if we want more)

1. `sdpa` (12.7%) — currently full fp32 `scaled_dot_product_attention`
   on dequantised K/V. Could drop to bf16 once we re-verify coherence.
2. `ring_overlay` (7.3%) — per-sequence Python loop; could fuse with
   the dequant output assembly.
3. Fuse per-group concatenation in `scatter_turboquant_output` with the
   matmul output — avoids one intermediate allocation.

None of these would alone close the gap to bf16 (32s vs a projected
~150s at 1.49× savings of the 348s step1 wall time). A Triton-fused
decode kernel remains the big lever — left for later.

## Files changed (uncommitted on branch `turboquant-dequant-perf`)

- `vllm/v1/attention/ops/turboquant_kv_cache.py`
  - New `_INVERSE_MATRIX_FROM_SIGNS_CACHE` and
    `_inverse_matrix_from_signs(signs, normalized)` helper.
  - Inner loop of `dequantize_turboquant_vectors` now does two matmuls
    instead of two FWHT butterfly stacks.

No call-site changes. No test changes. No new kernel files.
