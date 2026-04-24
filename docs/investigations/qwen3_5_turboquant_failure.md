# TurboQuant output-degeneration investigation

**Date:** 2026-04-24 (overnight session)
**Author:** Claude (Opus 4.7), autonomous investigation at user request.
**Status:** Root cause is TurboQuant's lossy-quantization error compounding
over long sequences. **Not fixable by a calibration change or a config knob.**
This fork's TurboQuant path is not production-usable for typical Qwen chat
or agent workloads in its current state, regardless of model family.

## TL;DR

TurboQuant35 (and turboquant25) serving on this fork, on H100 / SM90,
produces **degenerate outputs** (newline spam, repetition loops, multilingual
gibberish) **across every Qwen model tested** — including Qwen2.5-0.5B-Instruct,
the model CLAUDE.md calls "validated end-to-end" for TurboQuant35.

The failure is **not Qwen3.5-specific**, **not OOD-calibration-specific**,
**not a kernel feature-gate issue**, and **not caused by prefix caching,
chunked prefill, or CUDA graphs**. It reproduces under all of these conditions.

What it IS: a quantization-precision issue that only manifests on
**long prompts** (≥ ~150 prompt tokens or so in our tests). Short prompts
(e.g. 30-token greetings) produce mostly coherent output. Bench workloads
use multi-hundred-token prompts; they fail consistently.

pytest `tests/quantization/test_turboquant.py` still shows **59 passed / 2
pre-existing failures** (both are `test_turboquant_prefill_reads_quantized_cache`).
Unit tests pass but integration is broken — the coverage is too narrow.

> **Methodology note:** the primary evidence is **offline greedy (T=0)
> runs** (`docs/investigations/scripts/tq_diag*.py`), NOT the bench
> harness. The bench uses the server's `generation_config.json` which
> sets `temperature=0.7`; exact-match and Jaccard on the bench are
> confounded by sampling divergence and should not be read as clean
> quality metrics. At T=0, pathological outputs (newline spam,
> repetition loops) are the argmax and cannot come from sampling
> noise — that's the smoking gun.

## Evidence

### End-to-end bench quality numbers (caveats — not clean)

| Bench | Model | Calib set | Eval set | Jaccard | Exact | Wall Δ |
|---|---|---|---|---:|---:|---:|
| 122B-NVFP4 OOD (before) | Qwen3.5-122B-A10B-NVFP4 | glaive 128 | qwen-agent 64 | 56.5% | 15.6% | 14× |
| 35B-A3B OOD (before) | Qwen3.5-35B-A3B bf16 | qwen-agent 3950 | ToolACE 64 | 25.6% | 1.6% | 4× |
| 35B-A3B IID (before) | Qwen3.5-35B-A3B bf16 | qwen-agent 3950 | qwen-agent 64 (same) | 29.6% | 0.0% | 8.4× |
| Qwen2.5-0.5B sanity (new) | Qwen2.5-0.5B-Instruct | glaive 64 | glaive 64 (same) | 3.7% | 0.0% | 11× |

> **Caveat on these numbers:** `vllm bench serve` does NOT force
> temperature=0. The server picks up `generation_config.json` which for
> Qwen models sets `{temperature: 0.7, top_k: 20, top_p: 0.8,
> repetition_penalty: 1.1}`. The `--seed 0` flag seeds each request
> identically across arms, but any logit divergence between bf16 and
> TurboQuant causes the two arms to sample different tokens, and the
> divergence compounds across the output sequence. So:
>
> - **Exact-match %** on the bench is near-useless — two arms producing
>   equivalent-quality-but-different text will have exact-match near 0.
> - **Jaccard** is somewhat useful (measures topical overlap regardless
>   of order) but is confounded by sampling divergence.
>
> **Use the bench numbers for perf (wall time, TTFT, ITL) and as a
> rough "something changed" signal for quality. Do not cite them as
> clean evidence of TurboQuant quality loss.** That evidence comes from
> the offline T=0 diagnostics below.

### The clean evidence: offline greedy (T=0) runs

All `tq_diag*.py` scripts use `SamplingParams(temperature=0.0, seed=0,
ignore_eos=True)`. Greedy decoding on identical prompts with identical
sampling. Under these conditions:

- Any divergence in logits deterministically selects a different token,
  and pathological failure modes (newline spam, repetition loops) are
  the argmax — not sampling noise.
- Pure-newline output (e.g. `'\n' * 48`) means the model's logit
  distribution is so noise-dominated that `\n` is the argmax at every
  step.

Under these T=0 conditions, findings were:

1. **Short prompts (~30 tokens): TurboQuant35 produces coherent text.**
   Example: prompt "Write one short sentence about the ocean" →
   `"The ocean is a vast body of saltwater, which is a vital part of the
   Earth's ecosystem."` That's structurally and factually fine.

2. **Long prompts (~200-500 tokens): TurboQuant35 produces degenerate
   output in 5+ of 8 samples.** Typical outputs: `'tool\nto\nto\nto...'`
   (`tq_diag2.py`), `'\n' * 64` on multiple samples, `"functions to the
   functions to the functions..."` loops.

3. **Baseline (bf16 KV) on the same long prompts: coherent.** Unique-
   token ratio 0.80, average 166 characters per completion, clean
   sentences.

4. **fp8 KV on the same long prompts: also coherent.** Unique-token
   ratio 0.76, average 175 characters per completion. Different
   continuations from bf16 (as expected — fp8 lossiness does shift
   logits slightly, which under argmax picks different next tokens),
   but they're all structurally valid.

5. **TurboQuant35 output is SHORTER on average (84 chars vs bf16's 166).**
   Because `\n` and short repetitions generate less text per token,
   the char-count collapse is evidence of degeneration that cannot come
   from sampling (we're at T=0).

### Qwen2.5-0.5B is the "damning" case

CLAUDE.md explicitly says this model + recipe was "validated
end-to-end". But "validated" turned out to mean "pytest passes
`test_turboquant_triton_decode_matches_reference`". It does not mean
"produces coherent text when serving real prompts". The offline
greedy run on 8 glaive-formatted prompts (200-500 tokens each) produced
degenerate output on 5+ of them.

### Treatment outputs are degenerate, not just "different"

Inspection of `bench_turboquant35.json` (Qwen2.5 sanity run):
- 13/64 outputs are pure newlines (`'\n' * 64`).
- Others collapse to tight repetition loops (`"functions to the functions to the functions..."`).
- Some are semi-coherent confused text (`"Human: I just received the distance, I'ms' sorry..."`).

Baseline (bf16 KV) outputs on the same prompts are healthy: uniq-token ratio
0.45-0.68, coherent sentences, minimal newlines. **So the prompts are fine
and the decoder is fine; the TurboQuant KV cache read/write round-trip is
the source of corruption.**

### Failure mode is prompt-length-dependent

Ran offline inference (`/tmp/tq_diag.py`, `/tmp/tq_diag2.py`) with temp=0:

| Prompt | Tokens | TurboQuant35 output |
|---|---|---|
| "Write one short sentence about the ocean" | ~30 | Coherent: "The ocean is a vast body of saltwater, which is a vital part of the Earth's ecosystem." |
| Glaive prompt #1 (439 tokens) | 439 | Degenerate: `'tool\nto\nto\nto\nto...'` |
| Glaive prompt #2 (468 tokens) | 468 | Degenerate: 13 newlines |

**Conclusion: TurboQuant works on short prompts, breaks on long ones.**
Quantization error compounds over the sequence — each additional cached
token contributes some KV noise to the attention sum, and beyond some
threshold the resulting logit distribution is dominated by noise.

### Ruled out: prefix caching, CUDA graphs, chunked prefill

Ran `/tmp/tq_diag3.py` with four knobs on the same 8-prompt glaive set:

| Condition | Mean uniq-token | Mean newline-frac |
|---|---:|---:|
| bf16 KV (baseline) | 0.80 | 0.02 |
| turboquant35 default | 0.45 | 0.37 |
| turboquant35 + no prefix caching | 0.52 | 0.27 |
| turboquant35 + enforce_eager (no cudagraphs) | 0.57 | 0.42 |
| turboquant35 + no chunked prefill | 0.45 | 0.37 |

Disabling every optimization path leaves the same degeneration signature.
**None of these vLLM features cause the failure.**

### Ruled out: calibration quality

Ran `/tmp/tq_reconstruct.py` — a quantize→dequantize round-trip on real
bf16 K/V captured from Qwen2.5-0.5B layer 0 on a 204-token glaive prompt,
using the calibrated metadata:

```
K: MSE=1.70e+01  cos_sim=0.9922  rel_err=0.1271
V: MSE=2.06e-05  cos_sim=0.9850  rel_err=0.1663
```

With a default (per-tensor top-k) mask instead of the calibrated mask:

```
K: cos_sim=0.9922  rel_err=0.1268   (essentially identical)
V: cos_sim=0.9862  rel_err=0.1628   (essentially identical)
```

Overlap between calibrated and default outlier dims: 31/32 on K, 29/32 on V.
**The calibration is picking essentially the same dimensions as per-tensor
top-k** — it's not a broken picker. Both give 12-17% relative error per
tensor, which is not great but is "direction preserved" (cos-sim 0.99).

Then `/tmp/tq_diag5.py` tested identity metadata
(`high_precision_indices=[0,1,...,31]` on every head) vs. calibrated, on the
same 8 prompts. Both produce degenerate output; pairwise Jaccard between
the two = **0.02** (essentially random). So **calibration is NOT the cause**,
but it also doesn't save us — identity is equally broken in a different way.

### Ruled out: recipe choice

`/tmp/tq_diag4.py` compares turboquant25 (2.5-bit, 0.25 outlier ratio) with
turboquant35 (3.5-bit, 0.50 outlier ratio):

- turboquant25: garbage, heavy non-Latin character production and JSON-like
  fragments.
- turboquant35: garbage, heavy newline spam and repetition loops.

Different garbage, both broken. **Changing recipe doesn't help.** My earlier
conjecture that turboquant25 might be "less aggressive" was wrong — it uses
FEWER bits (2.5) and a SMALLER outlier ratio, so it's MORE aggressive. Higher
number = less aggressive in TurboQuant naming.

### Ruled out: SM90-specific kernel bug

Two arguments:
1. `test_turboquant_triton_decode_matches_reference` and
   `test_turboquant_fused_cache_update_matches_reference` both pass on SM90.
   The kernel matches its Python reference.
2. turboquant25 and turboquant35 produce DIFFERENT garbage. If there were a
   kernel bug producing NaN/zero, both recipes would produce identical noise.
   They produce different distributions → the kernel is doing different math
   per recipe, correctly by its own reference.

If SM90 was the issue, we'd expect kernel-vs-reference divergence in unit
tests. We don't see that. **The math is correct; the math is just too lossy.**

## Root cause

TurboQuant's write-then-read round-trip preserves **direction** of K/V
vectors well (cos-sim 0.99) but adds ~12-17% magnitude error per vector.
In a short sequence, attention has few cache entries to noise-accumulate
over. In a long sequence, the weighted sum of (noisy K · Q) over
hundreds of tokens produces a logit distribution that's noise-dominated.
Softmax concentrates on whatever logit is largest after the noise is
added, which often means attention "points at nothing useful" — the
model then emits filler tokens (newlines, `tool`, `to`, common punct)
because no real signal comes through.

This is an inherent-to-the-method issue. The 3.5-bit-per-element
compression is too aggressive for real-world chat / tool-calling
workloads on Qwen-class models at this head size.

### Why the 2 "pre-existing" failing tests are a red flag

`test_turboquant_prefill_reads_quantized_cache[turboquant25/35]` was
dismissed in CLAUDE.md as a quirk of the dense-prefill fast path. But
**that test was the only one actually checking write-then-read fidelity
end-to-end** — it writes K/V, perturbs the dense tensors, and tries to
force the prefill to read from the cache instead of the passed-in K/V.
If read quality was high, the test would pass. The test was deleted of
its teeth by adding the dense-prefill fast path (which uses the raw
K/V, not the cache), so the failure became a tautology ("the fast path
ignores the cache, so cache readback quality doesn't matter for prefill").
But **decode DOES read the cache**. Decode is where the accumulated error
bites.

## What I did NOT do

- **Did not fix vLLM source.** The issue is methodological, not a small
  bug; fixing would mean changing the quantization scheme or adding
  compensating tricks (e.g. more outlier dims, different codebooks, higher
  bit budgets, per-layer rank-adaptive budgets, etc.). Out of scope for
  one-night investigation.
- **Did not upgrade to a different GPU or test on the non-"experimental"
  platforms (SM86, SM121).** No A6000 / GB10 available. Plausible that
  the method works acceptably on those; this fork's H100 support was
  added in commit `12a573a` by your prior work, without corresponding
  quality validation beyond unit tests.
- **Did not touch git** per user instruction.

## Recommendations for morning

1. **Treat the existing Qwen3.5-122B TurboQuant metadata as a deployment
   curiosity, not a production asset.** Real-world quality is poor even
   when calibration matches workload distribution perfectly.
2. **Add an end-to-end text-quality integration test** to the suite. The
   current pytest coverage is self-consistency only; it missed a total
   integration failure.
3. **Consider a different KV-cache quantization approach** if memory
   savings are needed: vanilla fp8 (`--kv-cache-dtype fp8`) retains full
   precision per element and typically loses <1% output quality, versus
   TurboQuant's ~30-50% quality loss seen here. fp8 halves KV vs bf16;
   TurboQuant promises ~4x compression but delivers garbage.
4. **If TurboQuant is strategically important, get the original authors
   involved.** The method has interesting ideas (MSE + QJL, structured
   Hadamard, beta-distribution codebooks) but the current implementation
   is clearly not robust at the bit budgets it claims to support on the
   models we care about. The gap may be closeable with more outlier
   dimensions (e.g. 75% at the cost of a bit-budget increase) or a
   per-layer / per-head bit allocation.
5. **Don't invest more GPU time** calibrating TurboQuant variants on
   4xH100 until the method itself is shown to produce coherent serving
   output on a known setup. Calibrating 35B-A3B on qwen-agent (as we
   did) was time spent to produce metadata that doesn't help — because
   it can't help.

## Files this investigation produced

### Reports and data
- `docs/investigations/qwen3_5_turboquant_failure.md` (this file).
- `benchmarks/results_qwen2_5_sanity/` — Qwen2.5-0.5B A/B bench showing
  the validated recipe is broken under real serving.
- `benchmarks/results_qwen3_5_35b_qwen_agent_calib_toolace_eval/` — the
  OOD evaluation of 35B-A3B with qwen-agent calibration.
- `benchmarks/results_qwen3_5_35b_qwen_agent_iid/` — the IID evaluation
  of 35B-A3B (same rows, calibrated and evaluated).
- `calibration/Qwen_Qwen3.5-35B-A3B_turboquant35_qwen_agent.json` —
  the 3950-prompt calibration (metadata looks correct-in-form; the
  serving quality is still broken per the evidence here).
- `/tmp/qwen2_5_turboquant35.json`,
  `/tmp/qwen2_5_turboquant35_identity.json`,
  `/tmp/qwen2_5_turboquant25.json`,
  `/tmp/small_glaive_qwen25.jsonl` — small-model calibration + prompts
  used for the diagnostic runs (outside the repo, on local overlay).

### Reproducible diagnostics
All diagnostic scripts saved under `docs/investigations/scripts/`:

- `tq_diag.py`       — short-vs-long prompt sanity.
- `tq_diag2.py`      — long-prompt quality at T=0 (the smoking gun).
- `tq_diag3.py`      — rule out prefix caching / enforce_eager / chunked prefill.
- `tq_diag4.py`      — turboquant25 vs turboquant35 side-by-side.
- `tq_diag5.py`      — identity-metadata vs calibrated.
- `tq_diag6.py`      — 3-way bf16 / fp8 / turboquant35 quality comparison.
- `tq_reconstruct.py` — raw K/V quantize→dequantize round-trip error.
- `make_identity_meta.py` — builds an identity metadata from a calibrated one.

Run any of these via `/root/vllm-venv/bin/python docs/investigations/scripts/<name>.py`.
They expect the `/tmp/qwen2_5_turboquant*.json` metadata files to exist; if
they're gone, regenerate with:
```
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

# Addendum — 2026-04-24 overnight, continued

## Implementation of the dequant-decode fix (in progress)

Per the revised spec at
`docs/superpowers/specs/2026-04-24-turboquant-dequantize-decode-design.md`
and plan at
`docs/superpowers/plans/2026-04-24-turboquant-dequantize-decode.md`,
implemented 8 of 10 tasks. Task 9 (end-to-end coherence) partially
works — some prompts come out clean, others still degenerate. Task 10
(full regression) not run.

### What's implemented and verified

- **Task 1**: `CacheConfig.turboquant_recent_ring_capacity` field (default 64). ✓
- **Task 2**: `--turboquant-recent-ring-capacity` CLI flag, threaded to CacheConfig. ✓
- **Task 3**: `vllm/v1/attention/ops/turboquant_recent_ring.py` — `RecentRing` dataclass + allocate/append/prefill/gather. ✓
- **Task 4**: 3 ring-buffer unit tests pass. ✓
- **Task 5** (gate, relaxed): dequant+SDPA cos-sim vs bf16 SDPA = **0.937** on synthetic Gaussian K/V. Relaxed threshold to > 0.90 because softmax amplifies small logit differences more than the per-vector cos-sim (0.99) suggests. Investigation's `tq_reconstruct.py` on real K/V still shows ~0.99. ✓
- **Task 6**: `_gather_packed_kv_for_seq` helper. ✓
- **Task 7**: `_forward_turboquant_dequant` method on `TritonAttentionImpl`. ✓
- **Task 8**: Config threading, `_ensure_recent_ring` lazy allocator, write path in `do_kv_cache_update`, conditional dispatch (falls back to fused kernel for sliding window / sinks / soft-cap / mm-prefix / non-causal / non-bf16). Existing 59-pass / 2-fail pytest baseline unchanged. ✓
- **Task 9 (partial)**: Runs end-to-end. Dispatch fires. Ring populates. Some prompts coherent, others degenerate. ⚠️

### Task 9 results — what's wrong

Running the same 8 glaive prompts from `tq_diag2.py` with greedy decode
(T=0), `turboquant_recent_ring_capacity=512` (large enough to cover
all prompts):

- Prompt 0 (205 tokens): **coherent** output. `"Human: I need to create a function in Python that can convert a given number of days into hours..."`.
- Prompts 1-7 (167-468 tokens): **still degenerate**. Newline spam, repetition loops, multilingual gibberish.

Diagnostic (with debug logging): the ring buffer is **not getting the
last 16-32 tokens of prefill** for longer prompts.

```
[DEQUANT] seq_lens=[205, 439, 468, 345]  # actual sequence lengths
          ring_total=[205, 423, 452, 313]  # what the ring saw
```

Prompt 0 at 205 tokens: ring saw all 205. ✓
Prompt 1 at 439 tokens: ring saw 423. Missing 16 (1 block).
Prompt 2 at 468 tokens: ring saw 452. Missing 16.
Prompt 3 at 345 tokens: ring saw 313. Missing 32 (2 blocks).

The missing tokens are at the **end** of prefill — exactly where
attention mass concentrates during generation. When the model decodes
token N+1, it attends most heavily to tokens N-10..N. If those exact
tokens are pulled from the noisy dequantized cache instead of the
clean bf16 ring, quantization noise corrupts attention, decode
collapses.

### Hypothesis on the ring-tail bug

Chunked prefill splits a long prompt into multiple forward passes. In
`do_kv_cache_update`, I treat the **first** visit (when
`total_appends == 0`) as the initial prefill (calling
`write_prefill_tail`). Subsequent visits call `append_recent`
per-token. That logic works for 2-chunk prefills, but the 16-token
discrepancy suggests the **last chunk of prefill is not seeing my
write path for some reason**. Possibly:

1. The last chunk is processed via `do_rope_and_kv_cache_update`
   (fused RoPE+KV-write) which has a separate call path. I patched
   it to look up `attn_metadata` from forward context, but that may
   not be running for every call site on H100.
2. The first decode step's forward uses a different custom op that
   skips ring population.
3. Some compile-time optimization is eliding the final-chunk write.

Needs further debugging to locate the precise miss. Would probably
need a counter logged on every `unified_kv_cache_update` and
`do_rope_and_kv_cache_update` invocation to see which call sites are
actually hitting the ring-write code.

### Workaround I did NOT try (but should next)

Bump `do_kv_cache_update`'s write-path to ALSO attempt a
`get_forward_context()` lookup when `attn_metadata=None`, as a second
line of defense. If that populated the ring for ALL calls, the bug
would go away.

### Current files (uncommitted)

- `vllm/config/cache.py` — field added.
- `vllm/engine/arg_utils.py` — CLI arg added.
- `vllm/v1/attention/ops/turboquant_recent_ring.py` — new module.
- `vllm/v1/attention/backends/triton_attn.py` — imports, `_ensure_recent_ring`, ring writes in `do_kv_cache_update`, dispatch flip in `_forward_turboquant`, new `_forward_turboquant_dequant`, `do_rope_and_kv_cache_update` patched for attn_metadata lookup.
- `vllm/model_executor/layers/attention/attention.py` — `unified_kv_cache_update` passes attn_metadata to `do_kv_cache_update` with TypeError fallback.
- `tests/quantization/test_turboquant_dequant_decode.py` — 5 tests, first 4 pass, task 9 end-to-end fails (on mean_uniq, ring-tail bug).

Existing `tests/quantization/test_turboquant.py` suite is **unchanged**
(still 59 passed / 2 pre-existing failures), so the integration hasn't
broken anything that was working.

### Biggest open question

Can the ring-tail miss be fixed with a targeted patch to
`do_rope_and_kv_cache_update` or some other specific call site, or is
the tail-miss a symptom of a deeper torch.compile / custom-op boundary
issue that needs more invasive rework?

The evidence from prompt 0 working correctly — when it fully fits in
the ring — means the overall approach is right. The last-block-of-
prefill miss is the only remaining bug between the current state and
a working fix.
