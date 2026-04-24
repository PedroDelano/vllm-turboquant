# TurboQuant Dequant Perf — Implementation Plan

> **Status (2026-04-24):** Step 1 landed as planned. Step 2 was replaced:
> `torch.compile` hung on first compile under both `mode="reduce-overhead"`
> and `mode="default"` (see `benchmarks/results_tq_dequant_perf/step2_after_compile.md`).
> A simpler structural rewrite — replacing the FWHT butterfly loops in
> `dequantize_turboquant_vectors` with matmuls against precomputed inverse
> matrices — shipped instead, with equivalent numerical output and no
> compilation fragility. Results in
> `benchmarks/results_tq_dequant_perf/step2_matmul_rewrite.md`.
> Wall-time: 348 s → 301 s (-13.5%), median ITL: 50.8 ms → 39.1 ms (-23%).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 12.5× wall-time gap between TurboQuant decode (404 s) and bf16 (32 s) on the 35B-A3B IID bench, by (1) batching `dequantize_turboquant_vectors` across sequences in the current batch, then (2) `torch.compile`-ing the dequant function so its ~8 PyTorch ops fuse.

**Architecture:** Sequential, measurement-gated. Step 1 restructures the per-seq loop in `_forward_turboquant_dequant` to call dequant once over concatenated packed K/V; Step 2 wraps `dequantize_turboquant_vectors` with `torch.compile(mode="reduce-overhead", dynamic=True, fullgraph=False)` via a lazy accessor. After each step we re-profile with `TQ_PROFILE=1` and decide whether the next step is needed.

**Tech Stack:** PyTorch (bf16), `torch.compile` with CUDA-graph-backed reduce-overhead mode, vLLM v1 attention backend.

**Commit policy (user override):** NO commits until the user OKs. Work stays on branch `turboquant-dequant-perf`, uncommitted.

---

## File Structure

**Modified:**
- `vllm/v1/attention/backends/triton_attn.py` — `_forward_turboquant_dequant` restructured per-seq loop (Step 1); dispatch to compiled dequant accessor + `TQ_DEQUANT` env var fallback (Step 2).
- `vllm/v1/attention/ops/turboquant_kv_cache.py` — adds `_get_compiled_dequant()` accessor (Step 2).

**New:**
- `tests/quantization/test_turboquant_dequant_decode.py` — append one new unit test `test_batched_dequant_equals_per_seq_dequant` (Step 1).
- `benchmarks/results_tq_dequant_perf/` — directory for phase-breakdown profile reports (populated during measurement tasks).

**Untouched:**
- The profiling instrumentation (`TQ_PROFILE=1`) already in `_forward_turboquant_dequant` stays in place. We re-run it before/after each step to measure.
- `dequantize_turboquant_vectors`' signature and body stay exactly as they are; we're only changing call sites.
- No changes to ring buffer, SDPA, gather helpers, cache config, or CLI.

---

## Shared fixtures

- Repo root: `/workspace/vllm-turboquant/`
- Venv: `/root/vllm-venv/bin/python`, `/root/vllm-venv/bin/pytest`
- Small-model snapshot: `/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- Small-model calibration: `/tmp/qwen2_5_turboquant35.json`
- Small-model prompts: `/tmp/small_glaive_qwen25.jsonl`
- 35B-A3B fixtures (for end-to-end bench): `/workspace/hf-cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307`, `calibration/Qwen_Qwen3.5-35B-A3B_turboquant35_qwen_agent.json`, `calibration/prompts/qwen_agent_qwen3_5_35b.jsonl`.

---

## Task 0: Baseline measurement

**Files:** none modified (measurement only).

- [ ] **Step 0.1: Capture baseline profile on current `turboquant-dequant-perf` HEAD**

```bash
mkdir -p /workspace/vllm-turboquant/benchmarks/results_tq_dequant_perf
TQ_PROFILE=1 /root/vllm-venv/bin/pytest \
  /workspace/vllm-turboquant/tests/quantization/test_turboquant_dequant_decode.py::test_long_prompts_coherent_with_dequant_decode \
  -v -s > /tmp/tqprof_step0.log 2>&1
```

- [ ] **Step 0.2: Extract phase breakdown from log**

```bash
grep -A7 "after 50 calls" /tmp/tqprof_step0.log | head -40 \
  > /workspace/vllm-turboquant/benchmarks/results_tq_dequant_perf/step0_baseline.md
cat /workspace/vllm-turboquant/benchmarks/results_tq_dequant_perf/step0_baseline.md
```

Expected output: the familiar breakdown (gather ≤ 3%, dequant_k ~47%, dequant_v ~44%, sdpa ~3.5%, ring_overlay ~2%, total ~430-500ms per 50 calls × 4 seqs).

- [ ] **Step 0.3: Record the snapshot**

If the numbers deviate materially from the expected baseline (e.g. dequant share < 80% or > 95%), stop and investigate — something changed since the profile was last captured. Otherwise proceed.

---

## Task 1: New test — batched dequant equals per-seq dequant

**Files:**
- Modify: `tests/quantization/test_turboquant_dequant_decode.py` (append one test)

TDD step: write the test first, confirming the invariant that makes Step 1 safe. This test passes against the CURRENT non-batched codepath (it doesn't call into the attention backend; it tests the purity of `dequantize_turboquant_vectors`). Its job is to lock in the correctness contract before we rely on it for the refactor.

- [ ] **Step 1.1: Add the test**

Append to `/workspace/vllm-turboquant/tests/quantization/test_turboquant_dequant_decode.py`:

```python
def test_batched_dequant_equals_per_seq_dequant():
    """`dequantize_turboquant_vectors` is pure over its token dimension:
    concatenating multiple sequences' packed bytes along dim 0 and running
    one dequant call must produce element-equal output to running dequant
    per sequence then concatenating."""
    import torch
    from vllm.v1.attention.ops.turboquant_kv_cache import (
        build_turboquant_outlier_masks,
        dequantize_turboquant_vectors,
        get_turboquant_centroids,
        get_turboquant_layout,
        get_turboquant_qjl_matrix,
        get_turboquant_rotation,
        quantize_turboquant_vectors,
    )

    torch.manual_seed(1)
    head_dim = 64
    num_kv_heads = 2
    recipe = "turboquant35"
    device = torch.device("cuda")

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

    # Three sequences with different lengths.
    seq_lens = [37, 128, 77]
    K_list = [
        torch.randn(L, num_kv_heads, head_dim, device=device) * 2.0
        for L in seq_lens
    ]
    # Use one shared mask (matches real usage: masks are per-layer, not per-seq).
    masks = build_turboquant_outlier_masks(K_list[0].float(), recipe)

    packed_list = [
        quantize_turboquant_vectors(
            k.float(), recipe, rotations, qjl_matrices, centroids, masks,
        )
        for k in K_list
    ]

    # Per-seq dequant.
    per_seq_out = torch.cat(
        [
            dequantize_turboquant_vectors(
                p, recipe, head_dim,
                rotations, qjl_matrices, centroids, masks, torch.bfloat16,
            )
            for p in packed_list
        ],
        dim=0,
    )

    # Batched dequant.
    packed_all = torch.cat(packed_list, dim=0)
    batched_out = dequantize_turboquant_vectors(
        packed_all, recipe, head_dim,
        rotations, qjl_matrices, centroids, masks, torch.bfloat16,
    )

    assert per_seq_out.shape == batched_out.shape
    # Bit-identical on bf16 output (same math, same reductions, pure function).
    assert torch.equal(per_seq_out, batched_out), (
        f"max abs diff = {(per_seq_out.float() - batched_out.float()).abs().max().item()}"
    )
```

- [ ] **Step 1.2: Run**

```bash
/root/vllm-venv/bin/pytest \
  /workspace/vllm-turboquant/tests/quantization/test_turboquant_dequant_decode.py::test_batched_dequant_equals_per_seq_dequant \
  -v 2>&1 | tail -10
```

Expected: PASS on the unchanged codebase. If it fails with a max-abs-diff > 0, the dequant function has a non-associative reduction somewhere and Step 1's approach is unsafe — STOP and escalate.

---

## Task 2: Step 1 — batch the per-seq dequant loop

**Files:**
- Modify: `vllm/v1/attention/backends/triton_attn.py::_forward_turboquant_dequant`

- [ ] **Step 2.1: Read the current method to locate insertion points**

```bash
sed -n '1743,1890p' /workspace/vllm-turboquant/vllm/v1/attention/backends/triton_attn.py
```

Note: the method currently has `TQ_PROFILE=1` instrumentation blocks around gather, dequant, and sdpa phases. Those stay; we're restructuring the loop body, not deleting the instrumentation.

- [ ] **Step 2.2: Restructure the method**

Open `/workspace/vllm-turboquant/vllm/v1/attention/backends/triton_attn.py` and find `_forward_turboquant_dequant`. Replace the single `for seq_id in range(num_seqs):` loop body with a two-phase structure:

**Phase A** — gather all packed K/V and a single batched dequant:

```python
        # Step 1 (batched): gather all sequences' packed bytes into a single
        # contiguous tensor, call dequant once per side, then split back
        # per-seq for ring overlay + SDPA. Motivated by profiling showing
        # ~90% of decode time was in per-seq dequant calls dominated by
        # kernel-launch overhead. See
        # docs/superpowers/specs/2026-04-24-turboquant-dequant-perf-design.md.
        packed_k_list: list[torch.Tensor] = []
        packed_v_list: list[torch.Tensor] = []
        seq_offsets: list[int] = [0]
        # seq_id -> (dequant_start, dequant_end, query_start, query_end)
        seq_ranges: list[tuple[int, int, int, int] | None] = []

        if _profile:
            _t0 = torch.cuda.Event(enable_timing=True)
            _t1 = torch.cuda.Event(enable_timing=True)
            _t0.record()

        for seq_id in range(num_seqs):
            q_start = int(query_start_loc_cpu[seq_id].item())
            q_end = int(query_start_loc_cpu[seq_id + 1].item())
            if q_end == q_start:
                seq_ranges.append(None)
                continue
            seq_len = int(seq_lens_cpu[seq_id].item())
            block_table_row = attn_metadata.block_table[seq_id]
            packed_k_list.append(
                _gather_packed_kv_for_seq(key_cache, block_table_row, seq_len)
            )
            packed_v_list.append(
                _gather_packed_kv_for_seq(value_cache, block_table_row, seq_len)
            )
            d_start = seq_offsets[-1]
            d_end = d_start + seq_len
            seq_offsets.append(d_end)
            seq_ranges.append((d_start, d_end, q_start, q_end))

        if _profile:
            _t1.record()
            torch.cuda.synchronize()
            self._tq_phase_ms["gather_k"] += _t0.elapsed_time(_t1)
            # Old profile had separate gather_k / gather_v; in batched mode
            # they share a single timed region. gather_v stays at 0.

        if not packed_k_list:
            return output

        packed_k_all = torch.cat(packed_k_list, dim=0)
        packed_v_all = torch.cat(packed_v_list, dim=0)

        if _profile:
            _t0 = torch.cuda.Event(enable_timing=True)
            _t1 = torch.cuda.Event(enable_timing=True)
            _t0.record()
        dequant_k_all = dequantize_turboquant_vectors(
            packed_k_all, self.kv_cache_dtype, self.head_size,
            rotations, qjl_matrices, centroids, key_masks,
            torch.bfloat16,
        )
        if _profile:
            _t1.record()
            torch.cuda.synchronize()
            self._tq_phase_ms["dequant_k"] += _t0.elapsed_time(_t1)

        if _profile:
            _t0 = torch.cuda.Event(enable_timing=True)
            _t1 = torch.cuda.Event(enable_timing=True)
            _t0.record()
        dequant_v_all = dequantize_turboquant_vectors(
            packed_v_all, self.kv_cache_dtype, self.head_size,
            rotations, qjl_matrices, centroids, value_masks,
            torch.bfloat16,
        )
        if _profile:
            _t1.record()
            torch.cuda.synchronize()
            self._tq_phase_ms["dequant_v"] += _t0.elapsed_time(_t1)
```

**Phase B** — per-seq ring overlay + SDPA, now operating on slices of the batched dequant:

```python
        for seq_id in range(num_seqs):
            rng = seq_ranges[seq_id]
            if rng is None:
                continue
            d_start, d_end, q_start, q_end = rng
            dequant_k = dequant_k_all[d_start:d_end]
            dequant_v = dequant_v_all[d_start:d_end]

            if recent_ring is not None:
                if _profile:
                    _t0 = torch.cuda.Event(enable_timing=True)
                    _t1 = torch.cuda.Event(enable_timing=True)
                    _t0.record()
                recent_k, recent_v, n_recent = gather_recent(
                    recent_ring, seq_id=seq_id
                )
                n_use = min(n_recent, dequant_k.shape[0])
                if n_use > 0:
                    dequant_k[-n_use:] = recent_k[-n_use:]
                    dequant_v[-n_use:] = recent_v[-n_use:]
                if _profile:
                    _t1.record()
                    torch.cuda.synchronize()
                    self._tq_phase_ms["ring_overlay"] += _t0.elapsed_time(_t1)

            if _profile:
                _t0 = torch.cuda.Event(enable_timing=True)
                _t1 = torch.cuda.Event(enable_timing=True)
                _t0.record()
            q_3d = query[q_start:q_end].view(
                q_end - q_start, self.num_heads, self.head_size
            )

            # SDPA expects (batch, heads, tokens, dim).
            kv_repeat = self.num_heads // self.num_kv_heads
            k_rep = (
                dequant_k.repeat_interleave(kv_repeat, dim=1)
                if kv_repeat > 1 else dequant_k
            )
            v_rep = (
                dequant_v.repeat_interleave(kv_repeat, dim=1)
                if kv_repeat > 1 else dequant_v
            )
            q_t = q_3d.transpose(0, 1).unsqueeze(0)
            k_t = k_rep.transpose(0, 1).unsqueeze(0)
            v_t = v_rep.transpose(0, 1).unsqueeze(0)

            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                is_causal=(q_end - q_start) > 1,
                scale=self.scale,
            )
            output[q_start:q_end] = out.squeeze(0).transpose(0, 1)
            if _profile:
                _t1.record()
                torch.cuda.synchronize()
                self._tq_phase_ms["sdpa"] += _t0.elapsed_time(_t1)
                self._tq_phase_total_seqs += 1

        if _profile:
            self._tq_phase_calls += 1
            if self._tq_phase_calls % 50 == 0:
                total = sum(self._tq_phase_ms.values())
                lines = [f"[TQPROF] after {self._tq_phase_calls} calls "
                         f"({self._tq_phase_total_seqs} total-seqs): "
                         f"total={total:.1f}ms"]
                for k in ("gather_k","gather_v","dequant_k","dequant_v",
                          "ring_overlay","sdpa"):
                    v = self._tq_phase_ms[k]
                    frac = 100.0 * v / max(total, 1e-6)
                    lines.append(f"  {k:14s} {v:9.2f}ms ({frac:5.1f}%)")
                logger.warning("%s", "\n".join(lines))

        return output
```

The above replaces the current single `for seq_id in range(num_seqs):` block plus its trailing profile-dump. The in-method state variables (`rotations`, `qjl_matrices`, `centroids`, `key_masks`, `value_masks`, `recent_ring`, `query_start_loc_cpu`, `seq_lens_cpu`, `num_seqs`, `_profile`, `_tq_phase_ms` init) above the loop are unchanged.

- [ ] **Step 2.3: Syntax check**

```bash
/root/vllm-venv/bin/python -c "from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl; print('ok')"
```

Expected: `ok`.

---

## Task 3: Correctness after Step 1

**Files:** none modified. Verification only.

- [ ] **Step 3.1: New unit test still passes**

```bash
/root/vllm-venv/bin/pytest \
  /workspace/vllm-turboquant/tests/quantization/test_turboquant_dequant_decode.py::test_batched_dequant_equals_per_seq_dequant \
  -v 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 3.2: Existing 5 dequant_decode tests still pass**

```bash
/root/vllm-venv/bin/pytest \
  /workspace/vllm-turboquant/tests/quantization/test_turboquant_dequant_decode.py \
  -v 2>&1 | tail -10
```

Expected: 6 passed (4 ring + 1 coherence + 1 new). Coherence test: `test_long_prompts_coherent_with_dequant_decode` still meets `mean_uniq > 0.5` and `mean_nl < 0.1`.

- [ ] **Step 3.3: No regression on existing turboquant suite**

```bash
/root/vllm-venv/bin/pytest /workspace/vllm-turboquant/tests/quantization/test_turboquant.py -q 2>&1 | tail -3
```

Expected: 59 passed, 2 pre-existing failures (unchanged).

If any of the above fails, diagnose before proceeding. Most likely cause: `seq_ranges` / `seq_offsets` indexing bug. Inspect the failing test's prompt lengths vs the computed ranges.

---

## Task 4: Measure Step 1 and decide

**Files:**
- Create: `benchmarks/results_tq_dequant_perf/step1_after_batch.md`

- [ ] **Step 4.1: Re-profile**

```bash
TQ_PROFILE=1 /root/vllm-venv/bin/pytest \
  /workspace/vllm-turboquant/tests/quantization/test_turboquant_dequant_decode.py::test_long_prompts_coherent_with_dequant_decode \
  -v -s > /tmp/tqprof_step1.log 2>&1
grep -A7 "after 50 calls" /tmp/tqprof_step1.log | head -40 \
  > /workspace/vllm-turboquant/benchmarks/results_tq_dequant_perf/step1_after_batch.md
```

- [ ] **Step 4.2: Compare vs baseline**

Expected: dequant_k + dequant_v share drops significantly (from ~91% → ~50-70%) as kernel-launch overhead gets amortized across seqs; `total` drops proportionally (from ~440 ms / 50 calls → ~150-250 ms). Exact numbers depend on `num_seqs` in the test (4 in our case).

Record both snapshots for comparison in the markdown file. Example format:

```
Baseline (step 0):
  total ~440 ms   dequant_k ~47%   dequant_v ~44%   sdpa ~3.5%

After batching (step 1):
  total ~???      dequant_k ~???   dequant_v ~???   sdpa ~???
```

- [ ] **Step 4.3: Decide — proceed to Step 2 or stop?**

If total dropped by ≥3× AND dequant combined share fell below 50%, we're likely near our target already. End-to-end bench via Task 10 can confirm; Step 2 may be skippable.

If total dropped by <2×, Step 2 is still needed. Proceed.

If total INCREASED (unexpected) — the `torch.cat` on packed bytes is dominating. File a follow-up to use pre-allocated scratch buffers; for now, try Step 2 anyway and re-measure.

---

## Task 5: Step 2 — add compiled-dequant accessor

**Files:**
- Modify: `vllm/v1/attention/ops/turboquant_kv_cache.py`

Add a lazy `torch.compile` wrapper near the existing `dequantize_turboquant_vectors` function.

- [ ] **Step 5.1: Locate `dequantize_turboquant_vectors`**

```bash
grep -n "^def dequantize_turboquant_vectors\|^def quantize_turboquant_vectors" \
  /workspace/vllm-turboquant/vllm/v1/attention/ops/turboquant_kv_cache.py
```

Note the line number so the accessor can be added immediately AFTER the function body ends.

- [ ] **Step 5.2: Append the accessor**

Append these lines after `dequantize_turboquant_vectors` (but before any existing private helpers that appear later in the file; use your editor to place right after the function's `return` statement):

```python
_compiled_dequantize_turboquant_vectors = None


def _get_compiled_dequant():
    """Return a cached, torch.compile'd wrapper around
    dequantize_turboquant_vectors.

    Compiled lazily so code paths that don't exercise the serving hot
    path don't pay compile overhead. `dynamic=True` shares one compiled
    graph across varying seq_len. `fullgraph=False` tolerates graph
    breaks on any `.item()` or Python-control-flow points inside the
    function. `mode="reduce-overhead"` folds launches into CUDA graphs
    where safe.
    """
    global _compiled_dequantize_turboquant_vectors
    if _compiled_dequantize_turboquant_vectors is None:
        import torch
        _compiled_dequantize_turboquant_vectors = torch.compile(
            dequantize_turboquant_vectors,
            mode="reduce-overhead",
            dynamic=True,
            fullgraph=False,
        )
    return _compiled_dequantize_turboquant_vectors
```

- [ ] **Step 5.3: Verify import**

```bash
/root/vllm-venv/bin/python -c "
from vllm.v1.attention.ops.turboquant_kv_cache import _get_compiled_dequant, dequantize_turboquant_vectors
f = _get_compiled_dequant()
print('compiled_fn:', type(f).__name__)
print('ok')
"
```

Expected output includes `compiled_fn: OptimizedModule` or similar. Then `ok`. If it fails on import, the previous step placed the code inside another function — find the correct module-level insertion point and try again.

---

## Task 6: Step 2 — wire compiled dequant + env-var fallback

**Files:**
- Modify: `vllm/v1/attention/backends/triton_attn.py::_forward_turboquant_dequant`

- [ ] **Step 6.1: Update import**

At the top of `_forward_turboquant_dequant`, alongside `_profile = _os.environ.get("TQ_PROFILE", "0") == "1"`, add:

```python
        _dequant_impl = _os.environ.get("TQ_DEQUANT", "compiled").lower()
```

- [ ] **Step 6.2: Import the accessor**

Find the existing imports from `vllm.v1.attention.ops.turboquant_kv_cache` near the top of `triton_attn.py`. Add `_get_compiled_dequant` to the import list:

```python
from vllm.v1.attention.ops.turboquant_kv_cache import (
    # ... existing imports ...
    dequantize_turboquant_vectors,
    _get_compiled_dequant,
)
```

- [ ] **Step 6.3: Select the implementation at the top of the method**

Right after the `_dequant_impl = ...` line, add:

```python
        if _dequant_impl == "python":
            _dequant_fn = dequantize_turboquant_vectors
        else:
            try:
                _dequant_fn = _get_compiled_dequant()
            except Exception as e:  # pragma: no cover — compile regression fallback
                logger.warning_once(
                    "torch.compile of dequantize_turboquant_vectors failed "
                    "(%s); falling back to the eager PyTorch implementation.",
                    e, scope="local",
                )
                _dequant_fn = dequantize_turboquant_vectors
```

- [ ] **Step 6.4: Replace the two dequant call sites**

In Phase A (the batched dequant stanza from Task 2), replace:

```python
        dequant_k_all = dequantize_turboquant_vectors(
            packed_k_all, self.kv_cache_dtype, self.head_size,
            rotations, qjl_matrices, centroids, key_masks,
            torch.bfloat16,
        )
```

with:

```python
        dequant_k_all = _dequant_fn(
            packed_k_all, self.kv_cache_dtype, self.head_size,
            rotations, qjl_matrices, centroids, key_masks,
            torch.bfloat16,
        )
```

And similarly replace the `dequant_v_all = dequantize_turboquant_vectors(...)` call with `dequant_v_all = _dequant_fn(...)`.

- [ ] **Step 6.5: Syntax + import check**

```bash
/root/vllm-venv/bin/python -c "from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl; print('ok')"
```

Expected: `ok`.

---

## Task 7: Correctness after Step 2

**Files:** none modified. Verification only.

Correctness here has two dimensions: (a) the compiled path still gets the right outputs, (b) the env-var fallback to the Python path still works (for bisection).

- [ ] **Step 7.1: Compiled path — dequant_decode suite**

```bash
/root/vllm-venv/bin/pytest \
  /workspace/vllm-turboquant/tests/quantization/test_turboquant_dequant_decode.py \
  -v 2>&1 | tail -10
```

Expected: 6 passed. Long-prompt coherence still meets mean_uniq > 0.5, mean_nl < 0.1.

- [ ] **Step 7.2: Python fallback still works**

```bash
TQ_DEQUANT=python /root/vllm-venv/bin/pytest \
  /workspace/vllm-turboquant/tests/quantization/test_turboquant_dequant_decode.py::test_long_prompts_coherent_with_dequant_decode \
  -v 2>&1 | tail -5
```

Expected: 1 passed. The fallback is exercised on this run and must give the same coherent outputs.

- [ ] **Step 7.3: Both paths produce the same outputs**

Quick diff test — run the long-prompt test under both modes, capture the generated text, and confirm they match token-for-token at T=0:

```bash
TQ_DEQUANT=compiled /root/vllm-venv/bin/python -c "
import json
from vllm import LLM, SamplingParams
SNAP = '/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
META = '/tmp/qwen2_5_turboquant35.json'
with open('/tmp/small_glaive_qwen25.jsonl') as f:
    prompts = [json.loads(l)['prompt'] for l in f][:4]
sp = SamplingParams(temperature=0.0, max_tokens=24, seed=0, ignore_eos=True)
llm = LLM(model=SNAP, dtype='bfloat16', max_model_len=2048,
          max_num_seqs=4, gpu_memory_utilization=0.75,
          attention_backend='TRITON_ATTN',
          kv_cache_dtype='turboquant35',
          enable_turboquant=True,
          turboquant_metadata_path=META,
          turboquant_recent_ring_capacity=1024,
          enable_prefix_caching=False)
print(repr([o.outputs[0].text for o in llm.generate(prompts, sp)]))
" > /tmp/dequant_compiled.txt 2>&1

TQ_DEQUANT=python /root/vllm-venv/bin/python -c "
import json
from vllm import LLM, SamplingParams
SNAP = '/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
META = '/tmp/qwen2_5_turboquant35.json'
with open('/tmp/small_glaive_qwen25.jsonl') as f:
    prompts = [json.loads(l)['prompt'] for l in f][:4]
sp = SamplingParams(temperature=0.0, max_tokens=24, seed=0, ignore_eos=True)
llm = LLM(model=SNAP, dtype='bfloat16', max_model_len=2048,
          max_num_seqs=4, gpu_memory_utilization=0.75,
          attention_backend='TRITON_ATTN',
          kv_cache_dtype='turboquant35',
          enable_turboquant=True,
          turboquant_metadata_path=META,
          turboquant_recent_ring_capacity=1024,
          enable_prefix_caching=False)
print(repr([o.outputs[0].text for o in llm.generate(prompts, sp)]))
" > /tmp/dequant_python.txt 2>&1

tail -1 /tmp/dequant_compiled.txt > /tmp/dequant_compiled_out.txt
tail -1 /tmp/dequant_python.txt > /tmp/dequant_python_out.txt
diff /tmp/dequant_compiled_out.txt /tmp/dequant_python_out.txt && echo "COMPILED EQUALS PYTHON" || echo "DIFF FOUND — compiled path differs"
```

Expected: `COMPILED EQUALS PYTHON`. Greedy decode (T=0) is deterministic, so the two dequant implementations should produce bit-identical generations. If they differ, `torch.compile` introduced a numerical issue — revert Step 2 and diagnose (likely a fallback to bf16 accumulator inside a fused region; might need `mode="default"` instead of `reduce-overhead`).

- [ ] **Step 7.4: No regression on existing turboquant suite**

```bash
/root/vllm-venv/bin/pytest /workspace/vllm-turboquant/tests/quantization/test_turboquant.py -q 2>&1 | tail -3
```

Expected: still 59 passed, 2 pre-existing failures.

---

## Task 8: Measure Step 2

**Files:**
- Create: `benchmarks/results_tq_dequant_perf/step2_after_compile.md`

- [ ] **Step 8.1: Re-profile with compiled path**

```bash
TQ_PROFILE=1 /root/vllm-venv/bin/pytest \
  /workspace/vllm-turboquant/tests/quantization/test_turboquant_dequant_decode.py::test_long_prompts_coherent_with_dequant_decode \
  -v -s > /tmp/tqprof_step2.log 2>&1
grep -A7 "after 50 calls" /tmp/tqprof_step2.log | head -40 \
  > /workspace/vllm-turboquant/benchmarks/results_tq_dequant_perf/step2_after_compile.md
```

- [ ] **Step 8.2: Compare to Steps 0 and 1**

Assemble the three snapshots side-by-side in `step2_after_compile.md`:

```
Baseline (step 0):         total ~440 ms,  dequant ~91%
After batching (step 1):   total ~???,     dequant ~???
After compile (step 2):    total ~???,     dequant ~???
```

Expected: compile roughly halves the per-call dequant cost on top of batching, so total drops further. Final dequant share should be in the 20-40% range; SDPA share should be proportionally larger (say 20-40%).

- [ ] **Step 8.3: Log `torch.compile` recompile events**

Grep the step2 log for recompile messages:

```bash
grep -iE "recompil|recompile|cache_size|torch.compile cache" /tmp/tqprof_step2.log | head -10
```

If there are more than 1-2 recompilation events during the steady-state decode phase, `dynamic=True` isn't holding. Record in `step2_after_compile.md` and flag as a follow-up (e.g. shape bucketing).

---

## Task 9: End-to-end A/B (optional; large-model signal)

**Files:**
- Create: `benchmarks/results_qwen3_5_35b_dequant_perf/` (populated by the bench script)

This only runs when Steps 1 and 2 both show meaningful wins on the small model. Skip if the tiny-model profile says we're still > 5× bf16 — no point burning H100 time on a large-model bench that won't improve over the current `benchmarks/results_qwen3_5_35b_dequant_fix/` (404 s) beyond the tiny-model signal.

- [ ] **Step 9.1: Launch the bench**

```bash
SNAP=/workspace/hf-cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307 \
META=$PWD/calibration/Qwen_Qwen3.5-35B-A3B_turboquant35_qwen_agent.json \
PROMPTS=$PWD/calibration/prompts/qwen_agent_qwen3_5_35b.jsonl \
RESULT_DIR=$PWD/benchmarks/results_qwen3_5_35b_dequant_perf \
MAX_MODEL_LEN=8192 MAX_NUM_SEQS=2 MAX_CONCURRENCY=2 NUM_PROMPTS=64 OUTPUT_LEN=128 \
GPU_MEMORY_UTILIZATION=0.90 \
TURBOQUANT_RING_CAPACITY=1024 \
bash benchmarks/benchmark_turboquant_vs_baseline_qwen3_5.sh > /tmp/bench_35b_perf.log 2>&1
```

Runs ~20-25 min (two serves × ~5 min init + bench). Run in the background (`&`) if you want to track progress; otherwise wait.

- [ ] **Step 9.2: Summarize**

```bash
cat benchmarks/results_qwen3_5_35b_dequant_perf/comparison.md | head -30
```

Compare key numbers to `benchmarks/results_qwen3_5_35b_dequant_fix/comparison.md`:

- Wall time: was 404 s treatment / 32.4 s baseline.
- Median TTFT: was 4529 ms treatment / 218 ms baseline.
- Mean ITL: was 64.4 ms treatment / 6.2 ms baseline.

Target: wall-time below 100 s, TTFT below 1 s, ITL below 15 ms. Record actuals in a summary note appended to `step2_after_compile.md`.

---

## Task 10: Summary doc

**Files:**
- Modify: `benchmarks/results_tq_dequant_perf/step2_after_compile.md` (append)

- [ ] **Step 10.1: Write a short summary at the bottom**

Add a section summarizing what landed, the before/after numbers, and any follow-ups (e.g. recompile events, `torch.cat` allocator pressure, etc.). Keep it under 300 words. Something like:

```
# Summary

Before:    total 440 ms / 50 calls,  dequant 91%,  wall-time 404 s (12.5x bf16)
After S1:  total ??? ms / 50 calls,  dequant ???,  (micro-measure only)
After S2:  total ??? ms / 50 calls,  dequant ???,  wall-time ??? s (??x bf16)

Quality preserved: long-prompt coherence test still mean_uniq > 0.5,
mean_nl < 0.1 on the 8 glaive prompts that originally collapsed.

Follow-ups:
- [ ] (note any torch.compile recompile events observed)
- [ ] (note any torch.cat pressure observed on packed tensors)
- [ ] (if wall-time still > 3x bf16, reopen design question on Triton
  kernel or KV caching)
```

Fill in the `???`s from the measurement tasks.

---

## Self-Review

**Spec coverage:**

| Spec section | Covered by |
|---|---|
| Step 1 (batch per-seq loop) | Tasks 1, 2, 3, 4 |
| Step 2 (torch.compile) | Tasks 5, 6, 7, 8 |
| Measurement gated between steps | Task 4 (decision), Task 8 |
| `TQ_DEQUANT=python` fallback | Tasks 6.1, 6.3, 7.2, 7.3 |
| Batched-equals-per-seq invariant | Task 1 |
| Long-prompt coherence preserved | Tasks 3.2, 7.1 |
| No regression on existing turboquant suite | Tasks 3.3, 7.4 |
| Optional end-to-end 35B-A3B bench | Task 9 |
| Summary artifact | Task 10 |

**Placeholder scan:** Summary-doc Task 10 intentionally uses `???` as placeholders for numbers the implementer fills in from their measurements — that's data-to-be-collected, not a plan failure. All other steps have concrete code / commands.

**Type consistency:** `_get_compiled_dequant`, `_dequant_fn`, `dequant_k_all`, `dequant_v_all`, `seq_ranges`, `seq_offsets` — used consistently across Tasks 2, 5, 6. Instrumentation keys `"dequant_k"` / `"dequant_v"` / `"ring_overlay"` / `"sdpa"` match the existing profile dictionary.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-04-24-turboquant-dequant-perf.md`.

Two execution options:

1. **Subagent-Driven (recommended).** Fresh subagent per task. Best given the measurement-gated structure — each subagent reads exactly the task block and reports numbers back.
2. **Inline Execution.** Execute tasks in this session; checkpoint after Task 4 (Step 1 measurement) and Task 8 (Step 2 measurement) so the user can decide whether to continue.

Either way: NO commits. Everything stays on `turboquant-dequant-perf` until user OKs.
