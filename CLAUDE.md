@AGENTS.md

# TurboQuant Fork — Project Notes

Load-bearing context for future sessions on this repo. Update as facts change.

## This machine (RunPod H100)

- **GPU**: 1× NVIDIA H100 80 GB HBM3 (SM90). `nvidia-smi` confirms.
- **CUDA**: 12.8 at `/usr/local/cuda-12.8`.
- **RAM**: ~2 TB (enormous headroom for CPU offload / large models in RAM).
- **Disk quota**: ~200 GB on `/workspace` (MooseFS mount is 378 TB aggregate but user-allocated slice is limited — confirm before large downloads).
- **Local overlay**: `/root` on overlayfs, ~56 GB free. Use this for venv and uv cache, NEVER `/workspace` (see gotchas).
- **Python**: 3.12.13 via `uv`.
- **Venvs** (two; they intentionally have different `transformers` versions — see calibration notes):
  - `/root/vllm-venv` — the vLLM serving venv. `transformers` pinned by vLLM (4.57.x on this fork). Do not recreate under `/workspace/.venv`.
  - `/root/vllm-venv-calib` — the calibration venv. `transformers==5.6.0`, which **does** recognize `qwen3_5_moe`. Used by `benchmarks/generate_turboquant_metadata.py` and `scripts/build_prompts.py`. Keep these separate — upgrading transformers in the serving venv breaks vLLM.

## Key files I have written/modified on this fork

- `vllm/v1/attention/ops/turboquant_kv_cache.py:34` — `TURBOQUANT_SUPPORTED_CUDA_CAPABILITIES` includes `(9, 0)`. H100/SM90 is gated as "experimental".
- `tests/quantization/test_turboquant.py:954` — `test_turboquant_validate_configuration_rejects_unsupported_cuda_device` uses `DeviceCapability(7, 5)` (Turing) as the reject case, since SM90 is now on the allow-list.
- `docs/features/quantization/turboquant_a6000.md` — metadata-generation example uses correct CLI flags (`--model`, `--kv-cache-dtype`, `--prompts-file`). The old `--target-model`/`--recipe` flags never existed in the script.
- `README.md` — "Running on H100 (RunPod)" section with gate change, MooseFS workaround, and `vllm serve` example.

Commits: `12a573a` (SM90 gate + test fix + docs) and `f368100` (README H100 serve example).

## Venv install — fast path (skip ~2 h rebuild)

`uv pip install -e .` rebuilds vllm from source each time (fresh `/tmp/tmp*.build-temp`). To skip:

```bash
uv venv --python 3.12 /root/vllm-venv
VIRTUAL_ENV=/root/vllm-venv UV_LINK_MODE=copy uv pip install \
  /workspace/vllm-turboquant/build/release/vllm-*.whl
```

The `.whl` in `build/release/` is a 12 KB editable shim that points back at the source tree. The actual compiled `.abi3.so` files (~2 GB) live in `vllm/` and are `*.so`-gitignored — they persist across `git pull` but vanish on `git clean -fx`. If the `.so`s are missing, a full source build is the only option.

## Known-working TurboQuant recipe on H100

The **fused Triton decode kernel** (`turboquant_decode_attention_fwd`) produces
degenerate output (newline spam, repetition loops) on prompts longer than
~150 tokens — see `docs/investigations/qwen3_5_turboquant_failure.md` for the
evidence. The fork ships with a **dequantize-then-attend decode path** that
avoids the broken fused kernel. Enable it by setting
`--turboquant-recent-ring-capacity > 0` on `vllm serve`.

Design and implementation notes:
- `docs/superpowers/specs/2026-04-24-turboquant-dequantize-decode-design.md`
- `docs/superpowers/plans/2026-04-24-turboquant-dequantize-decode.md`

Perf follow-up on the dequant path (branch `turboquant-dequant-perf`):
- `docs/superpowers/plans/2026-04-24-turboquant-dequant-perf.md` —
  step 1 batches dequant across sequences, step 2 replaces the
  block-Hadamard FWHT loops in `dequantize_turboquant_vectors` with
  matmuls against precomputed inverse matrices.
- Wall-time on 35B-A3B qwen-agent bench: ~404 s → 348 s (step 1)
  → 301 s (step 2). Details in
  `benchmarks/results_tq_dequant_perf/step2_matmul_rewrite.md`.

Required serve flags for coherent output:

```
--attention-backend TRITON_ATTN
--kv-cache-dtype turboquant{25,35}
--enable-turboquant
--turboquant-metadata-path /path/to/turboquant_kv.json
--turboquant-recent-ring-capacity 1024    # recent-window bf16 ring; see notes
--no-enable-prefix-caching                 # see below — required with the ring
```

**`--no-enable-prefix-caching` is required** when the ring is active. With
prefix caching ON, vLLM reuses KV for shared prompt prefixes and those tokens
never flow through `do_kv_cache_update`, so the ring is missing the very
tokens attention relies on — outputs degenerate.

**Ring capacity tuning.** On this fork's turboquant35 (3.5-bit mixed-group
quantization), per-vector dequant error is ~12-17% per token. That error
accumulates in the attention sum, so beyond ~100 dequantized tokens the
softmax becomes noise-dominated. Set `--turboquant-recent-ring-capacity`
large enough that the ring covers the attention-heavy region of your
expected prompts:

| Context length | Recommended ring | Effective KV compression |
|---:|---:|---:|
| 4K              | 512             | ~88% |
| 16K             | 1024            | ~94% |
| 32K             | 2048            | ~94% |

Above ring size ≥ ~1024 the output quality matches bf16 baseline
(`tests/quantization/test_turboquant_dequant_decode.py` shows
mean-unique-token-ratio 0.84 vs bf16's 0.80 on the 8 glaive-formatted
prompts that previously collapsed to newlines).

## Tests — current state on SM90

`pytest tests/quantization/test_turboquant.py`: **59 passed, 2 failed**.

The 2 failures are `test_turboquant_prefill_reads_quantized_cache[turboquant25/35]`. They are **pre-existing on main and not SM90-specific** — the test perturbs dense K/V after cache-update to force readback, but the post-5fc73a3 dense-prefill fast path in `_forward_turboquant` consumes the passed-in K/V directly. Do not treat these as SM90 regressions; do not try to "fix" them as part of SM90 enablement work.

## Known blockers for specific model families

### Qwen3.5 checkpoints (e.g. `Qwen/Qwen3.5-122B-A10B`, `RedHatAI/Qwen3.5-122B-A10B-NVFP4`)

- **Tokenizer**: their `tokenizer_config.json` declares `"tokenizer_class": "TokenizersBackend"`, which only exists in transformers ≥ 5.0. This fork pins `transformers >= 4.56.0, < 5` (`requirements/common.txt:10`). Workaround: after `snapshot_download`, patch the local copy:
  ```
  sed -i 's/"TokenizersBackend"/"Qwen2TokenizerFast"/' <snapshot>/tokenizer_config.json
  ```
  This is the community-documented fix (see vLLM issues #35998, #38024, #36443).
- **Calibration on the serving venv fails** with `AutoConfig.from_pretrained("Qwen/Qwen3.5-122B-A10B")` raising `ValueError: ... model type 'qwen3_5_moe' but Transformers does not recognize this architecture` on transformers 4.57.x. **But calibration does work from `/root/vllm-venv-calib`** (transformers 5.6.0, verified loads `qwen3_5_moe` fine). The checked-in `calibration/Qwen_Qwen3.5-122B-A10B_turboquant35_toolcalling.json` metadata was produced through this calib venv. Do not upgrade transformers in the serving venv — it would break vLLM.
- **Serving RedHatAI NVFP4 on H100 works** (validated): 72 GB weights fit on 80 GB GPU with `--gpu-memory-utilization 0.95 --max-model-len 2048 --max-num-seqs 4 --language-model-only --dtype bfloat16`. Init takes ~280 s. KV cache headroom: ~2.4 GB → 24,576 tokens. Combine with TurboQuant metadata to enable `--kv-cache-dtype turboquant35`.
- **`MAX_NUM_SEQS=1` triggers a vLLM assertion** on this hybrid-attention/mamba model: `_update_hybrid_attention_mamba_layout` fails with `torch.Size([2, 2, 2096, 2, 256])` ambiguity because `num_blocks=2` collides with a KV dim of 2. Use `MAX_NUM_SEQS >= 2` for bench configs — otherwise the serve process crashes during KV cache init.

### What calibration actually needs

`benchmarks/generate_turboquant_metadata.py` loads the calibration model via HF `AutoModel.from_pretrained(...)`. If `AutoConfig` cannot parse the config, calibration fails before any forward pass. **Always run calibration from `/root/vllm-venv-calib` (transformers 5.6.0), not the serving venv.** Check `AutoConfig` support before downloading a large base model.

## Calibration dataset builder (`calibration/datasets/`)

Adapter-based builder. One adapter class per dataset, registered in `calibration.datasets.ADAPTERS`. Entry point `scripts/build_prompts.py --dataset {glaive,xlam,toolace,bfcl,qwen-agent}` renders any dataset through the target model's chat template (with tools passed via the tokenizer's native `tools=` argument) and writes a JSONL of `{"prompt": str, "text": str}` rows — same file feeds both `generate_turboquant_metadata.py` (reads `text`) and `vllm bench serve --dataset-name custom` (reads `prompt`).

Tests at `tests/calibration/datasets/` (35 fixture-driven, no live HF). Run: `/root/vllm-venv/bin/pytest tests/calibration/datasets`.

`scripts/build_toolcalling_prompts.py` was removed; `scripts/build_prompts.py --dataset glaive` is a strict superset.

## Calibration workflow for Qwen3.5-122B (e.g. on 4xH100)

```bash
# 1. Build prompts (use the qwen3.5-0.8B tokenizer — family shares template).
/root/vllm-venv-calib/bin/python scripts/build_prompts.py \
  --dataset qwen-agent \
  --tokenizer /path/to/Qwen3.5-0.8B/snapshot \
  --output calibration/prompts/qwen_agent_qwen3_5.jsonl \
  --num-prompts 256 --min-tokens 128 --max-tokens 4096

# 2. Calibrate (244 GB bf16 fits on 4x80 GB; CPU offload not needed).
/root/vllm-venv-calib/bin/python benchmarks/generate_turboquant_metadata.py \
  --model Qwen/Qwen3.5-122B-A10B \
  --kv-cache-dtype turboquant35 \
  --prompts-file calibration/prompts/qwen_agent_qwen3_5.jsonl \
  --output calibration/Qwen_Qwen3.5-122B-A10B_turboquant35_qwen_agent.json
```

Workload-matching matters: glaive-calibrated metadata does NOT generalize to agent trajectories — a 2026-04-23 A/B bench on `RedHatAI/Qwen3.5-122B-A10B-NVFP4` with qwen-agent prompts and glaive-calibrated metadata showed **14× wall-time regression, TTFT from 889 ms → 22,837 ms, and only 15.6% exact-match** vs bf16 KV. If you care about agentic tool-calling at serve time, calibrate on an agent dataset like `qwen-agent`, not on glaive. Raw results live in `benchmarks/results_qwen_agent/` (untracked).

## Gotchas

- **MooseFS + uv pip install = ESTALE.** `/workspace` is on MooseFS/FUSE, which returns `ESTALE` under high file churn (e.g., `uv pip install` extracting wheels). Always create the venv on local overlay (`/root/vllm-venv`). Source tree on `/workspace` is fine because reads are stable.
- **`uv pip install` on an editable install rebuilds from scratch.** Each invocation makes a fresh `/tmp/tmp*.build-temp`, so ninja has no incremental state. Either install the pre-built wheel from `build/release/`, or accept ~2 h compile. Do NOT use `TORCH_CUDA_ARCH_LIST=9.0` expecting it to narrow compile — it is not honored by vLLM's CMake.
- **`git clean -fx`** will delete the `.abi3.so` files in `vllm/` and force a full rebuild. Avoid it unless you have `build/release/` backing you up.
- **Run `vllm serve` with the snapshot path**, not the top-level HF cache dir — e.g. `/workspace/hf-cache/models--X--Y/snapshots/<sha>/`, not `.../models--X--Y/`.
- **NVFP4 footprint is larger than name suggests.** `RedHatAI/Qwen3.5-122B-A10B-NVFP4`: the card advertises "71B" but the actual checkpoint is 72 GB because LM head, embeddings, MLP gates, shared expert gates, linear attention layers stay unquantized (bf16/fp8).
- **Commit style (see AGENTS.md)**: DO NOT add `Co-Authored-By: Claude` trailers (user preference). Keep the AI-assistance acknowledgment in the body prose instead.

## Build artifacts

- `build/release/vllm-*.whl` — cached editable wheel from the successful pass-1 build. 12 KB. Tied to commit `c6b2ee90d`, works for later Python-only changes (SM90 gate, tests, docs) since those do not require recompilation.
- `build/` is gitignored.
