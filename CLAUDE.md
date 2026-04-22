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
- **Venv**: `/root/vllm-venv` (do not recreate under `/workspace/.venv`).

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

Validated end-to-end with `Qwen/Qwen2.5-0.5B-Instruct`, `turboquant35`, generated from `tests/prompts/example.txt`. Reference decode test passes for both recipes (`test_turboquant_triton_decode_matches_reference`). Full example flow is in the README under "Running on H100 (RunPod)".

Serve flags that must be present together:

```
--attention-backend TRITON_ATTN
--kv-cache-dtype turboquant{25,35}
--enable-turboquant
--turboquant-metadata-path /path/to/turboquant_kv.json
```

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
- **Calibration**: `AutoConfig.from_pretrained("Qwen/Qwen3.5-122B-A10B")` raises `ValueError: ... model type 'qwen3_5_moe' but Transformers does not recognize this architecture` on transformers 4.57.6. vLLM's *own* loader handles this model (`vllm/model_executor/models/qwen3_5.py` + `registry.py:510`), so `vllm serve` works — but `benchmarks/generate_turboquant_metadata.py` fails because it goes through HF `AutoModel`. **There is no TurboQuant metadata path for this family on this fork without a transformers upgrade**, which would likely break vLLM.
- **Serving RedHatAI NVFP4 on H100 works** (validated): 72 GB weights fit on 80 GB GPU with `--gpu-memory-utilization 0.95 --max-model-len 2048 --max-num-seqs 4 --language-model-only --dtype bfloat16`. Init takes ~280 s. KV cache headroom: ~2.4 GB → 24,576 tokens. No TurboQuant (see above).

### What calibration actually needs

`benchmarks/generate_turboquant_metadata.py` loads the calibration model via HF `AutoModel.from_pretrained(...)`. If `AutoConfig` cannot parse the config, calibration fails before any forward pass. Check `AutoConfig` support **before** downloading a large base model.

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
