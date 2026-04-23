# TurboQuant on Qwen3.5 (H100, two-venv calibration + NVFP4 serve)

End-to-end flow for running TurboQuant KV-cache quantization on the
Qwen3.5 model family on a single NVIDIA H100 (SM90). Validated against
[RedHatAI/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-NVFP4),
calibrated on the unquantized base
[Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
with workload-matched tool-calling prompts.

For the generic A6000 / CUDA 12.8 bring-up path, see
[turboquant_a6000.md](./turboquant_a6000.md). Qwen3.5 has extra
constraints — read on.

## Why Qwen3.5 needs its own path

Qwen3.5 ships with `model_type="qwen3_5_moe"` (or similar) in
`config.json`. Transformers < 5 cannot parse that, so HF
`AutoConfig.from_pretrained(...)` raises before any forward pass runs.

vLLM has its own loader for the architecture, so `vllm serve` works on
transformers 4.57 — but `benchmarks/generate_turboquant_metadata.py`
uses HF `AutoConfig` + `AutoModel.from_pretrained` under the hood, and
cannot run against Qwen3.5 in the main vLLM venv (which pins
`transformers >=4.56, <5`; upgrading in-place breaks vLLM internals).

The workaround is a second venv used only for calibration, pinned at
`transformers>=5`. The metadata JSON it produces is portable and
loads cleanly into vLLM running against transformers 4.57.

## Validated configuration

- **GPU**: 1× NVIDIA H100 80 GB HBM3 (SM90)
- **Base image**:
  `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
  — the `-devel` variant ships `nvcc` at
  `/usr/local/cuda-12.8/bin/nvcc` and system `gcc`/`g++`, both
  required for the Qwen3.5 serve path's first-request JIT compile.
- **Python**: uv-managed CPython 3.12 (host `python3.11` is ignored)
- **Disk**: ~320 GB needed for a 122B run (244 GB bf16 base for
  calibration + 72 GB NVFP4 for serving). Validated on a 600 GB
  `/workspace` quota.
- **RAM**: ~2 TB host RAM (accelerate offload target for the 244 GB
  bf16 base during calibration).

## Step 1 — main venv (used by `vllm serve`)

Built against the fork's pinned `transformers<5`. The fork's
pre-cached editable wheel at `build/release/vllm-*.whl` is a thin
shim that binds to the source tree.

```bash
uv venv --python 3.12 /root/vllm-venv
VIRTUAL_ENV=/root/vllm-venv UV_LINK_MODE=copy uv pip install \
  /workspace/vllm-turboquant/build/release/vllm-*.whl

# ninja is needed at *serve time* — flashinfer JIT-compiles the
# gdn_prefill_sm90 module on the first `/v1/completions` request.
VIRTUAL_ENV=/root/vllm-venv UV_LINK_MODE=copy uv pip install ninja
```

Without ninja the server comes up healthy, but the first request
500s with `FileNotFoundError: 'ninja'` in the EngineCore log. This is
a Qwen3.5 hybrid-attention requirement, not a TurboQuant thing.

`/root` (overlay) is the correct home for both venvs — MooseFS
`/workspace` returns `ESTALE` under the file churn of
`uv pip install`. Source tree on `/workspace` is fine because reads
are stable.

## Step 2 — calibration venv (used by the metadata generator)

Created on first run by `scripts/calibrate_qwen3_5.sh` at
`/root/vllm-venv-calib`. The script uses `uv --overrides` to install
the vLLM wheel with all of its deps *except* the transformers pin,
which is forced to `>=5`:

```
printf 'transformers>=5.0\n' > overrides.txt
uv pip install --overrides overrides.txt \
    "$WHEEL" torch 'transformers>=5.0' accelerate safetensors regex \
    hf-transfer huggingface_hub
```

This lets the same wheel (and its pre-built `.abi3.so` files) work
against transformers 5 for the duration of calibration. The wheel
isn't redistributed from this venv — only the JSON it produces is.

## Step 3 — workload-matched calibration prompts

The JSON that `generate_turboquant_metadata.py` writes is a
per-layer, per-KV-head ranking of the top-K channels by activation
energy (sum of squares). TurboQuant keeps those channels at high
precision and quantizes the rest aggressively. The ranking is only as
good as the activations that go into it — channels that never fire
during calibration get ranked as unimportant and quantized hard, even
if they dominate a different workload at serve time.

`tests/prompts/example.txt` is a plumbing fixture — 8 short lines,
~150 observed tokens. Enough to prove the pipeline runs; **not** a
calibration you should deploy against.

For a tool-calling workload the relevant activations are dominated by:

- chat-template boundary tokens (`<|im_start|>`, `<|im_end|>`, role
  strings, Qwen3.5's `<think>` blocks);
- long structured system prompts carrying function schemas (nested
  JSON);
- JSON punctuation in arguments (`{`, `}`, `[`, `]`, `"`, `:`) and
  `<tool_call>` / `<functioncall>` markers;
- `tool` / `FUNCTION RESPONSE` turns carrying API payloads.

Raw text calibration (WikiText etc.) misses every one.

`scripts/build_prompts.py --dataset glaive` streams
`glaiveai/glaive-function-calling-v2` (Apache-2.0, ungated, 113K
rows), parses each record's `USER` / `ASSISTANT` / `FUNCTION
RESPONSE` markers into Qwen roles (`user` / `assistant` / `tool`),
and renders through the target model's chat template. Output is
JSONL — one `{"prompt": "<rendered>", "text": "<rendered>"}` per
line. `benchmarks/generate_turboquant_metadata.py` reads `text`;
`vllm bench serve --dataset-name custom` reads `prompt`; the same
file works for both.

Other tool-calling sources (`xlam`, `toolace`, `bfcl`) are registered
in `calibration/datasets/` — swap via `--dataset <name>`.

```bash
SNAP_08=/workspace/hf-cache/models--Qwen--Qwen3.5-0.8B/snapshots/<sha>
/root/vllm-venv-calib/bin/python scripts/build_prompts.py \
  --dataset glaive \
  --tokenizer "$SNAP_08" \
  --output calibration/prompts/toolcalling_qwen3_5.jsonl \
  --num-prompts 128 --max-tokens 512
```

The Qwen3.5 family shares one chat template, so the 0.8B tokenizer
works for building prompts that will calibrate 122B.

## Step 4 — calibration on 1xH100 + CPU offload

`Qwen/Qwen3.5-122B-A10B` in bf16 is ~244 GB, far past one H100.
`scripts/calibrate_qwen3_5.sh` honors accelerate's `device_map`
machinery for this case:

```bash
MODEL=Qwen/Qwen3.5-122B-A10B \
DEVICE_MAP=auto MAX_MEMORY_PER_GPU=70GiB MAX_MEMORY_CPU=1500GiB \
BATCH_SIZE=1 MAX_SEQ_LEN=512 MAX_PROMPTS=128 \
PROMPTS_FILE=calibration/prompts/toolcalling_qwen3_5.jsonl \
OUTPUT_JSON=calibration/Qwen_Qwen3.5-122B-A10B_turboquant35_toolcalling.json \
  scripts/calibrate_qwen3_5.sh
```

`device_map=auto` fills the H100 to 70 GiB (hot path: embeddings,
router, attention layers, a subset of experts) and pages the
remaining ~175 GB through host RAM on-demand. The script drops the
`model.to()` call when a device map is set and places inputs on
whichever device the input embeddings landed on.

Observed runtime on this box: **~1h 45m** for 128 chat-rendered
prompts × 512 tokens, against a pre-warmed page cache. The first
forward is slow (cold expert paging); subsequent forwards speed up as
the OS page cache warms (~1.1 TB resident at steady state).

Sanity-check the metadata before serving:

```
$ python3 -c "
import json
m = json.load(open('.../toolcalling.json'))
print('layers:', len(m['layers']))
print('observed_tokens:', m['calibration']['num_observed_tokens'])
print('num_prompts:', m['calibration']['num_prompts'])
"
layers: 12
observed_tokens: 54100
num_prompts: 128
```

`observed_tokens` is the single best quality signal. 128
chat-rendered prompts × 512 tokens yields 54,100 observed tokens —
~350× more than the toy fixture.

## Step 5 — serve the NVFP4 checkpoint with TurboQuant

NVFP4 weights are ~72 GB and fit one H100 with room for KV cache and
CUDA graph pool. The TurboQuant metadata JSON, produced from the
bf16 base's activations, loads cleanly against the NVFP4 variant:

- The layer-name alias machinery in `TurboQuantMetadata.get_layer`
  resolves the `language_model.` prefix that Qwen3.5's
  conditional-generation wrapper adds at runtime.
- NVFP4 dequantizes to near-bf16 at runtime, so the channel ranking
  extracted from bf16 activations still tracks the NVFP4 activations
  closely enough for top-K selection.

```bash
SNAP=/workspace/hf-cache/models--RedHatAI--Qwen3.5-122B-A10B-NVFP4/snapshots/<sha>
META=/workspace/vllm-turboquant/calibration/Qwen_Qwen3.5-122B-A10B_turboquant35_toolcalling.json

# NVFP4 tokenizer also declares TokenizersBackend; patch before first load:
sed -i 's/"TokenizersBackend"/"Qwen2TokenizerFast"/' "$SNAP/tokenizer_config.json"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/root/vllm-venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"

CUDA_VISIBLE_DEVICES=0 /root/vllm-venv/bin/vllm serve "$SNAP" \
  --tensor-parallel-size 1 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype turboquant35 \
  --enable-turboquant \
  --turboquant-metadata-path "$META" \
  --max-model-len 2048 --max-num-seqs 4 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16 --language-model-only \
  --port 8000
```

Observed startup on this box: init engine (profile + KV cache + CUDA
graph capture) ~112 s. KV cache size: 24,576 tokens → ~12×
concurrency at the 2048-token hard cap (`--max-num-seqs 4` caps the
scheduler). First request after a warm flashinfer JIT cache came
back in ~5 s; a fully-cold venv budgets ~3 min for the first request
as ninja compiles `gdn_prefill_sm90`.

## Observed behaviour

Plain completion:

```
prompt: "The H100 GPU is"
reply:  " a high-performance graphics processing unit designed for
         advanced image rendering and data visualization..."
usage:  prompt=7, completion=40, wall=5.15 s
```

Chat completion with an inline tool schema (no
`--enable-auto-tool-choice` yet):

```
system: "You are a helpful assistant that can call functions.
         Available: get_weather(location: str). When you want to call
         it, emit <tool_call>{...}</tool_call>."
user:   "Is it raining in San Francisco?"
reply:
  <think>
  Okay, the user is asking about the weather in San Francisco. I need
  to call the get_weather function with the location set to
  "San Francisco". I'll emit the function call in the specified format.
  </think>
  <tool_call>{"name":"get_weather","arguments":{"location":"San Francisco"}}</tool_call>
usage:  prompt=64, completion=60, wall=2.05 s  (~30 tok/s)
```

~30 tok/s steady-state decode on a 122B-A10B MoE (~10B active) with
NVFP4 weights + turboquant35 KV is consistent with the active-param
compute envelope of one H100.

## Gotchas

- **Never upgrade transformers in the main venv.** The pin is load-
  bearing; vLLM internals depend on 4.x layouts. Keep the second
  calibration venv for anything that wants transformers 5.
- **ninja + nvcc required at serve time** for the Qwen3.5 hybrid-
  attention path (flashinfer `gdn_prefill_sm90` JIT on first request).
  Install ninja in the main venv and set `CUDA_HOME` /`PATH`.
- **Tokenizer patch.** Both the bf16 base and the NVFP4 checkpoint
  declare `"tokenizer_class": "TokenizersBackend"` (a transformers 5
  concept). On 4.57 replace with `"Qwen2TokenizerFast"` after
  `snapshot_download` — the fix is community-documented
  (vLLM issues #35998, #38024, #36443).
- **Calibration-base vs serve-checkpoint mismatch** is allowed. The
  JSON carries `model_name` for provenance only; vLLM does not
  cross-check it against the served model at load time. The two must
  share the same architecture (same `head_dim`, same
  `num_key_value_heads`, same `layer_types` pattern) — in practice,
  bf16 base and NVFP4 variant of the same model are safe.
- **Accelerate "meta device" warning** during calibration
  (`Some parameters are on the meta device because they were offloaded
  to the cpu.`) is about cold experts that never fire for the
  calibration prompts. Attention projections — the only thing the
  calibration hooks touch — are dense and always resident.
- **`tool_calls` in the OpenAI response** requires
  `--enable-auto-tool-choice --tool-call-parser <name>` on the serve
  command. Without it, put the tool schema in the system message and
  parse the model's raw `<tool_call>{...}</tool_call>` text yourself.
- **`build/release/vllm-*.whl` is tied to a specific commit** (a 12 KB
  editable shim that points at the source tree, plus ~2 GB of
  `.abi3.so` files in `vllm/`). It survives `git pull` but not
  `git clean -fx` — rebuild from source if the compiled `.so` files
  are missing.

## Files in this fork that implement this flow

- `scripts/calibrate_qwen3_5.sh` — calibration orchestration (venv +
  download + tokenizer patch + metadata generation).
- `scripts/build_prompts.py` + `calibration/datasets/` — adapter
  registry + CLI that renders any of {glaive, xlam, ToolACE, BFCL}
  into a chat-templated JSONL usable for calibration and
  `vllm bench serve`.
- `benchmarks/generate_turboquant_metadata.py` — the calibration
  entry-point. `--device-map` / `--max-memory-per-gpu` /
  `--max-memory-cpu` handle the 1xH100 offload case; the `.jsonl`
  prompt-file extension is detected automatically.
- `vllm/v1/attention/ops/turboquant_kv_cache.py` —
  `TURBOQUANT_SUPPORTED_CUDA_CAPABILITIES` includes `(9, 0)`.
- `vllm/v1/attention/ops/turboquant_metadata.py` —
  `_turboquant_layer_name_candidates` resolves the `language_model.`
  prefix aliasing between calibration JSON and runtime layer names.
