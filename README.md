<!-- markdownlint-disable MD001 MD041 -->

# vllm-turboquant — Qwen3.5-122B-A10B NVFP4 with TurboQuant KV cache on 1xH100

This fork's headline result: serve `RedHatAI/Qwen3.5-122B-A10B-NVFP4`
with `turboquant35` mixed-precision KV cache and a workload-matched
tool-calling calibration on a single NVIDIA H100 (SM90). Metadata was
generated from the unquantized bf16 base via the two-venv
calibration flow and is checked in at
[`calibration/Qwen_Qwen3.5-122B-A10B_turboquant35_toolcalling.json`](calibration/Qwen_Qwen3.5-122B-A10B_turboquant35_toolcalling.json)
(54,100 observed tokens from 128 glaive-derived chat-templated
prompts). One-shot serve command:

```bash
SNAP=/workspace/hf-cache/models--RedHatAI--Qwen3.5-122B-A10B-NVFP4/snapshots/49d19c108259a21450c40b8af38828b0a97390d8
META=/workspace/vllm-turboquant/calibration/Qwen_Qwen3.5-122B-A10B_turboquant35_toolcalling.json
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

Observed on the reference box (RunPod
`pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` image):
~112 s engine init, ~5 s first `/v1/completions` round-trip once
`ninja`'s flashinfer cache is warm, ~30 tok/s steady-state decode.
Full procedure (venv bootstrap, calibration, serve, and gotchas)
is in
[**`docs/features/quantization/turboquant_qwen3_5.md`**](docs/features/quantization/turboquant_qwen3_5.md).

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## TurboQuant Fork Highlights

This fork extends vLLM's experimental TurboQuant KV-cache path with a workflow
aimed at supported CUDA workstation GPUs:

- TurboQuant KV cache on `RTX A6000 / SM86`, `H100 / SM90` (experimental), and `GB10 / SM121`
- `turboquant25` and `turboquant35` KV-cache recipes on the Triton attention backend
- Per-layer TurboQuant metadata loading from `--turboquant-metadata-path` or a local model-side `turboquant_kv.json`
- Tensor-parallel metadata slicing for replicated and partitioned KV-head layouts
- Kernel tuning for supported CUDA targets and a Triton prefill fast path for common head sizes
- Benchmark and bring-up docs for long-context TurboQuant comparisons and 4x A6000 serving

Start here for the fork-specific docs:

- [Quantized KV Cache docs](docs/features/quantization/quantized_kvcache.md)
- [TurboQuant on RTX A6000 and CUDA 12.8](docs/features/quantization/turboquant_a6000.md)
- [TurboQuant on Qwen3.5 (H100, two-venv calibration + NVFP4 serve)](docs/features/quantization/turboquant_qwen3_5.md)
- [TurboQuant comparison benchmark](benchmarks/run_turboquant_gb10_compare.sh)

### Running on H100 (RunPod)

Tested on the `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
image — the `-devel` variant ships `nvcc` at `/usr/local/cuda-12.8/bin/nvcc`
and system `gcc`/`g++`, which the Qwen3.5 serve path requires for its
first-request JIT build (see the calibration prerequisites below).
Host `python3.11` is ignored in favor of a uv-managed 3.12 venv.

Experimental SM90 support is gated in
`vllm/v1/attention/ops/turboquant_kv_cache.py` (the `(9, 0)` entry in
`TURBOQUANT_SUPPORTED_CUDA_CAPABILITIES`). The TurboQuant Triton kernels
contain no SM86-specific intrinsics, so the generic tuning fallback in
`get_turboquant_kernel_meta` is used on H100. The reference decode test
(`test_turboquant_triton_decode_matches_reference`) passes on SM90 for
both `turboquant25` and `turboquant35`, but kernel tuning has not been
benchmarked for H100.

RunPod-specific gotcha: `/workspace` is mounted via MooseFS/FUSE, which
can return `ESTALE` ("Stale file handle") under the high file churn of
`uv pip install`. Create the venv on a local overlay filesystem instead:

```bash
uv venv --python 3.12 /root/vllm-venv
VIRTUAL_ENV=/root/vllm-venv UV_LINK_MODE=copy uv pip install -e .
```

The source tree can stay on `/workspace`; only the venv needs to be on
local disk.

#### Serving a small model on H100

With the venv built, generate per-layer TurboQuant metadata against the
target model and start `vllm serve` with the TurboQuant flags. This
example uses `Qwen/Qwen2.5-0.5B-Instruct` — substitute any unquantized
model with standard `model.layers.*.self_attn.{k,v}_proj` layout:

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct

# One-time calibration — writes turboquant_kv.json
/root/vllm-venv/bin/python benchmarks/generate_turboquant_metadata.py \
  --model "$MODEL" \
  --kv-cache-dtype turboquant35 \
  --prompts-file tests/prompts/example.txt \
  --output /root/turboquant_kv.json

# Serve
CUDA_VISIBLE_DEVICES=0 /root/vllm-venv/bin/vllm serve "$MODEL" \
  --tensor-parallel-size 1 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype turboquant35 \
  --enable-turboquant \
  --turboquant-metadata-path /root/turboquant_kv.json \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.5 \
  --port 8000
```

Smoke-test once `/health` returns 200:

```bash
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","prompt":"vLLM is","max_tokens":32,"temperature":0}'
```

#### Calibrating TurboQuant for Qwen3.5 (two-venv path)

Full procedure with rationale, observed timings, and gotchas:
**[docs/features/quantization/turboquant_qwen3_5.md](docs/features/quantization/turboquant_qwen3_5.md)**.
The summary below is enough to get started.

Qwen3.5 checkpoints declare `model_type="qwen3_5_moe"` (or similar), which
`transformers < 5` cannot parse via `AutoConfig`. vLLM's own loader handles
serving, but `benchmarks/generate_turboquant_metadata.py` goes through HF
`AutoModel.from_pretrained` and fails before any forward pass — see the
notes in `CLAUDE.md`.

This fork pins `transformers >=4.56,<5` because upgrading in-place breaks
vLLM. The workaround is a *second* venv used only for calibration, pinned
at `transformers>=5`. The emitted metadata JSON is just per-layer channel
indices — it's portable and loads cleanly into vLLM running against
`transformers<5`.

The `scripts/calibrate_qwen3_5.sh` helper orchestrates the procedure.
Start with the smallest Qwen3.5 checkpoint as a feasibility check — if
the pipeline produces a valid JSON on a 0.8B model and that JSON serves
correctly, scaling up to 2B and then 122B is just a matter of memory:

```bash
# Smallest Qwen3.5 checkpoint, ~1.8 GB bf16, fits easily on one H100.
# Calibration takes a few minutes. Writes to calibration/Qwen_Qwen3.5-0.8B_turboquant35.json
MODEL=Qwen/Qwen3.5-0.8B RECIPE=turboquant35 \
  scripts/calibrate_qwen3_5.sh

# Serve from the main venv (transformers<5) against the calibrated JSON.
# Qwen3.5 serve path JIT-compiles flashinfer's gdn_prefill SM90 module on
# first request, so ninja + nvcc must be on PATH for the main venv:
VIRTUAL_ENV=/root/vllm-venv UV_LINK_MODE=copy uv pip install ninja
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/root/vllm-venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
/root/vllm-venv/bin/vllm serve \
  /workspace/hf-cache/models--Qwen--Qwen3.5-0.8B/snapshots/<sha>/ \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype turboquant35 \
  --enable-turboquant \
  --turboquant-metadata-path \
    calibration/Qwen_Qwen3.5-0.8B_turboquant35.json \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.5
```

The JIT build of the SM90 gdn_prefill module takes ~3 min on first startup
and is cached in `~/.cache/flashinfer` for subsequent runs. Without ninja
or nvcc on PATH, the server comes up healthy but 500s on the first
`/v1/completions` request with `FileNotFoundError: 'ninja'` in the engine
log — this is not a TurboQuant issue, it's a Qwen3.5 hybrid-attention
requirement.

Scaling up:

| Model                        | bf16 size | Disk needed | Calibration hardware              | Expected runtime |
| ---------------------------- | --------- | ----------- | --------------------------------- | ---------------- |
| `Qwen/Qwen3.5-0.8B`          | ~1.8 GB   | ~5 GB       | `DEVICE=cuda`                     | minutes          |
| `Qwen/Qwen3.5-2B`            | ~4 GB     | ~10 GB      | `DEVICE=cuda`                     | minutes          |
| `Qwen/Qwen3.5-122B-A10B`     | ~244 GB   | ~260 GB     | `DEVICE=cpu` (needs ~250 GB RAM)  | many hours       |

Override with env vars: `MODEL=Qwen/Qwen3.5-2B scripts/calibrate_qwen3_5.sh`,
or `MODEL=Qwen/Qwen3.5-122B-A10B DEVICE=cpu BATCH_SIZE=1 scripts/calibrate_qwen3_5.sh`.

Caveats — read before running at 122B scale:

- **HF module-naming assumption.** The calibration script discovers
  `layers.{i}.self_attn.(k_proj|v_proj)` via regex. If transformers 5.x's
  Qwen3.5 implementation uses a fused QKV layout (as vLLM does — see
  `Qwen3NextAttention.qkv_proj` in `vllm/model_executor/models/qwen3_next.py`),
  the discovery step raises a clear error. Validate on 0.8B first.
- **Serving an NVFP4 checkpoint with TurboQuant is not yet validated on
  this fork.** The metadata produced from the bf16 base is format-portable,
  but the combination `--kv-cache-dtype turboquant35 --enable-turboquant`
  plus NVFP4 weights has not been exercised end-to-end here. Serving the
  NVFP4 checkpoint *without* TurboQuant is validated — see `CLAUDE.md`.
- **Disk.** At 122B, `/workspace` (200 GB quota) needs to be clear of
  the NVFP4 snapshot and any other large caches before the bf16 base
  will fit. Either free space first or target a larger filesystem by
  overriding `HF_CACHE`.
- **CPU calibration is slow at 122B scale.** Budget many hours for
  128 prompts × 2048 tokens on a 122B MoE on CPU. For 1xH100 + CPU
  offload via accelerate, pass `DEVICE_MAP=auto MAX_MEMORY_PER_GPU=70GiB
  MAX_MEMORY_CPU=1500GiB` to the script (see next section).

##### Picking calibration prompts that match your workload

The JSON that `generate_turboquant_metadata.py` writes is a per-layer,
per-KV-head list of "outlier" channel indices — the top-K channels with
the highest activation energy (sum-of-squares) across the calibration
set. TurboQuant keeps those channels at high precision and quantizes the
rest. The ranking is only as good as the activations that go into it:
channels that never fire during calibration get ranked as unimportant
and will be quantized aggressively at serve time, even if they
dominate a different workload.

In practice this means generic text (e.g. WikiText) is a poor calibration
set for a model that will serve **tool calling**. Tool-call KV is
dominated by:

- chat-template boundary tokens (`<|im_start|>`, `<|im_end|>`, role
  strings, Qwen3.5's `<think>` blocks) — these fire on every turn;
- long structured system prompts with function schemas (nested JSON);
- JSON punctuation in tool arguments (`{`, `}`, `[`, `]`, `"`, `:`) and
  `<tool_call>` / `<functioncall>` markers;
- `tool` / `FUNCTION RESPONSE` turns that carry API payloads.

Calibrating on raw text misses all of these channel patterns.

`scripts/build_toolcalling_prompts.py` produces a calibration-ready
JSONL from `glaiveai/glaive-function-calling-v2` (Apache-2.0, ungated,
113K examples). For each record it parses the multi-turn chat, maps
glaive's `USER` / `ASSISTANT` / `FUNCTION RESPONSE` markers to roles
(`user` / `assistant` / `tool`), and re-renders through the target
model's chat template so the calibration sees the exact token layout
production inference will produce. Output is one
`{"text": "<rendered prompt>"}` object per line.

`benchmarks/generate_turboquant_metadata.py` accepts `.jsonl` prompt
files natively — no format flag needed. `.txt` is unchanged:
one-prompt-per-line.

Typical flow on 1xH100 (tool calling on Qwen3.5-122B-A10B):

```bash
# 1. Build tool-calling prompts through Qwen3.5's chat template.
#    Tokenizer from any Qwen3.5 family member works — they share a template.
SNAPSHOT=/workspace/hf-cache/models--Qwen--Qwen3.5-0.8B/snapshots/<sha>
/root/vllm-venv-calib/bin/python scripts/build_toolcalling_prompts.py \
  --tokenizer "$SNAPSHOT" \
  --output calibration/prompts/toolcalling_qwen3_5.jsonl \
  --num-prompts 128 --max-tokens 512

# 2. Calibrate 122B on 1xH100 + CPU offload against the tool-calling set.
#    --device-map auto skips the .to() path and dispatches via accelerate;
#    max-memory keeps ~70 GiB on GPU and offloads the rest.
MODEL=Qwen/Qwen3.5-122B-A10B \
DEVICE_MAP=auto MAX_MEMORY_PER_GPU=70GiB MAX_MEMORY_CPU=1500GiB \
BATCH_SIZE=1 MAX_SEQ_LEN=512 MAX_PROMPTS=128 \
PROMPTS_FILE=calibration/prompts/toolcalling_qwen3_5.jsonl \
OUTPUT_JSON=calibration/Qwen_Qwen3.5-122B-A10B_turboquant35_toolcalling.json \
  scripts/calibrate_qwen3_5.sh
```

Sanity-check the resulting JSON before serving with it:

```bash
python3 -c "
import json
m = json.load(open('calibration/Qwen_Qwen3.5-122B-A10B_turboquant35_toolcalling.json'))
print('layers:', len(m['layers']))
print('observed_tokens:', m['calibration']['num_observed_tokens'])
print('num_prompts:', m['calibration']['num_prompts'])
"
```

`observed_tokens` is the single best quality signal. The stub
`tests/prompts/example.txt` produces ~150 observed tokens across 8
short prompts — that is a plumbing test, not a calibration. 128
chat-rendered prompts × 512 tokens yields tens of thousands of
observed tokens, enough to stabilize the per-head channel ranking
for a 48-layer / 2-KV-head / 256-head-size model.

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

For TurboQuant on this fork, use a source build instead of precompiled wheels:

```bash
uv venv --python 3.12
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_PRECOMPILED=0
export VLLM_MAIN_CUDA_VERSION=12.8

uv pip install -e .
```

Example TurboQuant serve command:

```bash
.venv/bin/vllm serve /models/target \
  --tensor-parallel-size 4 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype turboquant35 \
  --enable-turboquant \
  --turboquant-metadata-path /models/target/turboquant_kv.json
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
