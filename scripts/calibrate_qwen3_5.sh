#!/usr/bin/env bash
# Calibrate a TurboQuant KV-cache metadata JSON for Qwen3.5 models.
#
# Qwen3.5 config files declare model_type="qwen3_5_moe" (or similar),
# which transformers < 5 cannot parse via AutoConfig — so vLLM's own
# loader handles serving, but benchmarks/generate_turboquant_metadata.py
# (which goes through HF AutoModel) can't run in the main venv.
#
# This script sets up a *second* venv pinned at transformers>=5 and runs
# the existing calibration script there. The output JSON is portable:
# you can feed it to vllm serve from the regular transformers<5 venv.
#
# Default target is Qwen/Qwen3.5-0.8B (~1.8 GB bf16) — the smallest
# Qwen3.5-family checkpoint and the right first feasibility test.
# Override env vars to scale up; see README "Calibrating TurboQuant for
# Qwen3.5 (two-venv path)" for scale-up guidance on 2B and 122B.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
RECIPE="${RECIPE:-turboquant35}"
REPO_ROOT="${REPO_ROOT:-/workspace/vllm-turboquant}"
CALIB_VENV="${CALIB_VENV:-/root/vllm-venv-calib}"
HF_CACHE="${HF_CACHE:-/workspace/hf-cache}"
OUTPUT_JSON="${OUTPUT_JSON:-$REPO_ROOT/calibration/${MODEL//\//_}_${RECIPE}.json}"
PROMPTS_FILE="${PROMPTS_FILE:-$REPO_ROOT/tests/prompts/example.txt}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MAX_PROMPTS="${MAX_PROMPTS:-128}"

command -v uv >/dev/null 2>&1 || {
    echo "error: 'uv' not found; install via 'curl -LsSf https://astral.sh/uv/install.sh | sh'" >&2
    exit 1
}
[ -f "$PROMPTS_FILE" ] || {
    echo "error: prompts file not found: $PROMPTS_FILE" >&2
    exit 1
}

shopt -s nullglob
wheels=("$REPO_ROOT"/build/release/vllm-*.whl)
shopt -u nullglob
if [ ${#wheels[@]} -eq 0 ]; then
    echo "error: no pre-built wheel in $REPO_ROOT/build/release/." >&2
    echo "       Run a full 'uv pip install -e .' in the main venv first so" >&2
    echo "       a cached editable wheel lands in build/release/." >&2
    exit 1
fi
WHEEL="${wheels[0]}"

mkdir -p "$HF_CACHE" "$(dirname "$OUTPUT_JSON")"

echo "[1/5] Calibration venv: $CALIB_VENV"
if [ ! -d "$CALIB_VENV" ]; then
    uv venv --python 3.12 "$CALIB_VENV"
fi

export VIRTUAL_ENV="$CALIB_VENV"
export UV_LINK_MODE=copy

echo "[2/5] Installing transformers>=5 + vLLM wheel with transformers pin overridden"
OVERRIDES_FILE="$(mktemp -t calib-overrides-XXXX.txt)"
trap 'rm -f "$OVERRIDES_FILE"' EXIT
printf 'transformers>=5.0\n' > "$OVERRIDES_FILE"
uv pip install --upgrade --overrides "$OVERRIDES_FILE" \
    "$WHEEL" \
    torch \
    'transformers>=5.0' \
    accelerate \
    safetensors \
    regex \
    hf-transfer \
    huggingface_hub

echo "[3/5] Downloading $MODEL to $HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1
SNAPSHOT="$("$CALIB_VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id="$MODEL", cache_dir="$HF_CACHE"))
PY
)"
echo "      snapshot: $SNAPSHOT"

if [ -f "$SNAPSHOT/tokenizer_config.json" ] && \
   grep -q '"TokenizersBackend"' "$SNAPSHOT/tokenizer_config.json"; then
    echo "      patching tokenizer_config.json (TokenizersBackend -> Qwen2TokenizerFast)"
    sed -i 's/"TokenizersBackend"/"Qwen2TokenizerFast"/' "$SNAPSHOT/tokenizer_config.json"
fi

echo "[4/5] Running calibration ($MODEL, recipe=$RECIPE, device=$DEVICE, dtype=$DTYPE)"
"$CALIB_VENV/bin/python" "$REPO_ROOT/benchmarks/generate_turboquant_metadata.py" \
    --model "$SNAPSHOT" \
    --kv-cache-dtype "$RECIPE" \
    --prompts-file "$PROMPTS_FILE" \
    --output "$OUTPUT_JSON" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --batch-size "$BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --max-prompts "$MAX_PROMPTS" \
    --trust-remote-code

echo "[5/5] Done. Metadata: $OUTPUT_JSON"
echo
echo "Serve (from the main vLLM venv at /root/vllm-venv) with:"
echo "  vllm serve <model-path> \\"
echo "    --attention-backend TRITON_ATTN \\"
echo "    --kv-cache-dtype $RECIPE \\"
echo "    --enable-turboquant \\"
echo "    --turboquant-metadata-path $OUTPUT_JSON"
