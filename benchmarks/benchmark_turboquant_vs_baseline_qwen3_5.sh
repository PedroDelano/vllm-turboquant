#!/usr/bin/env bash
# Benchmark turboquant35 against the bf16 KV-cache baseline on
# RedHatAI/Qwen3.5-122B-A10B-NVFP4 (single H100 / SM90).
#
# For each arm:
#   1. Launch `vllm serve` with the arm-specific KV flags.
#   2. Wait for /health = 200.
#   3. Run `vllm bench serve` against the shared tool-calling prompts
#      JSONL with --save-detailed so per-request TTFT/ITL/generated
#      text is retained.
#   4. Kill the server.
#
# The compare script `scripts/compare_bench_results.py` consumes both
# result JSONs and emits a markdown comparison table + a quality diff
# (exact-match rate and token overlap between arms on the same prompts).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/vllm-turboquant}"
SNAP="${SNAP:-/workspace/hf-cache/models--RedHatAI--Qwen3.5-122B-A10B-NVFP4/snapshots/49d19c108259a21450c40b8af38828b0a97390d8}"
META="${META:-$REPO_ROOT/calibration/Qwen_Qwen3.5-122B-A10B_turboquant35_toolcalling.json}"
PROMPTS="${PROMPTS:-$REPO_ROOT/calibration/prompts/toolcalling_qwen3_5.jsonl}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/benchmarks/results}"

VENV_BIN="${VENV_BIN:-/root/vllm-venv/bin}"
PORT="${PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

NUM_PROMPTS="${NUM_PROMPTS:-64}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
REQUEST_RATE="${REQUEST_RATE:-inf}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$VENV_BIN:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

mkdir -p "$RESULT_DIR"

if [ ! -d "$SNAP" ]; then
    echo "error: SNAP not found: $SNAP" >&2
    exit 1
fi
if [ ! -f "$META" ]; then
    echo "error: META not found: $META" >&2
    exit 1
fi
if [ ! -f "$PROMPTS" ]; then
    echo "error: PROMPTS not found: $PROMPTS" >&2
    echo "      generate with: scripts/build_prompts.py --dataset glaive" >&2
    exit 1
fi

# --- helpers --------------------------------------------------------------

_launch_serve() {
    # $1: label (baseline | turboquant35)
    # $2..: extra vllm serve args specific to the arm
    local label="$1"
    shift
    local logfile="$RESULT_DIR/serve_${label}.log"
    echo "[serve:$label] launching (log: $logfile)"
    "$VENV_BIN/vllm" serve "$SNAP" \
        --tensor-parallel-size 1 \
        --attention-backend TRITON_ATTN \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --dtype bfloat16 \
        --language-model-only \
        --port "$PORT" \
        "$@" \
        >"$logfile" 2>&1 &
    SERVE_PID=$!
    echo "[serve:$label] pid=$SERVE_PID"
}

_wait_for_health() {
    local deadline=$((SECONDS + 900))  # 15 min
    while (( SECONDS < deadline )); do
        if curl -sfo /dev/null "http://localhost:$PORT/health" 2>/dev/null; then
            echo "[serve] /health 200 after $SECONDS seconds"
            return 0
        fi
        if ! kill -0 "$SERVE_PID" 2>/dev/null; then
            echo "error: serve process exited before /health came up" >&2
            tail -40 "$RESULT_DIR"/serve_*.log >&2
            return 1
        fi
        sleep 2
    done
    echo "error: /health never came up within 15 minutes" >&2
    return 1
}

_warmup_first_request() {
    # Qwen3.5 JIT-compiles flashinfer gdn_prefill_sm90 on the first
    # request. Burn that cost once before the benchmark clock starts.
    echo "[warmup] sending one /v1/completions to trigger JIT compile"
    curl -sS --max-time 600 "http://localhost:$PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SNAP\",\"prompt\":\"warmup\",\"max_tokens\":4,\"temperature\":0}" \
        >/dev/null
    echo "[warmup] done"
}

_run_bench() {
    local label="$1"
    local result_file="$RESULT_DIR/bench_${label}.json"
    local trace_file="$RESULT_DIR/trace_${label}.txt"
    echo "[bench:$label] running (num-prompts=$NUM_PROMPTS, output-len=$OUTPUT_LEN, max-concurrency=$MAX_CONCURRENCY)"
    "$VENV_BIN/vllm" bench serve \
        --backend vllm \
        --host localhost --port "$PORT" \
        --endpoint /v1/completions \
        --model "$SNAP" \
        --dataset-name custom \
        --dataset-path "$PROMPTS" \
        --num-prompts "$NUM_PROMPTS" \
        --custom-output-len "$OUTPUT_LEN" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --request-rate "$REQUEST_RATE" \
        --ignore-eos \
        --skip-chat-template \
        --save-result --save-detailed \
        --result-dir "$RESULT_DIR" \
        --result-filename "bench_${label}.json" \
        --seed 0 \
        2>&1 | tee "$trace_file"
    echo "[bench:$label] saved $result_file"
}

_kill_serve() {
    if [ -n "${SERVE_PID:-}" ] && kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[serve] stopping pid=$SERVE_PID"
        kill -TERM "$SERVE_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$SERVE_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SERVE_PID" 2>/dev/null || true
        wait "$SERVE_PID" 2>/dev/null || true
    fi
    unset SERVE_PID
    # give CUDA context a moment to clear between arms
    sleep 5
}

trap _kill_serve EXIT

# --- arm A: bf16 KV (baseline, no TurboQuant) -----------------------------

_launch_serve "baseline" \
    --kv-cache-dtype auto
_wait_for_health
_warmup_first_request
_run_bench "baseline"
_kill_serve

# --- arm B: turboquant35 + calibration metadata ---------------------------

_launch_serve "turboquant35" \
    --kv-cache-dtype turboquant35 \
    --enable-turboquant \
    --turboquant-metadata-path "$META" \
    --turboquant-recent-ring-capacity "${TURBOQUANT_RING_CAPACITY:-1024}" \
    --no-enable-prefix-caching
_wait_for_health
_warmup_first_request
_run_bench "turboquant35"
_kill_serve

# --- compare --------------------------------------------------------------

echo
echo "[compare] generating markdown report"
"$VENV_BIN/python" "$REPO_ROOT/scripts/compare_bench_results.py" \
    --baseline "$RESULT_DIR/bench_baseline.json" \
    --treatment "$RESULT_DIR/bench_turboquant35.json" \
    --output "$RESULT_DIR/comparison.md"
echo
echo "Report: $RESULT_DIR/comparison.md"
