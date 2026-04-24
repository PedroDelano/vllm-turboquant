"""Diagnostic: TurboQuant35 offline generation on Qwen2.5-0.5B.

Runs three conditions side-by-side:
  A) bf16 KV (known good)
  B) turboquant35 with prefix caching enabled (default)
  C) turboquant35 with prefix caching disabled
  D) turboquant35 with enforce_eager (no cudagraphs)

For each, generates 3 short completions with a fixed prompt and reports
output repr + unique-token ratio. Helps isolate whether degeneration is
caused by prefix caching or cudagraph artifacts.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("VLLM_USE_V1", "1")

from vllm import LLM, SamplingParams  # noqa: E402

SNAP = "/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
META = "/tmp/qwen2_5_turboquant35.json"

PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nWrite one short sentence about the ocean.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

SP = SamplingParams(temperature=0.0, max_tokens=48, seed=0)


def fingerprint(text: str) -> str:
    toks = text.split()
    uniq = len(set(toks)) / len(toks) if toks else 0.0
    nl = text.count("\n") / max(len(text), 1)
    return f"nchars={len(text)} uniq_tokens={uniq:.2f} newline_frac={nl:.2f}"


def run(label: str, **llm_kwargs) -> None:
    print(f"\n=== {label} ===")
    print(f"  llm_kwargs: {llm_kwargs}")
    llm = LLM(
        model=SNAP,
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=2,
        gpu_memory_utilization=0.80,
        attention_backend="TRITON_ATTN",
        **llm_kwargs,
    )
    outs = llm.generate([PROMPT] * 2, SP)
    for i, o in enumerate(outs):
        t = o.outputs[0].text
        print(f"  [{i}] {fingerprint(t)}")
        print(f"      repr={t[:200]!r}")
    del llm
    import torch
    torch.cuda.empty_cache()


def main() -> None:
    conditions = sys.argv[1:] if len(sys.argv) > 1 else ["A", "B", "C", "D"]
    if "A" in conditions:
        run("A  bf16 KV (baseline)", kv_cache_dtype="auto")
    if "B" in conditions:
        run(
            "B  turboquant35 with prefix caching ENABLED (default)",
            kv_cache_dtype="turboquant35",
            enable_turboquant=True,
            turboquant_metadata_path=META,
        )
    if "C" in conditions:
        run(
            "C  turboquant35 with prefix caching DISABLED",
            kv_cache_dtype="turboquant35",
            enable_turboquant=True,
            turboquant_metadata_path=META,
            enable_prefix_caching=False,
        )
    if "D" in conditions:
        run(
            "D  turboquant35 with enforce_eager=True (no CUDAgraphs)",
            kv_cache_dtype="turboquant35",
            enable_turboquant=True,
            turboquant_metadata_path=META,
            enforce_eager=True,
        )


if __name__ == "__main__":
    main()
