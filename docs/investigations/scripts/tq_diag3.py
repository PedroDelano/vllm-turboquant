"""Narrow down: which knob makes long-prompt TurboQuant work?

Tests turboquant35 on long prompts under 4 conditions. All greedy (T=0).
"""
from __future__ import annotations

import json
import statistics

from vllm import LLM, SamplingParams

SNAP = "/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
META = "/tmp/qwen2_5_turboquant35.json"

with open("/tmp/small_glaive_qwen25.jsonl") as f:
    prompts = [json.loads(l)["prompt"] for l in f][:8]

SP = SamplingParams(temperature=0.0, max_tokens=48, seed=0, ignore_eos=True)

BASE_KW = dict(
    model=SNAP,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_seqs=4,
    gpu_memory_utilization=0.80,
    attention_backend="TRITON_ATTN",
)
TQ_KW = dict(
    kv_cache_dtype="turboquant35",
    enable_turboquant=True,
    turboquant_metadata_path=META,
)


def fingerprint(text: str) -> tuple[float, float]:
    toks = text.split()
    uniq = len(set(toks)) / len(toks) if toks else 0.0
    nl = text.count("\n") / max(len(text), 1)
    return uniq, nl


def run(label: str, **extra) -> list[str]:
    print(f"\n=== {label} ===")
    llm = LLM(**BASE_KW, **extra)
    outs = llm.generate(prompts, SP)
    texts = [o.outputs[0].text for o in outs]
    uniq_scores, nl_scores = [], []
    for i, t in enumerate(texts):
        u, n = fingerprint(t)
        uniq_scores.append(u)
        nl_scores.append(n)
        print(f"  [{i}] u={u:.2f} nl={n:.2f}  {t[:120]!r}")
    print(f"  mean: uniq={statistics.mean(uniq_scores):.2f} nl_frac={statistics.mean(nl_scores):.2f}")
    del llm
    import torch
    torch.cuda.empty_cache()
    return texts


base = run("baseline bf16 KV", kv_cache_dtype="auto")
_ = run("tq default", **TQ_KW)
_ = run("tq + no prefix cache", **TQ_KW, enable_prefix_caching=False)
_ = run("tq + enforce_eager (no cudagraphs)", **TQ_KW, enforce_eager=True)
_ = run("tq + no chunked prefill", **TQ_KW, enable_chunked_prefill=False)
