"""Check whether TurboQuant35 degenerates on long prompts even at T=0.

Uses the same 1024-token glaive-templated prompts the bench used,
generates greedy (temp=0) to avoid sampling confounds.
"""
from __future__ import annotations

import json
import sys

from vllm import LLM, SamplingParams

SNAP = "/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
META = "/tmp/qwen2_5_turboquant35.json"

prompts: list[str] = []
with open("/tmp/small_glaive_qwen25.jsonl") as f:
    for line in f:
        d = json.loads(line)
        prompts.append(d["prompt"])
prompts = prompts[:8]  # small subset

SP = SamplingParams(temperature=0.0, max_tokens=48, seed=0, ignore_eos=True)


def fingerprint(text: str) -> str:
    toks = text.split()
    uniq = len(set(toks)) / len(toks) if toks else 0.0
    nl = text.count("\n") / max(len(text), 1)
    return f"nchars={len(text)} uniq={uniq:.2f} nl={nl:.2f}"


def run(label: str, **kw) -> list[str]:
    print(f"\n=== {label} ===")
    llm = LLM(
        model=SNAP,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.80,
        attention_backend="TRITON_ATTN",
        **kw,
    )
    outs = llm.generate(prompts, SP)
    texts = [o.outputs[0].text for o in outs]
    for i, t in enumerate(texts):
        print(f"  [{i}] {fingerprint(t)}")
        print(f"      {t[:180]!r}")
    del llm
    import torch
    torch.cuda.empty_cache()
    return texts


def jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


base = run("A  bf16 KV, T=0, long prompts", kv_cache_dtype="auto")
tq = run(
    "B  turboquant35, T=0, long prompts",
    kv_cache_dtype="turboquant35",
    enable_turboquant=True,
    turboquant_metadata_path=META,
)
print("\n=== Pairwise Jaccard (base vs tq) ===")
for i, (a, b) in enumerate(zip(base, tq)):
    print(f"  [{i}] jaccard={jaccard(a, b):.2f}")
import statistics
jvals = [jaccard(a, b) for a, b in zip(base, tq)]
print(f"  mean: {statistics.mean(jvals):.2f}")
