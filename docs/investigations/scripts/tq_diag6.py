"""Compare bf16 / fp8 / turboquant35 on long prompts. Verifies that
fp8 gives quality close to bf16 while TurboQuant35 doesn't."""
from __future__ import annotations

import json
import statistics

from vllm import LLM, SamplingParams

SNAP = "/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
with open("/tmp/small_glaive_qwen25.jsonl") as f:
    prompts = [json.loads(l)["prompt"] for l in f][:8]
SP = SamplingParams(temperature=0.0, max_tokens=48, seed=0, ignore_eos=True)


def fp(text):
    toks = text.split()
    return (
        len(set(toks)) / len(toks) if toks else 0.0,
        text.count("\n") / max(len(text), 1),
    )


def jaccard(a, b):
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb: return 1.0
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)


def run(label, **kw):
    llm = LLM(
        model=SNAP,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.80,
        **kw,
    )
    outs = llm.generate(prompts, SP)
    texts = [o.outputs[0].text for o in outs]
    us = [fp(t)[0] for t in texts]
    print(f"{label:30s}  mean_uniq={statistics.mean(us):.2f}  nchars_mean={sum(len(t) for t in texts)/len(texts):.0f}")
    del llm
    import torch
    torch.cuda.empty_cache()
    return texts


base = run("bf16 (baseline)", kv_cache_dtype="auto", attention_backend="TRITON_ATTN")
fp8 = run("fp8_e4m3", kv_cache_dtype="fp8", attention_backend="TRITON_ATTN")
tq = run(
    "turboquant35",
    kv_cache_dtype="turboquant35",
    enable_turboquant=True,
    turboquant_metadata_path="/tmp/qwen2_5_turboquant35.json",
    attention_backend="TRITON_ATTN",
)

print()
print(f"Jaccard(bf16, fp8):           mean={statistics.mean([jaccard(a,b) for a,b in zip(base, fp8)]):.2f}")
print(f"Jaccard(bf16, turboquant35):  mean={statistics.mean([jaccard(a,b) for a,b in zip(base, tq)]):.2f}")

print("\nSample outputs:")
for i, (b, f, t) in enumerate(zip(base, fp8, tq)):
    print(f"[{i}]")
    print(f"  bf16: {b[:100]!r}")
    print(f"  fp8 : {f[:100]!r}")
    print(f"  tq35: {t[:100]!r}")
