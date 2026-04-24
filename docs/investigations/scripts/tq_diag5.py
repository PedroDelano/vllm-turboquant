"""Test identity-metadata vs calibrated metadata on long prompts.

If calibration is the source of the problem, identity should produce
different / better outputs.  If kernel/setup is the problem, identity
should produce equally-broken outputs.
"""
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


def run(label, **kw):
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
    us, nls = [], []
    for i, t in enumerate(texts):
        u, nl = fp(t)
        us.append(u)
        nls.append(nl)
        print(f"  [{i}] u={u:.2f} nl={nl:.2f}  {t[:100]!r}")
    print(f"  mean: uniq={statistics.mean(us):.2f}  nl={statistics.mean(nls):.2f}")
    del llm
    import torch
    torch.cuda.empty_cache()
    return texts


tq_calib = run(
    "tq35 CALIBRATED",
    kv_cache_dtype="turboquant35",
    enable_turboquant=True,
    turboquant_metadata_path="/tmp/qwen2_5_turboquant35.json",
)
tq_ident = run(
    "tq35 IDENTITY",
    kv_cache_dtype="turboquant35",
    enable_turboquant=True,
    turboquant_metadata_path="/tmp/qwen2_5_turboquant35_identity.json",
)

print("\n=== pairwise Jaccard (calibrated vs identity) ===")
def jaccard(a, b):
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb: return 1.0
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)
vals = [jaccard(a, b) for a, b in zip(tq_calib, tq_ident)]
print(f"  mean: {statistics.mean(vals):.2f}")
for i, (a, b) in enumerate(zip(tq_calib, tq_ident)):
    print(f"  [{i}] {jaccard(a, b):.2f}")
