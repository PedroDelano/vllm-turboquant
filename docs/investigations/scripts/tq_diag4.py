"""Compare turboquant25 vs turboquant35 on the same broken prompts."""
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


run(
    "turboquant25",
    kv_cache_dtype="turboquant25",
    enable_turboquant=True,
    turboquant_metadata_path="/tmp/qwen2_5_turboquant25.json",
)
run(
    "turboquant35",
    kv_cache_dtype="turboquant35",
    enable_turboquant=True,
    turboquant_metadata_path="/tmp/qwen2_5_turboquant35.json",
)
