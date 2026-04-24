"""Measure actual quantization reconstruction error on Qwen2.5-0.5B K/V.

Captures K/V from a real forward pass (one of the glaive prompts),
quantizes with the calibration metadata we generated, dequantizes, and
compares to the original. This isolates whether TurboQuant lossiness
is beyond what Qwen can tolerate on head_dim=64.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm.v1.attention.ops.turboquant_kv_cache import (
    build_turboquant_outlier_masks,
    dequantize_turboquant_vectors,
    get_turboquant_centroids,
    get_turboquant_layout,
    get_turboquant_qjl_matrix,
    get_turboquant_rotation,
    quantize_turboquant_vectors,
)


def _get_turboquant_tables(recipe, head_size, device):
    layout = get_turboquant_layout(recipe, head_size)
    rotations = (
        get_turboquant_rotation(device, layout.groups[0].dim, seed_offset=101),
        get_turboquant_rotation(device, layout.groups[1].dim, seed_offset=211),
    )
    qjl_matrices = (
        get_turboquant_qjl_matrix(device, layout.groups[0].dim, seed_offset=307),
        get_turboquant_qjl_matrix(device, layout.groups[1].dim, seed_offset=401),
    )
    centroids = {
        group.mse_bits: get_turboquant_centroids(device, group.dim, group.mse_bits)
        for group in layout.groups
        if group.mse_bits > 0
    }
    return rotations, qjl_matrices, centroids, layout

SNAP = "/workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
META_PATH = "/tmp/qwen2_5_turboquant35.json"

meta = json.loads(Path(META_PATH).read_text())
head_size = meta["head_size"]
recipe = meta["recipe"]
print(f"model head_size={head_size} recipe={recipe}")

tok = AutoTokenizer.from_pretrained(SNAP)
with open("/tmp/small_glaive_qwen25.jsonl") as f:
    prompt = json.loads(f.readline())["prompt"]
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
print(f"prompt tokens: {ids.shape[-1]}")

model = AutoModelForCausalLM.from_pretrained(
    SNAP, torch_dtype=torch.bfloat16, attn_implementation="eager"
).to("cuda:0")
model.eval()

# Forward with output_attentions=False, output_hidden_states=False, use_cache=True.
with torch.no_grad():
    out = model(ids, use_cache=True)

# past_key_values contains per-layer (K, V) tuples for the attention layers.
# For Qwen2.5-0.5B, they're all standard attention.
pkv = out.past_key_values
if hasattr(pkv, "to_legacy_cache"):
    pkv = pkv.to_legacy_cache()

# Pick layer 0's key tensor. Shape (batch, heads, tokens, head_dim).
layer0_k = pkv[0][0].squeeze(0).transpose(0, 1).contiguous()  # (tokens, heads, head_dim)
layer0_v = pkv[0][1].squeeze(0).transpose(0, 1).contiguous()
print(f"layer0 K shape: {layer0_k.shape}, dtype: {layer0_k.dtype}")
print(f"layer0 K mean-abs: {layer0_k.abs().mean().item():.4f}  std: {layer0_k.std().item():.4f}")

# Build or load the calibration masks for layer 0
# Calibration metadata stores high_precision_indices per KV head.
layer_name = sorted(meta["layers"].keys())[0]
calib = meta["layers"][layer_name]
key_hpi = calib["key_high_precision_indices"]  # list of lists
value_hpi = calib["value_high_precision_indices"]
print(f"calib layer {layer_name}: key_hpi heads={len(key_hpi)}, len0={len(key_hpi[0])}")

# Convert hpi list-of-lists to (outlier_idx, regular_idx) int64 tensors
def build_group_indices(hpi, head_size):
    n_heads = len(hpi)
    outlier = torch.tensor(hpi, dtype=torch.int64)  # (heads, outlier_count)
    all_idx = torch.arange(head_size, dtype=torch.int64).unsqueeze(0).expand(n_heads, -1)
    mask = torch.ones(n_heads, head_size, dtype=torch.bool)
    mask.scatter_(1, outlier, False)
    regular = all_idx[mask].reshape(n_heads, -1)
    return outlier, regular

key_mask = build_group_indices(key_hpi, head_size)    # tuple (outlier, regular)
value_mask = build_group_indices(value_hpi, head_size)

# quantize_turboquant_vectors expects masks sized (n_heads, head_size) bool
# Shape the K to match: the function expects (..., n_heads, head_size).
# layer0_k is already (tokens, heads, head_dim). Let's quantize that.
device = torch.device("cuda:0")
tables = _get_turboquant_tables(recipe, head_size, device)
key_mask_gpu = tuple(t.to(device) for t in key_mask)
value_mask_gpu = tuple(t.to(device) for t in value_mask)

# Quantize with CALIBRATED mask
for name, x, mask_from_calib in [("K", layer0_k, key_mask_gpu), ("V", layer0_v, value_mask_gpu)]:
    x_fp32 = x.float()  # quant functions want float32 per test
    # The mask_from_calib is the CALIBRATED outlier mask; use it directly.
    packed = quantize_turboquant_vectors(
        x_fp32, recipe, tables[0], tables[1], tables[2], mask_from_calib,
    )
    restored = dequantize_turboquant_vectors(
        packed, recipe, head_size, tables[0], tables[1], tables[2], mask_from_calib, x_fp32.dtype,
    )
    mse_c = torch.mean((x_fp32 - restored) ** 2).item()
    cos_c = torch.nn.functional.cosine_similarity(
        x_fp32.reshape(-1, head_size), restored.reshape(-1, head_size), dim=-1,
    ).mean().item()
    rel_c = (torch.norm(x_fp32 - restored) / torch.norm(x_fp32)).item()

    # Now quantize with *default* mask (what happens without calibration)
    default_mask = build_turboquant_outlier_masks(x_fp32, recipe)
    packed_d = quantize_turboquant_vectors(
        x_fp32, recipe, tables[0], tables[1], tables[2], default_mask,
    )
    restored_d = dequantize_turboquant_vectors(
        packed_d, recipe, head_size, tables[0], tables[1], tables[2], default_mask, x_fp32.dtype,
    )
    mse_d = torch.mean((x_fp32 - restored_d) ** 2).item()
    cos_d = torch.nn.functional.cosine_similarity(
        x_fp32.reshape(-1, head_size), restored_d.reshape(-1, head_size), dim=-1,
    ).mean().item()
    rel_d = (torch.norm(x_fp32 - restored_d) / torch.norm(x_fp32)).item()

    print(f"\n{name}:")
    print(f"  CALIBRATED mask:  MSE={mse_c:.4e}  cos_sim={cos_c:.4f}  rel_err={rel_c:.4f}")
    print(f"  DEFAULT mask:     MSE={mse_d:.4e}  cos_sim={cos_d:.4f}  rel_err={rel_d:.4f}")
    # Compare which outlier dims each picks (just head 0)
    calib_set = set((key_mask if name == "K" else value_mask)[0][0].tolist())
    default_set = set(default_mask[0][0].cpu().tolist())
    print(f"  head0 outlier overlap: {len(calib_set & default_set)}/{len(calib_set)}")
