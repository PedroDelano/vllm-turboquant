"""Create an 'identity' metadata: high_precision_indices = [0..outlier_count-1] for every head."""
import json
import sys

src = "/tmp/qwen2_5_turboquant35.json"
dst = "/tmp/qwen2_5_turboquant35_identity.json"

d = json.load(open(src))
outlier_count = 32  # for head_size=64, turboquant35
for layer_name, layer in d["layers"].items():
    heads_k = len(layer["key_high_precision_indices"])
    heads_v = len(layer["value_high_precision_indices"])
    layer["key_high_precision_indices"] = [list(range(outlier_count))] * heads_k
    layer["value_high_precision_indices"] = [list(range(outlier_count))] * heads_v

d["calibration"]["method"] = "identity_test"
json.dump(d, open(dst, "w"), indent=2)
print(f"wrote {dst}")
