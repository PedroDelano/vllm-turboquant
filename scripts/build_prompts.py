#!/usr/bin/env python3
"""Build a JSONL calibration / benchmark prompts file from an arbitrary
tool-calling or chat dataset.

Dispatches to one of the adapters registered in
`calibration.datasets.ADAPTERS` and runs the generic build pipeline.
Output JSONL has `{"prompt": str, "text": str}` per line — the same
shape consumed by both `benchmarks/generate_turboquant_metadata.py`
(reads `text`) and `vllm bench serve --dataset-name custom` (reads
`prompt`).

Example:

    python scripts/build_prompts.py \\
        --dataset xlam \\
        --tokenizer /workspace/hf-cache/.../Qwen3.5-0.8B/snapshots/<sha>/ \\
        --output calibration/prompts/toolcalling_qwen3_5_xlam.jsonl \\
        --num-prompts 300 \\
        --max-tokens 1024
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

# Allow `python scripts/build_prompts.py ...` to work from the repo root
# without requiring PYTHONPATH=. to be set by the caller.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from calibration.datasets import ADAPTERS, build  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(ADAPTERS.keys()),
        help="Source dataset name (registered in calibration.datasets.ADAPTERS).",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HF id or local path of a tokenizer carrying the target chat template.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=300,
        help="Number of rendered prompts to emit.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=128,
        help="Drop rendered prompts shorter than this many tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Truncate rendered prompts to this many tokens.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split (defaults to the adapter's default_split).",
    )
    parser.add_argument(
        "--tools-fallback",
        choices=("error", "render-as-system", "drop"),
        default="error",
        help=(
            "Behavior when tokenizer.apply_chat_template rejects the `tools=` "
            "argument. Default: error (fail fast)."
        ),
    )
    parser.add_argument(
        "--subset",
        default=None,
        help=(
            "Adapter subset (only meaningful for datasets with configs, "
            "e.g. BFCL). Passed to adapter constructor if the adapter "
            "accepts it."
        ),
    )
    args = parser.parse_args()

    adapter_cls = ADAPTERS[args.dataset]
    adapter_params = inspect.signature(adapter_cls).parameters
    adapter_kwargs: dict = {}
    if args.subset is not None and "subset" in adapter_params:
        adapter_kwargs["subset"] = args.subset
    adapter = adapter_cls(**adapter_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )

    report = build(
        adapter,
        tokenizer=tokenizer,
        output_path=args.output,
        num_prompts=args.num_prompts,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        split=args.split,
        tools_fallback=args.tools_fallback,
    )
    print(
        f"Wrote {report.kept} prompts to {report.output_path} "
        f"(skipped {report.skipped} malformed/too-short)."
    )


if __name__ == "__main__":
    main()
