#!/usr/bin/env python3
"""
Compare two `vllm bench serve --save-detailed` result JSONs.

Emits a markdown report with:

- Top-level throughput / latency deltas (baseline → treatment)
- Percentile latencies (TTFT, ITL, E2E) side by side
- Quality diff: exact-match rate and token-overlap between per-prompt
  generated outputs, since both arms saw the same seeded prompt set.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable
from pathlib import Path

SCALAR_METRICS: list[tuple[str, str, str]] = [
    ("duration", "total wall time", "s"),
    ("completed", "requests completed", ""),
    ("total_input_tokens", "input tokens (sum)", ""),
    ("total_output_tokens", "output tokens (sum)", ""),
    ("request_throughput", "request throughput", "req/s"),
    ("output_throughput", "output token throughput", "tok/s"),
    ("total_token_throughput", "total token throughput", "tok/s"),
    ("mean_ttft_ms", "mean TTFT", "ms"),
    ("median_ttft_ms", "median TTFT", "ms"),
    ("p99_ttft_ms", "p99 TTFT", "ms"),
    ("mean_itl_ms", "mean ITL", "ms"),
    ("median_itl_ms", "median ITL", "ms"),
    ("p99_itl_ms", "p99 ITL", "ms"),
    ("mean_e2el_ms", "mean E2E latency", "ms"),
    ("median_e2el_ms", "median E2E latency", "ms"),
    ("p99_e2el_ms", "p99 E2E latency", "ms"),
]


def _fmt_num(x: object) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x != x:  # NaN
            return "nan"
        if abs(x) >= 1000:
            return f"{x:,.1f}"
        if abs(x) >= 10:
            return f"{x:.2f}"
        return f"{x:.3f}"
    return str(x)


def _fmt_delta(base: object, treat: object) -> str:
    if not isinstance(base, (int, float)) or not isinstance(treat, (int, float)):
        return "—"
    if base == 0:
        return "—"
    pct = (treat - base) / base * 100.0
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap over whitespace tokens — a rough but cheap proxy."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _quality_stats(base_texts: list[str], treat_texts: list[str]) -> dict:
    n = min(len(base_texts), len(treat_texts))
    if n == 0:
        return {"n": 0}
    exact = sum(1 for i in range(n) if base_texts[i] == treat_texts[i])
    overlaps = [_token_overlap(base_texts[i], treat_texts[i]) for i in range(n)]
    return {
        "n": n,
        "exact_match_rate": exact / n,
        "mean_token_overlap": statistics.mean(overlaps),
        "median_token_overlap": statistics.median(overlaps),
        "min_token_overlap": min(overlaps),
    }


def _diff_examples(
    base_texts: list[str],
    treat_texts: list[str],
    k: int = 3,
    maxlen: int = 300,
) -> list[tuple[int, str, str]]:
    """Pick up to k requests where the arms diverged the most."""
    n = min(len(base_texts), len(treat_texts))
    scored: list[tuple[float, int]] = []
    for i in range(n):
        scored.append((_token_overlap(base_texts[i], treat_texts[i]), i))
    scored.sort()  # lowest overlap first
    out = []
    for _, i in scored[:k]:
        out.append((
            i,
            base_texts[i][:maxlen],
            treat_texts[i][:maxlen],
        ))
    return out


def _render(base_path: Path, treat_path: Path, out: Iterable[str]) -> str:
    lines: list[str] = []
    base = json.loads(base_path.read_text())
    treat = json.loads(treat_path.read_text())

    lines.append(
        f"# TurboQuant vs baseline — {base.get('model_id', 'model')}\n"
    )
    lines.append(
        f"- Baseline: `{base_path.name}` — {base.get('kv_cache_dtype', '?')}"
    )
    lines.append(
        f"- Treatment: `{treat_path.name}` — "
        f"{treat.get('kv_cache_dtype', '?')}"
    )
    lines.append(
        f"- Same prompts, same seed, same model path, same concurrency."
    )
    lines.append("")

    lines.append("## Throughput and latency\n")
    lines.append("| Metric | Baseline | Treatment | Δ |")
    lines.append("|---|---:|---:|---:|")
    for key, name, unit in SCALAR_METRICS:
        b = base.get(key)
        t = treat.get(key)
        unit_s = f" {unit}" if unit else ""
        lines.append(
            f"| {name} | {_fmt_num(b)}{unit_s} | {_fmt_num(t)}{unit_s} | "
            f"{_fmt_delta(b, t)} |"
        )
    lines.append("")

    base_texts = base.get("generated_texts") or []
    treat_texts = treat.get("generated_texts") or []
    q = _quality_stats(base_texts, treat_texts)

    lines.append("## Output quality (seeded, temp=0 where set)\n")
    if q.get("n", 0) == 0:
        lines.append("_No per-request `generated_texts` in result JSONs — "
                     "rerun bench with `--save-detailed`._\n")
    else:
        lines.append(f"- Requests compared: {q['n']}")
        lines.append(
            f"- Exact-match rate: {q['exact_match_rate']*100:.1f}%"
        )
        lines.append(
            f"- Mean token-overlap (Jaccard): "
            f"{q['mean_token_overlap']*100:.1f}%"
        )
        lines.append(
            f"- Median token-overlap: "
            f"{q['median_token_overlap']*100:.1f}%"
        )
        lines.append(
            f"- Worst token-overlap: "
            f"{q['min_token_overlap']*100:.1f}%"
        )
        lines.append("")

        diffs = _diff_examples(base_texts, treat_texts, k=3)
        if diffs:
            lines.append("### Largest divergences (lowest token overlap)\n")
            for idx, b, t in diffs:
                lines.append(f"#### Request {idx}\n")
                lines.append("_Baseline_:\n")
                lines.append("```")
                lines.append(b.strip() or "(empty)")
                lines.append("```\n")
                lines.append("_Treatment_:\n")
                lines.append("```")
                lines.append(t.strip() or "(empty)")
                lines.append("```\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--treatment", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, write markdown here in addition to stdout.",
    )
    args = parser.parse_args()
    report = _render(args.baseline, args.treatment, [])
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)


if __name__ == "__main__":
    main()
