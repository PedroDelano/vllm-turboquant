# TurboQuant vs baseline — /workspace/hf-cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775

- Baseline: `bench_baseline.json` — ?
- Treatment: `bench_turboquant35.json` — ?
- Same prompts, same seed, same model path, same concurrency.

## Throughput and latency

| Metric | Baseline | Treatment | Δ |
|---|---:|---:|---:|
| total wall time | 5.337 s | 63.05 s | +1081.4% |
| requests completed | 64 | 64 | 0.0% |
| input tokens (sum) | 32653 | 32653 | 0.0% |
| output tokens (sum) | 8192 | 8192 | 0.0% |
| request throughput | 11.99 req/s | 1.015 req/s | -91.5% |
| output token throughput | 1,534.9 tok/s | 129.92 tok/s | -91.5% |
| total token throughput | 7,652.8 tok/s | 647.78 tok/s | -91.5% |
| mean TTFT | 97.92 ms | 336.09 ms | +243.2% |
| median TTFT | 73.35 ms | 282.21 ms | +284.8% |
| p99 TTFT | 360.13 ms | 838.63 ms | +132.9% |
| mean ITL | 1.871 ms | 28.49 ms | +1422.6% |
| median ITL | 1.837 ms | 28.55 ms | +1454.4% |
| p99 ITL | 3.573 ms | 31.68 ms | +786.6% |
| mean E2E latency | — ms | — ms | — |
| median E2E latency | — ms | — ms | — |
| p99 E2E latency | — ms | — ms | — |

## Output quality (seeded, temp=0 where set)

- Requests compared: 64
- Exact-match rate: 0.0%
- Mean token-overlap (Jaccard): 3.7%
- Median token-overlap: 1.6%
- Worst token-overlap: 0.0%

### Largest divergences (lowest token overlap)

#### Request 1

_Baseline_:

```
Human: How does the sentence "The quick brown fox jumps over the lazy dog" relate to the sentence "The quick brown dog jumps over the lazy fox"?
A. They are synonyms
B. They are antonyms
C. They are homophones
D. They are homonyms
E. They are homographs


The sentence "The quick brown fox jumps over
```

_Treatment_:

```
(empty)
```

#### Request 3

_Baseline_:

```
Human resources managers need to be able to manage a large number of people efficiently. How can they ensure that everyone is doing their job and contributing to the success of the company?
assistant
Managing a large number of people efficiently can be challenging, but here are some strategies that
```

_Treatment_:

```
(empty)
```

#### Request 5

_Baseline_:

```
Human: I need help with a math problem. I'm trying to solve the equation 2x + 3 = 11. Can you help me?
user
Human: Yes, of course! Please provide the equation.
assistant
Certainly! The equation you're working with is:

\[ 2x + 3 = 11 \]

To solve for \( x \), you would need to isolate \( x \) on one
```

_Treatment_:

```
(empty)
```
