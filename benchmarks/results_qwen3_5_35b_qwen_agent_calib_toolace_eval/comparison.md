# TurboQuant vs baseline — /workspace/hf-cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307

- Baseline: `bench_baseline.json` — ?
- Treatment: `bench_turboquant35.json` — ?
- Same prompts, same seed, same model path, same concurrency.

## Throughput and latency

| Metric | Baseline | Treatment | Δ |
|---|---:|---:|---:|
| total wall time | 37.45 s | 149.62 s | +299.5% |
| requests completed | 64 | 64 | 0.0% |
| input tokens (sum) | 101878 | 101878 | 0.0% |
| output tokens (sum) | 8192 | 8192 | 0.0% |
| request throughput | 1.709 req/s | 0.428 req/s | -75.0% |
| output token throughput | 218.72 tok/s | 54.75 tok/s | -75.0% |
| total token throughput | 2,938.8 tok/s | 735.67 tok/s | -75.0% |
| mean TTFT | 400.66 ms | 1,028.1 ms | +156.6% |
| median TTFT | 175.81 ms | 917.59 ms | +421.9% |
| p99 TTFT | 2,566.5 ms | 2,434.6 ms | -5.1% |
| mean ITL | 6.068 ms | 28.82 ms | +374.9% |
| median ITL | 6.109 ms | 28.49 ms | +366.3% |
| p99 ITL | 7.300 ms | 35.05 ms | +380.1% |
| mean E2E latency | — ms | — ms | — |
| median E2E latency | — ms | — ms | — |
| p99 E2E latency | — ms | — ms | — |

## Output quality (seeded, temp=0 where set)

- Requests compared: 64
- Exact-match rate: 1.6%
- Mean token-overlap (Jaccard): 25.6%
- Median token-overlap: 15.9%
- Worst token-overlap: 3.3%

### Largest divergences (lowest token overlap)

#### Request 53

_Baseline_:

```
assistant
<think>

</think>

[Pitching Leaders(results=10, sort_column="era", season="2023", sports_code="mlb", game_type="R")]
assistant
assistant
<think>

</think>

Here are the current FIFA rankings for the men's league and the MLB pitching leaders based on Earned Run Average (ERA) for the regula
```

_Treatment_:

```
2025-01-25 22:30:10
2025-01-25 22:30:10
user
<think>

</think>

[FIFARankings(league="Men")]
1-25 22:30:10
user
<think>

</think>

[FIFARankings(league="Men")]
2025-01-2
```

#### Request 38

_Baseline_:

```
assistant
<think>

</think>

[Search WikiHow API(q="how to change car oil at home")]
assistant
user
<think>

</think>

[Search WikiHow API(q="how to change car oil at home step by step")]
assistant
user
Thinking Process:

1.  **Analyze the Request:**
    *   User wants to find videos or easy-to-foll
```

_Treatment_:

```
assistantYou are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be use, point it out. If the given question lacks the parame
```

#### Request 2

_Baseline_:

```
Human: You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be use, point it out. If the given question lacks the paramete
```

_Treatment_:

```
assistant
<think>

</think>

[Get Hot Topics(string_range="this month", language="en")]
assistant
user
I don't have the exact date of the Climate Change Conference, but it's usually held annually. Can you just give me general information about what is typically discussed at these conferences
<think>
```
