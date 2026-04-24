# TurboQuant vs baseline — /workspace/hf-cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307

- Baseline: `bench_baseline.json` — ?
- Treatment: `bench_turboquant35.json` — ?
- Same prompts, same seed, same model path, same concurrency.

## Throughput and latency

| Metric | Baseline | Treatment | Δ |
|---|---:|---:|---:|
| total wall time | 32.40 s | 404.40 s | +1148.2% |
| requests completed | 64 | 64 | 0.0% |
| input tokens (sum) | 262144 | 262144 | 0.0% |
| output tokens (sum) | 8192 | 8192 | 0.0% |
| request throughput | 1.975 req/s | 0.158 req/s | -92.0% |
| output token throughput | 252.85 tok/s | 20.26 tok/s | -92.0% |
| total token throughput | 8,344.0 tok/s | 668.49 tok/s | -92.0% |
| mean TTFT | 217.89 ms | 4,479.9 ms | +1956.0% |
| median TTFT | 218.19 ms | 4,529.3 ms | +1975.8% |
| p99 TTFT | 268.45 ms | 4,799.2 ms | +1687.8% |
| mean ITL | 6.254 ms | 64.39 ms | +929.5% |
| median ITL | 6.188 ms | 64.06 ms | +935.3% |
| p99 ITL | 7.671 ms | 82.00 ms | +969.0% |
| mean E2E latency | — ms | — ms | — |
| median E2E latency | — ms | — ms | — |
| p99 E2E latency | — ms | — ms | — |

## Output quality (seeded, temp=0 where set)

- Requests compared: 64
- Exact-match rate: 0.0%
- Mean token-overlap (Jaccard): 28.9%
- Median token-overlap: 20.4%
- Worst token-overlap: 7.4%

### Largest divergences (lowest token overlap)

#### Request 45

_Baseline_:

```
freeze cards and accounts
- transfer to a human agent if the user's issue is not solvable with the available tools.

IMPORTANT: Never generate a JSON object for a non-JSON tool call. Only make JSON tool calls when calling a tool.
assistant
<think>
The user is asking to transfer money from one accoun
```

_Treatment_:

```
freeze cards and accounts
- cancel pending transfers or unfreeze accounts (in case of user error)

Important guidelines:
- You can only help the user with their own profile, accounts, cards, loans, and beneficiaries.
- Always ensure proper authentication before making any changes.
- Always ensure th
```

#### Request 25

_Baseline_:

```
freeze accounts or cards if there is suspicious activity or user requests
- transfer to human agents for issues beyond your scope
user
I'm having trouble with my card
assistant
<think>

</think>

I understand you're having trouble with your card. To help you, I need to identify your client profile f
```

_Treatment_:

```
freeze accounts and cards

You should NOT:
- Provide information about someone else's profile, accounts, cards, loans, or transactions
- Provide information about a user's transactions if they ask about other users' transactions
- Give legal, tax, or investment advice
- Provide information about a u
```

#### Request 40

_Baseline_:

```
freeze accounts and cards
- provide financial advice
- handle complaints
- transfer to a human agent if necessary.

You must follow the policy exactly. Each tool call corresponds to a specific action.
You must call tools to perform actions; you cannot just "think" them into existence.
<think>

</thi
```

_Treatment_:

```
freeze cards
- freeze/unfreeze accounts
- process card freezing (including issuing new cards after freezing)

You cannot:
- Process card freezing (including issuing new cards after freezing) if the user does not provide a reason for freezing.
- Freeze or unfreeze an account if the user does not prov
```
