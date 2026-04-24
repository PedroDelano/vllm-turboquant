# TurboQuant vs baseline — /workspace/hf-cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307

- Baseline: `bench_baseline.json` — ?
- Treatment: `bench_turboquant35.json` — ?
- Same prompts, same seed, same model path, same concurrency.

## Throughput and latency

| Metric | Baseline | Treatment | Δ |
|---|---:|---:|---:|
| total wall time | 33.85 s | 284.81 s | +741.3% |
| requests completed | 64 | 64 | 0.0% |
| input tokens (sum) | 262144 | 262144 | 0.0% |
| output tokens (sum) | 8192 | 8192 | 0.0% |
| request throughput | 1.891 req/s | 0.225 req/s | -88.1% |
| output token throughput | 241.99 tok/s | 28.76 tok/s | -88.1% |
| total token throughput | 7,985.6 tok/s | 949.19 tok/s | -88.1% |
| mean TTFT | 264.50 ms | 3,960.3 ms | +1397.2% |
| median TTFT | 224.81 ms | 4,256.9 ms | +1793.6% |
| p99 TTFT | 1,744.5 ms | 4,745.8 ms | +172.0% |
| mean ITL | 6.247 ms | 38.89 ms | +522.6% |
| median ITL | 6.184 ms | 37.53 ms | +506.9% |
| p99 ITL | 7.350 ms | 40.25 ms | +447.7% |
| mean E2E latency | — ms | — ms | — |
| median E2E latency | — ms | — ms | — |
| p99 E2E latency | — ms | — ms | — |

## Output quality (seeded, temp=0 where set)

- Requests compared: 64
- Exact-match rate: 0.0%
- Mean token-overlap (Jaccard): 29.6%
- Median token-overlap: 22.4%
- Worst token-overlap: 8.5%

### Largest divergences (lowest token overlap)

#### Request 13

_Baseline_:

```
freeze cards and accounts
- make beneficiary adjustments and other updates
- transfer to a human agent if needed
user
I'm trying to send money to a beneficiary named 'Sarah Jones'. I know her name but need to find her ID.
<think>

</think>

I can help you find the beneficiary ID for 'Sarah Jones'. T
```

_Treatment_:

```
freeze accounts and cards

Always adhere to the following rules:
1. Do not answer questions about policies not covered in the provided policy text.
2. Do not provide technical/internal terminology, explain in plain English.
3. For any action (transfer, freeze, add beneficiary, etc.), you must explai
```

#### Request 33

_Baseline_:

```
freeze cards or accounts

**Do NOT** provide any information about other people's profiles, accounts, cards, loans, beneficiaries, or transactions.
**Do NOT** make any changes to the user's profile unless explicitly asked to do so by the user.
**Do NOT** make any changes to the user's profile unless
```

_Treatment_:

```
freeze cards and accounts
- transfer to a human agent for further assistance

Rules:
- You must authenticate the user before providing any personal information.
- For sensitive actions (freezing, adding beneficiaries, making transfers, etc.), you must explain the details and get explicit user confir
```

#### Request 50

_Baseline_:

```
freeze cards and accounts
- transfer the user to a human agent (which requires a summary of the issue)

Rules:
- Never give information about other people's accounts, cards, loans, etc.
- Never give any passwords, PINs, or security codes.
- You may use the calculate tool to do basic arithmetic if re
```

_Treatment_:

```
freeze accounts and cards
- transfer to human agents for issues outside your scope (e.g., credit line increases, dispute resolution)

Your primary tools for information gathering are:
- get_client_details
- get_account_details
- get_card_details
- find_client_id_by_email

user
I need to transfer $50
```
