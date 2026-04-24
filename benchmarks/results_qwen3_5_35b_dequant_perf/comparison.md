# TurboQuant vs baseline — /workspace/hf-cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307

- Baseline: `bench_baseline.json` — ?
- Treatment: `bench_turboquant35.json` — ?
- Same prompts, same seed, same model path, same concurrency.

## Throughput and latency

| Metric | Baseline | Treatment | Δ |
|---|---:|---:|---:|
| total wall time | 32.50 s | 348.33 s | +971.6% |
| requests completed | 64 | 64 | 0.0% |
| input tokens (sum) | 262144 | 262144 | 0.0% |
| output tokens (sum) | 8192 | 8192 | 0.0% |
| request throughput | 1.969 req/s | 0.184 req/s | -90.7% |
| output token throughput | 252.03 tok/s | 23.52 tok/s | -90.7% |
| total token throughput | 8,317.0 tok/s | 776.09 tok/s | -90.7% |
| mean TTFT | 220.07 ms | 4,459.2 ms | +1926.2% |
| median TTFT | 224.95 ms | 4,508.0 ms | +1904.0% |
| p99 TTFT | 279.60 ms | 4,645.6 ms | +1561.5% |
| mean ITL | 6.262 ms | 50.73 ms | +710.1% |
| median ITL | 6.184 ms | 50.77 ms | +721.0% |
| p99 ITL | 7.260 ms | 53.89 ms | +642.3% |
| mean E2E latency | — ms | — ms | — |
| median E2E latency | — ms | — ms | — |
| p99 E2E latency | — ms | — ms | — |

## Output quality (seeded, temp=0 where set)

- Requests compared: 64
- Exact-match rate: 0.0%
- Mean token-overlap (Jaccard): 27.1%
- Median token-overlap: 21.2%
- Worst token-overlap: 9.2%

### Largest divergences (lowest token overlap)

#### Request 30

_Baseline_:

```
freeze accounts and cards
- provide information about the system
- answer questions about policy

You are NOT authorized to:
- Give advice or make decisions on behalf of the user (e.g., which loan to pay, which card to freeze, which beneficiary to add/verify)
- Provide advice on how to avoid policy
```

_Treatment_:

```
freeze accounts and cards, and provide relevant explanations

For any other requests, you must transfer the user to a human agent.
</policy
user
I want to add a new beneficiary named Sarah Connor with the following details: first name Sarah, last name Connor, bank name Chase, account number 12345678
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
freeze cards or accounts
- generate reports (if the user explicitly asks for it)

Important:
- You must follow the policy.
- You should never make up information or lie to the user.
- You should provide information about the user's own profile, accounts, cards, loans, beneficiaries, and transactions
```

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
freeze accounts or cards
- search transactions with various filters
- get account details

General rules:
- You can only help the user with their own profile/accounts/cards/loans/beneficiaries/transactions.
- You cannot help with anyone else's information or actions.
- The user has not been authenti
```
