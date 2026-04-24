# TurboQuant vs baseline — /workspace/hf-cache/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307

- Baseline: `bench_baseline.json` — ?
- Treatment: `bench_turboquant35.json` — ?
- Same prompts, same seed, same model path, same concurrency.

## Throughput and latency

| Metric | Baseline | Treatment | Δ |
|---|---:|---:|---:|
| total wall time | 32.59 s | 301.21 s | +824.3% |
| requests completed | 64 | 64 | 0.0% |
| input tokens (sum) | 262144 | 262144 | 0.0% |
| output tokens (sum) | 8192 | 8192 | 0.0% |
| request throughput | 1.964 req/s | 0.212 req/s | -89.2% |
| output token throughput | 251.38 tok/s | 27.20 tok/s | -89.2% |
| total token throughput | 8,295.5 tok/s | 897.49 tok/s | -89.2% |
| mean TTFT | 222.87 ms | 4,458.2 ms | +1900.3% |
| median TTFT | 218.43 ms | 4,506.7 ms | +1963.2% |
| p99 TTFT | 513.48 ms | 4,610.1 ms | +797.8% |
| mean ITL | 6.262 ms | 39.13 ms | +524.9% |
| median ITL | 6.183 ms | 39.10 ms | +532.3% |
| p99 ITL | 7.296 ms | 41.91 ms | +474.4% |
| mean E2E latency | — ms | — ms | — |
| median E2E latency | — ms | — ms | — |
| p99 E2E latency | — ms | — ms | — |

## Output quality (seeded, temp=0 where set)

- Requests compared: 64
- Exact-match rate: 0.0%
- Mean token-overlap (Jaccard): 26.5%
- Median token-overlap: 19.9%
- Worst token-overlap: 5.8%

### Largest divergences (lowest token overlap)

#### Request 26

_Baseline_:

```
freeze accounts and cards
- process account closures
- process account transfers (internal or external)

You are not allowed to:
- Help users locate or authenticate other users.
- Provide information about any profile, account, card, loan, beneficiary, or transaction that does not belong to the user
```

_Treatment_:

```
freeze accounts and cards
- unblock UPI

Unblock UPI:
- This is a specific feature where a UPI ID (username @ bank) has been blocked due to too many failed login attempts.
- It can only be unblocked by a human agent.
- If the user requests this, transfer to a human agent.
user
I need to add a new be
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
freeze cards and accounts

You cannot:
- Help users with authentication or locate profiles of other people
- Access other people's information or perform actions on their profiles
- Provide information about accounts, cards, loans, or transactions that the user does not own
- Freeze/unfreeze cards o
```

#### Request 54

_Baseline_:

```
freeze accounts and cards
- cancel loans
- add or remove beneficiaries from the beneficiary list
- change account settings (e.g., limit settings)

**Important rules:**
- You must always be certain of the user's identity before proceeding.
- You must never discuss the policy.
- You must never volunte
```

_Treatment_:

```
freeze accounts and cards
- provide information about their own profile, accounts, cards, loans, beneficiaries, and transactions

You MUST NOT:
- Provide information about anyone else's profile, accounts, cards, loans, beneficiaries, or transactions
- Provide information about any third-party servic
```
