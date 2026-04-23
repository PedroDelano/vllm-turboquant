# TurboQuant vs baseline — /workspace/hf-cache/models--RedHatAI--Qwen3.5-122B-A10B-NVFP4/snapshots/49d19c108259a21450c40b8af38828b0a97390d8

- Baseline: `bench_baseline.json` — ?
- Treatment: `bench_turboquant35.json` — ?
- Same prompts, same seed, same model path, same concurrency.

## Throughput and latency

| Metric | Baseline | Treatment | Δ |
|---|---:|---:|---:|
| total wall time | 71.36 s | 997.06 s | +1297.1% |
| requests completed | 64 | 64 | 0.0% |
| input tokens (sum) | 390265 | 390265 | 0.0% |
| output tokens (sum) | 8192 | 8192 | 0.0% |
| request throughput | 0.897 req/s | 0.064 req/s | -92.8% |
| output token throughput | 114.79 tok/s | 8.216 tok/s | -92.8% |
| total token throughput | 5,583.4 tok/s | 399.63 tok/s | -92.8% |
| mean TTFT | 911.59 ms | 22,812.7 ms | +2402.5% |
| median TTFT | 888.83 ms | 22,837.2 ms | +2469.4% |
| p99 TTFT | 1,735.5 ms | 39,956.0 ms | +2202.3% |
| mean ITL | 10.38 ms | 65.72 ms | +533.3% |
| median ITL | 9.645 ms | 57.44 ms | +495.5% |
| p99 ITL | 10.95 ms | 66.70 ms | +509.4% |
| mean E2E latency | — ms | — ms | — |
| median E2E latency | — ms | — ms | — |
| p99 E2E latency | — ms | — ms | — |

## Output quality (seeded, temp=0 where set)

- Requests compared: 64
- Exact-match rate: 15.6%
- Mean token-overlap (Jaccard): 56.5%
- Median token-overlap: 57.6%
- Worst token-overlap: 5.6%

### Largest divergences (lowest token overlap)

#### Request 0

_Baseline_:

```
user
Thanks! Now I want to change the delivery address for this order. The new address is 456 Oak Avenue, Bpt 7C, Springfield, USA, IL, 62704.
assistant
<think>

</think>

I understand you want to change the delivery address for order yY5CH3V. The new address is:

- Line1: 456 Oak Avenue
- Line2: Bp
```

_Treatment_:

```
assistant
<think>

</think>

[{"function": {"arguments": "{\"account_key\": \"BERuCRx\", \"line1\": \"456 Oak Avenue\", \"line2\": \"Bpt 7C\", \"municipality\": \"Springfield\", \"region\": \"IL\", \"nation\": \"USA\", \"postal_code\": \"62704\"}", "name": "modify_account_location"}, "id": "call_0e9
```

#### Request 30

_Baseline_:

```
user
Yes, I'd like to know more about the repayment history.
assistant
<think>

</think>

[{"function": {"arguments": "{\"account_id\": \"ACC-UMRTMR8-1\"}", "limit": 10, "name": "get_recent_transactions"}, "id": "call_71c84425742441f69b88a936", "type": "function"}]
user
<tool_response>
[{"transactio
```

_Treatment_:

```
user
No, I'm just looking for the confirmation.
assistant
<think>

</think>

I'm glad to confirm that your auto loan payment was posted successfully on August 15th. If you need any further assistance or have questions about your account, cards, or beneficiaries, feel free to ask.
user

assistant
<th
```

#### Request 11

_Baseline_:

```
user
Yes, that's good. My account key is acct_123. Now I want to see my profile and the details of my recent order.
assistant
<think>

</think>

[{"function": {"arguments": "{\"account_key\": \"acct_123\"}", "name": "get_account_details"}, "id": "call_7896543210987654", "type": "function"}]
user
Gre
```

_Treatment_:

```
user
I'm in a rush prepping for a school event, so I just need quick answers. Can you tell me which card was charged, if I have a tracking number yet, and what product category those bags are listed under? My email is jamie.lee@example.com.
assistant
<think>

</think>

<tool_call>
<function=find_acc
```
