# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-(layer, seq) ring buffer of uncompressed recent K/V.

Used alongside the turboquant packed cache so the newest N tokens — where
attention mass concentrates — are available at decode time without going
through the (slightly lossy) dequantize path.

The ring is position-based, not time-based: slot `(count % capacity)` is
the next write target. After the ring fills, writes overwrite the oldest
entry. `gather_recent` returns the entries in sequence-position order,
regardless of the physical slot layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RecentRing:
    """Per-layer uncompressed bf16 K/V ring buffer.

    Shapes (one layer, all sequences):
        keys:        (num_seqs, capacity, num_kv_heads, head_dim)  bf16
        values:      (num_seqs, capacity, num_kv_heads, head_dim)  bf16
        write_ptr:   (num_seqs,) int32 — next slot to overwrite.
        fill_count:  (num_seqs,) int32 — populated slots [0, capacity].
        total_appends: (num_seqs,) int32 — cumulative appends
                                          (= min(seq_len, ring has been populated to)).
    """

    keys: torch.Tensor
    values: torch.Tensor
    write_ptr: torch.Tensor
    fill_count: torch.Tensor
    total_appends: torch.Tensor

    @property
    def capacity(self) -> int:
        return self.keys.shape[1]

    @property
    def num_seqs(self) -> int:
        return self.keys.shape[0]


def allocate_recent_ring(
    *,
    num_seqs: int,
    capacity: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> RecentRing:
    if capacity <= 0:
        raise ValueError(f"ring capacity must be > 0; got {capacity}")
    shape = (num_seqs, capacity, num_kv_heads, head_dim)
    return RecentRing(
        keys=torch.zeros(shape, dtype=dtype, device=device),
        values=torch.zeros(shape, dtype=dtype, device=device),
        write_ptr=torch.zeros(num_seqs, dtype=torch.int32, device=device),
        fill_count=torch.zeros(num_seqs, dtype=torch.int32, device=device),
        total_appends=torch.zeros(num_seqs, dtype=torch.int32, device=device),
    )


def append_recent(
    ring: RecentRing,
    *,
    seq_id: int,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Append one token's K/V (shape: (num_kv_heads, head_dim)) for one seq.

    Overwrites the oldest slot when the ring is full.
    """
    assert key.shape == value.shape
    capacity = ring.capacity
    ptr = int(ring.write_ptr[seq_id].item())
    ring.keys[seq_id, ptr].copy_(key)
    ring.values[seq_id, ptr].copy_(value)
    ring.write_ptr[seq_id] = (ptr + 1) % capacity
    fc = int(ring.fill_count[seq_id].item())
    if fc < capacity:
        ring.fill_count[seq_id] = fc + 1
    ring.total_appends[seq_id] = ring.total_appends[seq_id] + 1


def write_prefill_tail(
    ring: RecentRing,
    *,
    seq_id: int,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Seed the ring from a prefill of length L by keeping the last min(L, capacity) tokens.

    Shapes: (L, num_kv_heads, head_dim) for keys and values.
    """
    assert keys.shape == values.shape
    assert keys.dim() == 3
    L = keys.shape[0]
    capacity = ring.capacity
    n = min(L, capacity)
    if n == 0:
        ring.write_ptr[seq_id] = 0
        ring.fill_count[seq_id] = 0
        ring.total_appends[seq_id] = L
        return
    ring.keys[seq_id, :n].copy_(keys[L - n : L])
    ring.values[seq_id, :n].copy_(values[L - n : L])
    ring.write_ptr[seq_id] = n % capacity
    ring.fill_count[seq_id] = n
    ring.total_appends[seq_id] = L


def gather_recent(
    ring: RecentRing,
    *,
    seq_id: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return the populated ring entries for one seq in insertion order.

    Returns:
        keys:    (n, num_kv_heads, head_dim) bf16
        values:  (n, num_kv_heads, head_dim) bf16
        n:       number of populated entries (0..capacity)
    """
    n = int(ring.fill_count[seq_id].item())
    capacity = ring.capacity
    if n == 0:
        empty = ring.keys.new_zeros((0, ring.keys.shape[2], ring.keys.shape[3]))
        return empty, empty, 0
    if n < capacity:
        return ring.keys[seq_id, :n], ring.values[seq_id, :n], n
    # Ring full — oldest entry is at write_ptr.
    ptr = int(ring.write_ptr[seq_id].item())
    if ptr == 0:
        return ring.keys[seq_id], ring.values[seq_id], capacity
    keys_ordered = torch.cat(
        (ring.keys[seq_id, ptr:], ring.keys[seq_id, :ptr]), dim=0
    )
    values_ordered = torch.cat(
        (ring.values[seq_id, ptr:], ring.values[seq_id, :ptr]), dim=0
    )
    return keys_ordered, values_ordered, capacity
