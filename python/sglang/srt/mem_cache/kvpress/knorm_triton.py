# SPDX-License-Identifier: Apache-2.0
"""Fused Triton kernel for knorm scoring (optimization g).

Replaces the stacked-gather + per-layer Python loop with a single kernel that reads
the per-layer K buffer pointers directly and emits the layer-summed -||k|| score per
token, no intermediate [L, T, H, D] tensor materialization. Same kept-set as
KnormPress.score_batched (no negation, NV "higher = keep" convention).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _knorm_fused_kernel(
    K_ptrs,           # [L] uint64 — per-layer pointer to k_buffer[l] data
    slots_ptr,        # [num_valid] int64 — pool slot ids
    out_ptr,          # [num_valid] float32 — scores
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_l: tl.constexpr,   # k_buffer[l] stride along slots (= H * D elements)
    stride_h: tl.constexpr,   # stride along heads (= D elements)
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.load(slots_ptr + t).to(tl.int64)
    score = 0.0
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    for l in range(L):
        base = tl.load(K_ptrs + l).to(tl.pointer_type(tl.float32))
        # base points to k_buffer[l][0,0,0] as float32; for bf16 store cast in caller.
        for h in range(H):
            ptr = base + s * stride_l + h * stride_h
            k = tl.load(ptr + offs_d, mask=mask_d, other=0.0)
            sq = tl.sum(k * k)
            score += tl.sqrt(sq)
    # Average over heads (matches KnormPress.score_batched: mean over heads, sum over layers).
    # Sign: NV "higher = keep" => keys with SMALL norm survive => negate the accumulated norm.
    tl.store(out_ptr + t, -score / H)


def knorm_fused_score(
    k_buffers: list[torch.Tensor],   # list of [num_slots, H, D] per layer; SAME dtype
    valid_slots: torch.Tensor,       # [num_valid] long
) -> torch.Tensor:
    """One-kernel knorm score, returns [num_valid] float32 (layer-summed -||k|| / H)."""
    assert len(k_buffers) > 0
    L = len(k_buffers)
    H, D = k_buffers[0].shape[1], k_buffers[0].shape[2]
    device = k_buffers[0].device
    num_valid = valid_slots.numel()

    # Cast K to float32 once if not already — Triton path is fp32 for stability.
    if k_buffers[0].dtype != torch.float32:
        kf = [kb.float() for kb in k_buffers]
    else:
        kf = k_buffers
    # Pointer table
    K_ptrs = torch.tensor([kb.data_ptr() for kb in kf], dtype=torch.uint64, device=device)
    slots = valid_slots.to(torch.int64)
    out = torch.empty(num_valid, dtype=torch.float32, device=device)
    BLOCK_D = max(16, triton.next_power_of_2(D))
    _knorm_fused_kernel[(num_valid,)](
        K_ptrs, slots, out, L=L, H=H, D=D,
        stride_l=H * D, stride_h=D, BLOCK_D=BLOCK_D,
    )
    return out
