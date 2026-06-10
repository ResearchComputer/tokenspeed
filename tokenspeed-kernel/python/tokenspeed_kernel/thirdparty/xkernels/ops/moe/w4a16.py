# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Shared helpers for the INT4 W4A16 fused-MoE GEMM (compressed-tensors
``pack-quantized``): weight unpack/dequant, a random-weight generator for tests,
and the ``moe_align_block_size`` sort/pad dispatch builder reused by both the
reference and the Triton backend so they consume identical routing.
"""

from __future__ import annotations

import torch

__all__ = [
    "dequant_w4a16",
    "make_w4a16_weights",
    "moe_align_block_size_ref",
]


def dequant_w4a16(
    packed: torch.Tensor, scale: torch.Tensor, group_size: int = 32
) -> torch.Tensor:
    """Unpack + dequantize compressed-tensors W4A16 weights to bf16.

    Args:
        packed: ``[E, N, K // 8]`` int32 holding 8 ``uint4b8`` nibbles per int32,
            low nibble = lowest K index.
        scale: ``[E, N, K // group_size]`` group scales (bf16).
        group_size: quant group size along K.

    Returns:
        ``[E, N, K]`` bf16 dequantized weights, where
        ``W[e,n,k] = (((packed[e,n,k//8] >> (4*(k%8))) & 0xF) - 8) * scale[e,n,k//g]``.
    """
    E, N, kp = packed.shape
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    nib = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(E, N, kp * 8)  # [E,N,K]
    K = kp * 8
    w = (nib.float() - 8.0).view(E, N, K // group_size, group_size)
    w = w * scale.float().unsqueeze(-1)
    return w.view(E, N, K).to(torch.bfloat16)


def moe_align_block_size_ref(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """Sort/pad routed token-slots into per-expert blocks (issue #4 baseline).

    Reused so the optimized kernel and the reference consume identical dispatch.

    Returns ``(sorted_token_ids, expert_ids, num_tokens_post_padded)``.
    """
    flat_e = topk_ids.flatten().long()
    total = flat_e.numel()
    pad = total
    max_pad = total + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.full((max_pad,), pad, dtype=torch.int32, device=topk_ids.device)
    order = torch.argsort(flat_e, stable=True)
    toks = torch.arange(total, device=topk_ids.device, dtype=torch.int32)[order]
    counts = torch.bincount(flat_e, minlength=num_experts + 1)
    expert_ids = []
    w = off = 0
    for e in range(num_experts):
        n = int(counts[e])
        grp = toks[off : off + n]
        nb = (n + block_size - 1) // block_size
        for b in range(nb):
            c = min(block_size, n - b * block_size)
            sorted_ids[w + b * block_size : w + b * block_size + c] = grp[
                b * block_size : b * block_size + c
            ]
            expert_ids.append(e)
        w += nb * block_size
        off += n
    if not expert_ids:
        expert_ids = [0]
    return (
        sorted_ids,
        torch.tensor(expert_ids, dtype=torch.int32, device=topk_ids.device),
        torch.tensor([w], dtype=torch.int32, device=topk_ids.device),
    )


def make_w4a16_weights(E: int, N: int, K: int, group_size: int = 32, *, device="cuda", seed=0):
    """Generate random valid W4A16 packed weights + scales for testing.

    Returns ``(packed [E,N,K//8] int32, scale [E,N,K//g] bf16, w_ref [E,N,K] bf16)``
    where ``w_ref`` is the exact dequantization of ``(packed, scale)``.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    assert K % 8 == 0 and K % group_size == 0
    # Random signed nibbles in [-8, 7] -> uint4b8 in [0, 15].
    q = torch.randint(0, 16, (E, N, K), generator=g, device=device, dtype=torch.int32)
    shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
    packed = (q.view(E, N, K // 8, 8) << shifts).sum(dim=-1).to(torch.int32)  # [E,N,K//8]
    scale = (
        torch.rand((E, N, K // group_size), generator=g, device=device, dtype=torch.float32)
        * 0.02
        + 0.005
    ).to(torch.bfloat16)
    w_ref = dequant_w4a16(packed, scale, group_size)
    return packed, scale, w_ref
