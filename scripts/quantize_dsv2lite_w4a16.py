#!/usr/bin/env python
# Copyright (c) 2026 LightSeek Foundation. MIT-style license (see repo headers).
"""Quantize DeepSeek-V2-Lite's ROUTED experts to compressed-tensors W4A16.

Produces a small compressed-tensors ``pack-quantized`` (num_bits=4, group_size=32,
symmetric) checkpoint -- only ``mlp.experts.*.{gate,up,down}_proj`` are quantized;
everything else stays bf16 -- mirroring Kimi-K2.6. Used to validate the AMD
``Wna16DequantBackend`` end-to-end against the known bf16 baseline.

Env: SRC = source HF snapshot dir, DST = output dir, GROUP_SIZE (default 32).
Pack format matches ``w4a16_dequant.unpack_w4a16``: uint4b8 (signed+8), 8 nibbles
per int32 low-first, packed along the input dim; weight_packed is [out, in//8].
"""

import glob
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

SRC = os.environ["SRC"]
DST = os.environ["DST"]
GROUP = int(os.environ.get("GROUP_SIZE", "32"))
SHARD_BYTES = 5 * (1 << 30)


def quantize_w4a16(w: torch.Tensor):
    """[out, in] bf16 -> (packed[out, in//8] int32, scale[out, in//GROUP])."""
    out, inf = w.shape
    assert inf % GROUP == 0, (inf, GROUP)
    ng = inf // GROUP
    wf = w.float().reshape(out, ng, GROUP)
    scale = (wf.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-8)
    q = torch.round(wf / scale).clamp(-8, 7).reshape(out, inf)
    u = (q + 8).to(torch.int32).reshape(out, inf // 8, 8)
    shifts = torch.arange(8, dtype=torch.int32) * 4
    packed = (u << shifts).sum(dim=-1).to(torch.int32)
    return packed, scale.reshape(out, ng).to(w.dtype)


def is_routed_expert(key: str) -> bool:
    return (
        ".mlp.experts." in key
        and key.endswith("_proj.weight")
        and any(p in key for p in ("gate_proj", "up_proj", "down_proj"))
    )


def main():
    os.makedirs(DST, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(SRC, "*.safetensors")))
    assert shards, f"no safetensors in {SRC}"
    print(f"loading {len(shards)} shards from {SRC}", flush=True)

    new_state = {}
    n_quant = 0
    sample = None
    for sh in shards:
        sd = load_file(sh)
        for k, v in sd.items():
            if is_routed_expert(k):
                packed, scale = quantize_w4a16(v)
                base = k[: -len(".weight")]
                new_state[base + ".weight_packed"] = packed
                new_state[base + ".weight_scale"] = scale
                new_state[base + ".weight_shape"] = torch.tensor(
                    list(v.shape), dtype=torch.int32
                )
                n_quant += 1
                if sample is None:
                    sample = (v.float().clone(), packed, scale)
            else:
                new_state[k] = v
        del sd
    print(f"quantized {n_quant} routed-expert projections", flush=True)

    # Self-check: our dequant must reconstruct the original within int4 group error.
    import sys

    sys.path.insert(0, os.path.join(os.environ["CODE"], "python"))
    from tokenspeed.runtime.layers.quantization.compressed_tensors.w4a16_dequant import (
        dequantize_w4a16,
    )

    w_orig, packed, scale = sample
    w_dq = dequantize_w4a16(packed, scale, GROUP, out_dtype=torch.float32)
    err = (w_dq - w_orig).abs().mean().item()
    print(f"SELFCHECK dequant mean-abs-err={err:.4f} (expect <0.15)", flush=True)
    assert err < 0.15, "pack/dequant format mismatch!"

    # Shard + save.
    os.makedirs(DST, exist_ok=True)
    shard_idx, cur, cur_bytes, weight_map = 1, {}, 0, {}
    items = list(new_state.items())

    def flush(idx, d):
        name = f"model-{idx:05d}.safetensors"
        save_file({k: v.contiguous() for k, v in d.items()}, os.path.join(DST, name))
        for k in d:
            weight_map[k] = name
        return name

    for k, v in items:
        nb = v.numel() * v.element_size()
        if cur_bytes + nb > SHARD_BYTES and cur:
            flush(shard_idx, cur)
            shard_idx += 1
            cur, cur_bytes = {}, 0
        cur[k] = v
        cur_bytes += nb
    if cur:
        flush(shard_idx, cur)
    total = shard_idx
    # rename to of-{total} convention
    for i in range(1, total + 1):
        src = os.path.join(DST, f"model-{i:05d}.safetensors")
        dstn = f"model-{i:05d}-of-{total:05d}.safetensors"
        os.rename(src, os.path.join(DST, dstn))
        for k in list(weight_map):
            if weight_map[k] == f"model-{i:05d}.safetensors":
                weight_map[k] = dstn
    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {"total_size": 0}, "weight_map": weight_map}, f, indent=1
        )
    print(f"saved {total} shards to {DST}", flush=True)

    # Copy aux files (tokenizer, *.py, generation_config) and patch config.json.
    for fn in os.listdir(SRC):
        if fn.endswith(".safetensors") or fn.endswith(".index.json"):
            continue
        s = os.path.join(SRC, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(DST, fn))
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    cfg["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": None,
                "output_activations": None,
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": GROUP,
                    "dynamic": False,
                    "actorder": None,
                    "block_structure": None,
                    "observer": "minmax",
                    "observer_kwargs": {},
                },
            }
        },
        "format": "pack-quantized",
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "kv_cache_scheme": None,
        # Ignore every Linear EXCEPT the routed experts (mlp.experts.*.*_proj).
        # The dense MLP uses the FUSED gate_up_proj module, so it must be listed
        # explicitly; mlp.gate is the router. Expert keys contain ".experts." and
        # do not match these (so they stay quantized).
        "ignore": [
            "lm_head",
            "re:.*self_attn.*",
            "re:.*shared_experts.*",
            "re:.*\\.mlp\\.gate_up_proj$",
            "re:.*\\.mlp\\.gate_proj$",
            "re:.*\\.mlp\\.up_proj$",
            "re:.*\\.mlp\\.down_proj$",
            "re:.*\\.mlp\\.gate$",
        ],
    }
    with open(os.path.join(DST, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote config.json with quantization_config; DONE", flush=True)


if __name__ == "__main__":
    main()
