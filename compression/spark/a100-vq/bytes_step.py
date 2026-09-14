"""Bytes the decode step reads per group, per configuration.

Parameter counts come from the safetensors header (shapes, not the tensors), so this costs a
few MB of reads and can run next to a benchmark. FP8 groups carry a UE8M0 scale per 32x32
block; FP4 halves the weight plane and keeps the same scale plane.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from collections import defaultdict

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
idx = os.path.join(md, "model.safetensors.index.json")
wmap = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
shards = sorted(set(wmap.values())) if wmap else [f for f in os.listdir(md) if f.endswith(".safetensors")]

shapes = {}
for sh in shards:
    with open(os.path.join(md, sh), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    for k, v in hdr.items():
        if k != "__metadata__":
            shapes[k] = (v["dtype"], v["shape"])


def group(name: str) -> str:
    if ".experts." in name and "shared" not in name:
        return "routed_experts"
    for pat in ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b"):
        if pat in name:
            return "attn"
    if "attn.wo_a" in name:
        return "wo_a"
    if "shared_experts" in name:
        return "shared"
    if name.startswith("lm_head") or "lm_head" in name:
        return "lm_head"
    if "embed" in name:
        return "embed"
    return "other"


n_el = defaultdict(int)
n_sc = defaultdict(int)
for k, (dt, sp) in shapes.items():
    el = 1
    for d in sp:
        el *= d
    (n_sc if k.endswith(".scale") else n_el)[group(k)] += el

print(f"{'group':16s} {'weight elems':>15s} {'scale elems':>13s} {'FP8 MB':>9s} {'FP4 MB':>9s} {'saved MB':>9s}")
tot = defaultdict(float)
for g in ("attn", "wo_a", "shared", "routed_experts", "lm_head", "embed", "other"):
    w, s = n_el[g], n_sc[g]
    fp8 = (w + s * 4) / 2**20
    fp4 = (w / 2 + s * 4) / 2**20
    tot["fp8"] += fp8; tot["fp4"] += fp4
    print(f"{g:16s} {w:15,d} {s:13,d} {fp8:9.1f} {fp4:9.1f} {fp8 - fp4:9.1f}")
print(f"{'TOTAL':16s} {'':15s} {'':13s} {tot['fp8']:9.1f} {tot['fp4']:9.1f} {tot['fp8']-tot['fp4']:9.1f}")
