"""Checkpoint loading with layer placement across GPUs and Engram tables in host RAM."""
from __future__ import annotations

import json
import os
import time

import torch

from .engram import Engram, EngramLayout, HostEngramTable, NgramHashState
from .model import Args, Block, Transformer
from .w8 import ENABLED as W8_ENABLED, W8
from .quant import dequant_fp8_block
from .stio import Checkpoint

HOT_EXPERTS: dict[int, list[int]] = {}  # layer -> expert ids to keep on the GPU (cpu offload mode)
LAYER_GB = 7.1  # experts (6.72 GiB fp4 + scales) + dense bf16 (0.34 GiB) per layer
LAYER_GB_OFFLOAD = 0.25  # dense bf16 only; experts live in host RAM
RESERVE_GB = 2.0  # activations / temporaries per device
RESERVE_GB_OFFLOAD = 6.0  # + expert staging buffers (decode 1.8 GiB, prefill chunk 1.2 GiB) and prefill temporaries


def plan_placement(n_layers: int, devices: list[int], budgets_gb: dict[int, float] | None = None,
                   offload: bool = False) -> list[torch.device]:
    """Greedy: fill each device (in the given order) with as many layers as its free memory allows."""
    free = {}
    for d in devices:
        if budgets_gb and d in budgets_gb:
            free[d] = budgets_gb[d]
        else:
            f, _ = torch.cuda.mem_get_info(d)
            free[d] = f / 2**30
    placement: list[torch.device] = []
    layer_gb = LAYER_GB_OFFLOAD if offload else LAYER_GB
    if offload and HOT_EXPERTS:
        layer_gb += 18.8e6 * (max(len(v) for v in HOT_EXPERTS.values()) + 1) / 2**30
    reserve = RESERVE_GB_OFFLOAD if offload else RESERVE_GB
    for d in devices:
        n = int(max(0, free[d] - reserve - (2.6 if d == devices[0] else 0)) // layer_gb)
        for _ in range(n):
            if len(placement) < n_layers:
                placement.append(torch.device(f"cuda:{d}"))
    if len(placement) < n_layers:
        raise RuntimeError(f"not enough GPU memory: placed {len(placement)}/{n_layers} layers with free={free}")
    return placement


def _dense(ckpt: Checkpoint, name: str, device):
    """A Linear weight on device: FP8 block-scaled stays packed (W8, tensor-core GEMV) unless DSV41_W8=0
    (then dequantized to bf16); bf16 stays."""
    dtype, _ = ckpt.meta(name)
    w = ckpt.get(name, device)
    if dtype == "F8_E4M3":
        s = ckpt.get(name.replace(".weight", ".scale"), device)
        if W8_ENABLED:
            return W8(w.view(torch.uint8), s)
        return dequant_fp8_block(w, s)
    return w


def load_layer(ckpt: Checkpoint, i: int, device, offload=False, ep: list | None = None, prefix: str | None = None) -> dict:
    """ep: expert parallelism, a list of (device, first_expert, n_experts) shards; the dense part goes to `device`.
    prefix: checkpoint namespace (default layers.{i}.; the DSpark blocks live under mtp.{i}.)."""
    p = prefix if prefix is not None else f"layers.{i}."
    w: dict[str, torch.Tensor] = {}
    for n in ckpt.names(p):
        key = n[len(p):]
        if ".experts." in key or key.startswith("engram.embed") or key.endswith(".scale") or "bias_vl" in key:
            continue
        if key.endswith(".weight") and ckpt.meta(n)[0] in ("F8_E4M3", "BF16"):
            w[key] = _dense(ckpt, n, device)
        else:
            w[key] = ckpt.get(n, device)
    # experts, stacked: w13 = [w1; w3] along N
    e_names = sorted({int(n.split(".experts.")[1].split(".")[0]) for n in ckpt.names(p + "ffn.experts.")})
    E = len(e_names)
    inter, dimh = ckpt.meta(p + "ffn.experts.0.w1.weight")[1]
    dim = ckpt.meta(p + "ffn.experts.0.w2.weight")[1][0]
    if offload == "cpu":  # experts in NUMA-split host RAM, computed on the CPU (dsv41/cpu/moe_cpu.cpp)
        from .cpumoe import HostExperts
        host = HostExperts(E, inter, dim)
        cols = {k: [] for k in ("w1.weight", "w3.weight", "w1.scale", "w3.scale", "w2.weight", "w2.scale")}
        for e in e_names:
            q = f"{p}ffn.experts.{e}."
            for k in cols:
                cols[k].append(ckpt.get(q + k).view(torch.uint8))
        host.load_layer(cols["w1.weight"], cols["w3.weight"], cols["w1.scale"], cols["w3.scale"], cols["w2.weight"], cols["w2.scale"])
        if i == 0:
            print(f"  host experts NUMA-local fraction (layer 0): {host.local_fraction():.3f}", flush=True)
        w13, s13, w2, s2 = host.views()
        w.update({"experts.w13": w13, "experts.s13": s13, "experts.w2": w2, "experts.s2": s2, "experts.offload": True, "experts.host": host})
        hot = HOT_EXPERTS.get(i) if HOT_EXPERTS else None
        if hot:
            # GPU-resident copies of the most used experts of this layer; slot len(hot) is a zero dummy
            n = len(hot)
            g = lambda src: torch.zeros(n + 1, *src.shape[1:], dtype=torch.uint8, device=device)
            hw13, hs13, hw2, hs2 = g(w13), g(s13), g(w2), g(s2)
            for slot, e in enumerate(hot):
                hw13[slot].copy_(w13[e]); hs13[slot].copy_(s13[e]); hw2[slot].copy_(w2[e]); hs2[slot].copy_(s2[e])
            w["experts.hot"] = {"w13": hw13, "s13": hs13, "w2": hw2, "s2": hs2, "slot": {int(e): k for k, e in enumerate(hot)}, "dummy": n}
        torch.cuda.synchronize(device)
        return w
    if ep:
        shards = []
        for (sd, start, n) in ep:
            g = lambda *shape: torch.empty(*shape, dtype=torch.uint8, device=sd)
            sw13, ss13, sw2, ss2 = g(n, 2 * inter, dimh), g(n, 2 * inter, dimh * 2 // 32), g(n, dim, inter // 2), g(n, dim, inter // 32)
            for j in range(n):
                q = f"{p}ffn.experts.{start + j}."
                sw13[j, :inter].copy_(ckpt.get(q + "w1.weight").view(torch.uint8), non_blocking=True)
                sw13[j, inter:].copy_(ckpt.get(q + "w3.weight").view(torch.uint8), non_blocking=True)
                ss13[j, :inter].copy_(ckpt.get(q + "w1.scale"), non_blocking=True)
                ss13[j, inter:].copy_(ckpt.get(q + "w3.scale"), non_blocking=True)
                sw2[j].copy_(ckpt.get(q + "w2.weight").view(torch.uint8), non_blocking=True)
                ss2[j].copy_(ckpt.get(q + "w2.scale"), non_blocking=True)
            shards.append({"device": sd, "start": start, "n": n, "w13": sw13, "s13": ss13, "w2": sw2, "s2": ss2})
        for sh in shards:
            torch.cuda.synchronize(sh["device"])
        own = [sh for sh in shards if sh["device"] == device][0]
        w.update({"experts.w13": own["w13"], "experts.s13": own["s13"], "experts.w2": own["w2"], "experts.s2": own["s2"], "experts.offload": False, "experts.ep": shards})
        return w
    if offload:  # experts stay in page-locked host memory; the MoE streams the selected ones per token
        alloc = lambda *shape: torch.empty(*shape, dtype=torch.uint8, pin_memory=True)
    else:
        alloc = lambda *shape: torch.empty(*shape, dtype=torch.uint8, device=device)
    w13 = alloc(E, 2 * inter, dimh)
    s13 = alloc(E, 2 * inter, dimh * 2 // 32)
    w2 = alloc(E, dim, inter // 2)
    s2 = alloc(E, dim, inter // 32)
    for e in e_names:
        q = f"{p}ffn.experts.{e}."
        w13[e, :inter].copy_(ckpt.get(q + "w1.weight").view(torch.uint8), non_blocking=True)
        w13[e, inter:].copy_(ckpt.get(q + "w3.weight").view(torch.uint8), non_blocking=True)
        s13[e, :inter].copy_(ckpt.get(q + "w1.scale"), non_blocking=True)
        s13[e, inter:].copy_(ckpt.get(q + "w3.scale"), non_blocking=True)
        w2[e].copy_(ckpt.get(q + "w2.weight").view(torch.uint8), non_blocking=True)
        s2[e].copy_(ckpt.get(q + "w2.scale"), non_blocking=True)
    w.update({"experts.w13": w13, "experts.s13": s13, "experts.w2": w2, "experts.s2": s2, "experts.offload": offload})
    torch.cuda.synchronize(device)
    return w


def choose_hot_experts(stats_path: str, per_layer: int, n_layers: int) -> dict[int, list[int]]:
    """Top-`per_layer` experts of each layer by decode hit count (from --route-stats)."""
    st = torch.load(stats_path)
    return {l: st[l].topk(per_layer).indices.tolist() for l in range(n_layers) if l in st}


def load_model(ckpt_path: str, devices: list[int], max_seq_len: int = 16384, max_batch: int = 1, max_seqs: int | None = None,
               budgets_gb: dict[int, float] | None = None, n_layers: int | None = None, engram: bool = True,
               tokenizer=None, offload_experts=False, hot_experts: int = 0, route_stats: str = "", ep: bool = False,
               ep_shards: list[int] | None = None) -> Transformer:
    """ep: expert parallelism over `devices` (dense layers pipelined over them in order, every layer's experts
    sharded across all of them; see dsv41/ep.py). ep_shards: experts per device (default: equal split)."""
    global HOT_EXPERTS
    cfg = json.load(open(os.path.join(ckpt_path, "inference", "config.json")))
    if offload_experts == "cpu" and hot_experts > 0:
        stats = route_stats or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "route_stats.pt")
        if os.path.exists(stats):
            HOT_EXPERTS = choose_hot_experts(stats, hot_experts, n_layers or cfg["n_layers"])
        else:
            HOT_EXPERTS = {l: list(range(hot_experts)) for l in range(n_layers or cfg["n_layers"])}
        print(f"hot experts on GPU: {hot_experts}/layer ({'from ' + stats if os.path.exists(stats) else 'no stats: first N'})", flush=True)
    else:
        HOT_EXPERTS = {}
    args = Args(cfg, max_batch_size=max_batch, max_seq_len=max_seq_len, max_seqs=max_seqs or max_batch)
    max_seqs = max_seqs or max_batch
    ckpt = Checkpoint(ckpt_path)
    n_layers = n_layers or cfg["n_layers"]
    if ep:
        nd = len(devices)
        placement = [torch.device(f"cuda:{devices[min(i * nd // n_layers, nd - 1)]}") for i in range(n_layers)]
        E = cfg["n_routed_experts"]
        if ep_shards:
            assert len(ep_shards) == nd and sum(ep_shards) == E, (ep_shards, E)
            bounds = [sum(ep_shards[:j]) for j in range(nd + 1)]
        else:
            bounds = [E * j // nd for j in range(nd + 1)]
        ep_shards = [(torch.device(f"cuda:{devices[j]}"), bounds[j], bounds[j + 1] - bounds[j]) for j in range(nd)]
        print("expert shards:", {str(d): n for d, _, n in ep_shards}, flush=True)
    else:
        placement = plan_placement(n_layers, devices, budgets_gb, offload=offload_experts)
        ep_shards = None
    _host_layers = []
    print("placement:", {str(d): placement.count(d) for d in dict.fromkeys(placement)}, flush=True)
    model = Transformer(args)
    t0 = time.time()
    # Per-(owner layer, device) mirrors of compressed-KV/index-K.
    #
    # Do NOT mirror every owner onto every GPU.  An owner's cache is only
    # consumed by layers from that owner up to (but excluding) the next
    # KV source layer.  With long contexts, full mirroring wastes several
    # times the required VRAM.
    #
    # DSV41_FULL_CACHE_MIRROR=1 restores the historical behaviour.
    owners = [o for o in cfg["kv_source_layers"] if o < n_layers]
    all_devices = list(dict.fromkeys(placement))
    # Safe default: mirror caches on every device.
    #
    # Consumer-local cache placement is still experimental and can miss
    # valid cross-device accesses such as (owner_layer, cuda:0).
    #
    # Opt in explicitly with:
    #   DSV41_CONSUMER_LOCAL_CACHE=1
    consumer_local_cache = (
        os.environ.get("DSV41_CONSUMER_LOCAL_CACHE", "0") == "1"
    )

    cache_total_bytes = 0

    for oi, owner in enumerate(owners):
        ratio = cfg["compress_ratios"][owner]
        rows = max_seq_len // ratio + 1  # + dummy row for static decode

        next_owner = owners[oi + 1] if oi + 1 < len(owners) else n_layers

        if consumer_local_cache:
            # Experimental memory-saving mode.
            # May be unsafe if a later execution path accesses this owner's
            # cache from a device not represented in placement[owner:next_owner].
            consumer_devices = list(
                dict.fromkeys(placement[owner:next_owner])
            )
        else:
            # Safe/default mode.
            consumer_devices = all_devices

        if not consumer_devices:
            raise RuntimeError(
                f"no cache consumer device for owner layer {owner}"
            )

        one_mirror_bytes = (
            max_seqs
            * rows
            * (cfg["head_dim"] + cfg["index_head_dim"])
            * 2  # bf16
        )

        for d in consumer_devices:
            model.shared.compress_kv[(owner, d)] = torch.zeros(
                max_seqs,
                rows,
                cfg["head_dim"],
                dtype=torch.bfloat16,
                device=d,
            )

            model.shared.index_k[(owner, d)] = torch.zeros(
                max_seqs,
                rows,
                cfg["index_head_dim"],
                dtype=torch.bfloat16,
                device=d,
            )

            cache_total_bytes += one_mirror_bytes

        print(
            f"  cache owner {owner:2d}: rows={rows:,} "
            f"ratio={ratio} devices={[str(d) for d in consumer_devices]} "
            f"mirror={one_mirror_bytes / 2**30:.3f} GiB/device",
            flush=True,
        )

    print(
        f"shared compressed cache total: "
        f"{cache_total_bytes / 2**30:.3f} GiB "
        f"({'consumer-local mirrors' if consumer_local_cache else 'FULL MIRROR'})",
        flush=True,
    )
    for i in range(n_layers):
        dev = placement[i]
        w = load_layer(ckpt, i, dev, offload=offload_experts, ep=ep_shards)
        model.blocks.append(Block(args, i, w, dev, model.shared))
        del w
        print(f"  layer {i:2d} -> {dev}  ({time.time() - t0:5.0f}s)", flush=True)
    dev0, devL = placement[0], placement[n_layers - 1]
    model.embed = ckpt.get("embed.weight", dev0)
    model.head = ckpt.get("head.weight", devL)
    model.norm_w = ckpt.get("norm.weight", devL)
    if engram and cfg.get("engram_layer_ids"):
        layout = EngramLayout(cfg)
        assert tokenizer is not None, "the Engram hash needs the tokenizer"
        model.engram_hash = NgramHashState(cfg, layout, tokenizer, max_seqs, max_seq_len, dev0)
        for li, lid in enumerate(layout.layer_ids):
            if lid >= n_layers:
                continue
            p = f"layers.{lid}.engram."
            t1 = time.time()
            weight = ckpt.get(p + "embed.weight").view(torch.uint8).clone()  # into RAM (sequential read)
            scale = ckpt.get(p + "embed.scale").clone()
            print(f"  engram table layer {lid}: {weight.numel() / 2**30:.1f} GiB in host RAM ({time.time() - t1:.0f}s)", flush=True)
            dev = placement[lid]
            blk = model.blocks[lid]
            blk.engram = Engram(cfg["dim"], cfg["hc_mult"], layout, HostEngramTable(weight, scale),
                                _dense(ckpt, p + "wkv.weight", dev), ckpt.get(p + "q_weight", dev), ckpt.get(p + "k_weight", dev), cfg["norm_eps"])
            blk.engram.layer_hash_index = li
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    return model
