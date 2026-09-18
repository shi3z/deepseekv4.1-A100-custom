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


def plan_placement(
    n_layers: int,
    devices: list[int],
    budgets_gb: dict[int, float] | None = None,
    offload: bool = False,
) -> list[torch.device]:
    """Capacity-aware balanced layer placement.

    The old implementation greedily filled devices in order, which meant
    later devices were never used if earlier GPUs could already fit all
    layers.

    This implementation:
      1. computes how many layers each GPU can safely hold,
      2. distributes layers across all requested GPUs,
      3. never exceeds a GPU's calculated capacity.

    --budgets can still be used to reserve extra memory on selected GPUs.
    """

    free: dict[int, float] = {}

    for d in devices:
        if budgets_gb and d in budgets_gb:
            free[d] = budgets_gb[d]
        else:
            f, _ = torch.cuda.mem_get_info(d)
            free[d] = f / 2**30

    layer_gb = LAYER_GB_OFFLOAD if offload else LAYER_GB

    if offload and HOT_EXPERTS:
        layer_gb += (
            18.8e6
            * (max(len(v) for v in HOT_EXPERTS.values()) + 1)
            / 2**30
        )

    reserve = RESERVE_GB_OFFLOAD if offload else RESERVE_GB

    capacity: dict[int, int] = {}

    for d in devices:
        extra = 2.6 if d == devices[0] else 0.0

        capacity[d] = int(
            max(
                0.0,
                free[d] - reserve - extra,
            )
            // layer_gb
        )

    if sum(capacity.values()) < n_layers:
        raise RuntimeError(
            f"not enough GPU memory: capacity={capacity}, "
            f"need={n_layers}, free={free}"
        )

    # Allocate layer counts as evenly as possible while respecting
    # per-device capacities.
    counts = {d: 0 for d in devices}

    remaining = n_layers

    while remaining > 0:
        candidates = [
            d
            for d in devices
            if counts[d] < capacity[d]
        ]

        if not candidates:
            raise RuntimeError(
                f"placement exhausted unexpectedly: "
                f"counts={counts} capacity={capacity}"
            )

        # Prefer the GPU with the lowest fraction of its capacity used.
        #
        # Tie-break by the original --devices ordering so placement stays
        # deterministic.
        d = min(
            candidates,
            key=lambda dev: (
                counts[dev] / max(capacity[dev], 1),
                devices.index(dev),
            ),
        )

        counts[d] += 1
        remaining -= 1

    # Preserve pipeline locality: each GPU gets one contiguous block.
    placement: list[torch.device] = []

    for d in devices:
        placement.extend(
            [torch.device(f"cuda:{d}")] * counts[d]
        )

    print(
        "placement capacity:",
        {
            f"cuda:{d}": {
                "layers": counts[d],
                "capacity": capacity[d],
                "budget_gb": round(free[d], 2),
            }
            for d in devices
        },
        flush=True,
    )

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
        if getattr(device, "type", None) == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
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
            if getattr(sh["device"], "type", None) == "cuda" or (isinstance(sh["device"], str) and sh["device"].startswith("cuda")):
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
    if getattr(device, "type", None) == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
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
        # Optional pipeline rebalance for long-context cache headroom.
        # Counts are in device order and must sum to n_layers; this lets a
        # GPU with less free VRAM host fewer dense layers while preserving
        # sequential pipeline order.
        _counts = os.environ.get("DSV41_LAYER_COUNTS", "")
        if _counts:
            counts = [int(x) for x in _counts.split(",") if x.strip()]
            if len(counts) != nd or sum(counts) != n_layers or any(x <= 0 for x in counts):
                raise ValueError(f"DSV41_LAYER_COUNTS must be {nd} positive counts summing to {n_layers}: {_counts!r}")
            placement = []
            for dev, count in zip(devices, counts):
                placement.extend([torch.device(f"cuda:{dev}")] * count)
            print(f"layer counts override: {counts}", flush=True)
        else:
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
    # -----------------------------------------------------------------
    # Dynamically sized FULL-MIRROR compressed caches.
    #
    # max_seq_len remains the logical context limit, but we do NOT reserve
    # max_seq_len worth of compressed KV/index cache at startup.
    #
    # Default initial capacity corresponds to 32K logical tokens.
    #
    # Override, for example:
    #
    #   DSV41_CACHE_INIT_TOKENS=65536
    #
    # Mirrors still exist on every participating device.  This deliberately
    # avoids the unsafe consumer-local placement experiment.
    # -----------------------------------------------------------------

    cache_init_tokens = int(
        os.environ.get("DSV41_CACHE_INIT_TOKENS", "131072")
    )

    cache_init_tokens = max(
        1024,
        min(cache_init_tokens, max_seq_len),
    )

    cache_devices = list(dict.fromkeys(placement))

    print(
        f"dynamic cache: initial logical capacity "
        f"{cache_init_tokens:,}/{max_seq_len:,} tokens; "
        f"full mirrors on {[str(d) for d in cache_devices]}",
        flush=True,
    )

    cache_initial_bytes = 0
    cache_max_bytes = 0

    for owner in cfg["kv_source_layers"]:
        if owner >= n_layers:
            continue

        ratio = cfg["compress_ratios"][owner]

        max_rows = (
            max_seq_len // ratio + 1
        )

        initial_rows = min(
            max_rows,
            cache_init_tokens // ratio + 1,
        )

        model.shared.cache_max_rows[owner] = max_rows

        one_initial = (
            max_seqs
            * initial_rows
            * (
                cfg["head_dim"]
                + cfg["index_head_dim"]
            )
            * 2
        )

        one_max = (
            max_seqs
            * max_rows
            * (
                cfg["head_dim"]
                + cfg["index_head_dim"]
            )
            * 2
        )

        for d in cache_devices:
            model.shared.compress_kv[(owner, d)] = torch.empty(
                max_seqs,
                initial_rows,
                cfg["head_dim"],
                dtype=torch.bfloat16,
                device=d,
            )

            model.shared.index_k[(owner, d)] = torch.empty(
                max_seqs,
                initial_rows,
                cfg["index_head_dim"],
                dtype=torch.bfloat16,
                device=d,
            )

            cache_initial_bytes += one_initial
            cache_max_bytes += one_max

        print(
            f"  cache owner {owner:2d}: "
            f"ratio={ratio} "
            f"rows={initial_rows:,}/{max_rows:,} "
            f"mirrors={len(cache_devices)}",
            flush=True,
        )

    print(
        f"compressed cache allocated at startup: "
        f"{cache_initial_bytes / 2**30:.2f} GiB "
        f"(1M full capacity would be "
        f"{cache_max_bytes / 2**30:.2f} GiB)",
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
