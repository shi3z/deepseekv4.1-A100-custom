"""Static read-only expert replicas for the EP decode runtime (Phase 2 of the replica experiment).

Canonical expert ownership (the EP shards, dsv41/load.py) is unchanged. On top of it every GPU may hold extra
copies of experts of the layers it OWNS (the layers whose attention/gate it runs): a routed expert that has a
local replica is computed by the owner itself instead of by the shard's GPU, so the peer holding the canonical
copy skips it. The copies are the FP4 shard tensors byte for byte (no dequantisation), copied GPU to GPU once at
load time; no weight ever moves during decode.

Selection is static: a plan JSON written by dsv41/replica_plan.py (per GPU a list of [layer, expert]), optionally
capped to a byte budget. Enabled with DSV41_EXPERT_REPLICA_CACHE=1 (DSV41_REPLICA_PLAN=<json>,
DSV41_REPLICA_GIB=<per-GPU cap>); the baseline path is untouched when the variable is unset.

Dispatch contract (see EPRuntime._experts_shard / _owner_layer / _peer_layer):
  owner of layer L:  own-shard experts as before + replica experts through the same grouped FP4 kernels
                     (ids remapped by rep_lut[L]: global id -> replica slot or -1)
  peer p of layer L: ids remapped by peer_keep[L]: an expert replicated on L's owner becomes -1 (skipped, its rows
                     zeroed), so nothing is computed twice
  optional (DSV41_REPLICA_SKIP_WAIT=1): the owner waits only for the peers that hold a routed, non-replicated
                     expert of this token (device-computed mask; the rows of the others are zeroed), so a layer
                     whose remote experts are all replicated never blocks on the peers
Counters (int64 on the device, written inside the CUDA graphs): replica hits, remote misses, layer-tokens fully
local; replica_vram_bytes is static."""
from __future__ import annotations

import json
import os

import torch


class ReplicaCache:
    def __init__(self, model, rt, plan_path: str, budget_gib: float | None = None):
        self.model = model
        self.rt = rt
        blocks = model.blocks
        nl = len(blocks)
        E = model.args.n_routed_experts
        self.E = E
        self.nl = nl
        if not plan_path or not os.path.exists(plan_path):
            raise FileNotFoundError(f"DSV41_REPLICA_PLAN: no plan file {plan_path!r} (write one with dsv41.replica_plan --plan)")
        plan = json.load(open(plan_path))
        bpe = int(plan.get("bytes_per_expert", 0))
        self.plan_path = plan_path
        owner_of = {L: blocks[L].device for L in range(nl)}
        # shard tensors by (device, layer) and the shard bounds
        shard = {sh["device"]: (sh["start"], sh["n"]) for sh in blocks[0].ffn.ep}
        # per (owner device, layer): list of expert ids to replicate there
        want: dict[tuple[torch.device, int], list[int]] = {}
        for g, lst in plan["replicas"].items():
            dev = torch.device(f"cuda:{int(g)}")
            slots = int(budget_gib * 2**30 // bpe) if (budget_gib and bpe) else None
            n_take = 0
            for L, e in lst:
                if slots is not None and n_take >= slots:
                    break
                L, e = int(L), int(e)
                assert owner_of[L] == dev, f"plan replicates L{L} E{e} on {dev} which is not the layer's owner {owner_of[L]}"
                s, n = shard[dev]
                assert not (s <= e < s + n), f"plan replicates L{L} E{e} on its own shard {dev}"
                want.setdefault((dev, L), []).append(e)
                n_take += 1
        self.rep: dict[tuple[torch.device, int], dict] = {}
        self.peer_keep: dict[tuple[torch.device, int], torch.Tensor] = {}
        self.need_lut: dict[tuple[torch.device, int], torch.Tensor] = {}
        self.vram_bytes: dict[torch.device, int] = {d: 0 for d in rt.devs}
        self.n_replicas: dict[torch.device, int] = {d: 0 for d in rt.devs}
        for (dev, L), ids in want.items():
            moe = blocks[L].ffn
            R = len(ids)
            src_sh = {sh["device"]: sh for sh in moe.ep}
            proto = src_sh[dev]
            with torch.cuda.device(dev):
                t = {k: torch.empty(R, *proto[k].shape[1:], dtype=torch.uint8, device=dev) for k in ("w13", "s13", "w2", "s2")}
            for j, e in enumerate(ids):
                src = next(sh for sh in moe.ep if sh["start"] <= e < sh["start"] + sh["n"])
                le = e - src["start"]
                for k in ("w13", "s13", "w2", "s2"):
                    t[k][j].copy_(src[k][le], non_blocking=True)  # GPU -> GPU, once, at load
            lut = torch.full((E,), -1, dtype=torch.int32)
            for j, e in enumerate(ids):
                lut[e] = j
            t["lut"] = lut.to(dev)
            t["ids"] = list(ids)
            self.rep[(dev, L)] = t
            self.vram_bytes[dev] += sum(int(t[k].numel()) for k in ("w13", "s13", "w2", "s2"))
            self.n_replicas[dev] += R
            keep = torch.arange(E, dtype=torch.int32)
            keep[torch.tensor(ids)] = -1
            for p in rt.devs:
                if p != dev:
                    self.peer_keep[(p, L)] = keep.to(p)
        for d in rt.devs:
            torch.cuda.synchronize(d)
        # need_lut[(owner, L)][e] = slot index (rt.idx) of the peer that must compute e for this owner, or -1
        for L in range(nl):
            dev = owner_of[L]
            lut = torch.full((E,), -1, dtype=torch.int32)
            for p, (s, n) in shard.items():
                if p != dev:
                    lut[s:s + n] = rt.idx[p]
            r = self.rep.get((dev, L))
            if r is not None:
                lut[torch.tensor(r["ids"])] = -1
            self.need_lut[(dev, L)] = lut.to(dev)
        # counters (device, in-graph)
        self.hits = {d: torch.zeros(nl, dtype=torch.int64, device=d) for d in rt.devs}
        self.misses = {d: torch.zeros(nl, dtype=torch.int64, device=d) for d in rt.devs}
        self.full_local = {d: torch.zeros(nl, dtype=torch.int64, device=d) for d in rt.devs}
        self.layer_tokens = {d: torch.zeros(nl, dtype=torch.int64, device=d) for d in rt.devs}
        self.ones = {d: torch.ones(1, dtype=torch.int64, device=d) for d in rt.devs}
        print("[replica] " + ", ".join(f"{d}: {self.n_replicas[d]} experts / {self.vram_bytes[d] / 2**30:.2f} GiB" for d in rt.devs)
              + f" (plan {plan_path}, cap {budget_gib} GiB)", flush=True)

    def reset_counters(self):
        for d in self.rt.devs:
            self.hits[d].zero_(); self.misses[d].zero_(); self.full_local[d].zero_(); self.layer_tokens[d].zero_()

    def report(self) -> dict:
        hits = sum(int(self.hits[d].sum()) for d in self.rt.devs)
        misses = sum(int(self.misses[d].sum()) for d in self.rt.devs)
        full = sum(int(self.full_local[d].sum()) for d in self.rt.devs)
        lt = sum(int(self.layer_tokens[d].sum()) for d in self.rt.devs)
        return {
            "replica_hits": hits, "replica_misses": misses,
            "replica_hit_rate": hits / max(hits + misses, 1),
            "remote_routes_avoided": hits,
            "layer_tokens_fully_local": full, "layer_tokens": lt, "fully_local_rate": full / max(lt, 1),
            "replica_vram_bytes": {str(d): int(v) for d, v in self.vram_bytes.items()},
            "replicas_per_gpu": {str(d): int(v) for d, v in self.n_replicas.items()},
        }
