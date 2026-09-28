"""Rank expert-replica candidates from saved routing profiles (dsv41/expert_profile.py output).

A routed expert is "remote" when the layer's owner GPU (the GPU that runs the layer's attention and gate) does
not hold it in its EP shard. A replica of expert (layer L, id E) is only useful on owner(L), so a candidate is
the triple (L, E, G=owner(L)); its memory cost is one FP4 expert (bytes_per_expert from the profile).

Reports (all from measured routing, nothing estimated as a speedup):
  * route_count[layer][source_gpu][expert] summary, remote vs local routes
  * top experts globally (per (layer, expert) and per expert id summed over layers)
  * top replica candidates per GPU by pure route frequency and by the cost-weighted score
        score(E, G) = remote_route_count(E, G) * bytes_saved_per_route * measured_P2P_cost
  * for replica budgets (GiB per GPU): share of remote routes that becomes local, and share of (token, layer)
    events whose remote experts are ALL replicated (the only case in which the owner could skip waiting for
    its peers)
  * routing locality: P(E_{L+1} | E_L) and P(E_{t+1, L} | E_{t, L}) top-k prediction quality (k = 1, 2, 4, 8)
  * an optional static replica plan (JSON: per GPU the list of (layer, expert)) for Phase 2

usage: python -m dsv41.replica_plan results/expert_profile_S1.pt [more.pt ...] [--budgets 1,2,4,8] [--plan out.json --plan-gib 4]
"""
from __future__ import annotations

import argparse, json, math, sys
from collections import defaultdict

import torch

ap = argparse.ArgumentParser()
ap.add_argument("profiles", nargs="+", help="profiles the coverage / locality numbers are evaluated on")
ap.add_argument("--fit", default="", help="comma-separated profiles the route counts, rankings and the plan are fitted on (default: the evaluated ones, i.e. in-sample)")
ap.add_argument("--budgets", default="1,2,4,8", help="GiB of replica cache per GPU")
ap.add_argument("--top", type=int, default=50)
ap.add_argument("--per-gpu", type=int, default=20)
ap.add_argument("--plan", default="", help="write a static replica plan JSON for this budget")
ap.add_argument("--plan-gib", type=float, default=4.0)
ap.add_argument("--holdout", type=float, default=0.3, help="fraction of steps held out for the locality evaluation")
a = ap.parse_args()

profs = [torch.load(p, weights_only=False) for p in a.profiles]
fit_profs = [torch.load(p, weights_only=False) for p in a.fit.split(",")] if a.fit else profs
p0 = profs[0]
nl, E, topk = p0["n_layers"], p0["n_experts"], p0["topk"]
owner = p0["owner_of_layer"]
shard = {int(g): tuple(v) for g, v in p0["shard_of_dev"].items()}
gpus = sorted(shard)
bpe = p0["bytes_per_expert"]
dim = p0["dim"]
for p in profs[1:]:
    assert p["owner_of_layer"] == owner and {int(g): tuple(v) for g, v in p["shard_of_dev"].items()} == shard, "profiles from different placements"

def is_local(L, e):
    s, n = shard[owner[L]]
    return s <= e < s + n

# ---------------------------------------------------------------- routing matrix
# route_count[L][G][E]: G is the source GPU (= owner of L); one 3-D tensor [nl, n_gpu, E]
gidx = {g: i for i, g in enumerate(gpus)}
route_count = torch.zeros(nl, len(gpus), E, dtype=torch.float64)
route_mass = torch.zeros(nl, len(gpus), E, dtype=torch.float64)
events = []  # per evaluated profile: eid [steps, nl, S, topk] for locality / coverage
for p in profs:
    events.append(p["eid"].long())
for p in fit_profs:
    eid = p["eid"].long()  # [steps, nl, S, topk]
    wt = p["wt"].double()
    for L in range(nl):
        flat = eid[:, L].reshape(-1)
        route_count[L, gidx[owner[L]]].index_add_(0, flat, torch.ones(flat.numel(), dtype=torch.float64))
        route_mass[L, gidx[owner[L]]].index_add_(0, flat, wt[:, L].reshape(-1))
counts = route_count.sum(1)  # [nl, E] (each layer has exactly one source GPU), from the fitted profiles
if a.fit:
    print(f"fitted on {a.fit}; evaluated on {a.profiles}")
# evaluation-side counts for the coverage numbers
ev_counts = torch.zeros(nl, E, dtype=torch.float64)
for eid in events:
    for L in range(nl):
        flat = eid[:, L].reshape(-1)
        ev_counts[L].index_add_(0, flat, torch.ones(flat.numel(), dtype=torch.float64))
local_mask = torch.zeros(nl, E, dtype=torch.bool)
for L in range(nl):
    s, n = shard[owner[L]]
    local_mask[L, s:s + n] = True
total_routes = counts.sum().item()
remote_routes = counts[~local_mask].sum().item()
print(f"profiles: {a.profiles}")
print(f"routes: {int(total_routes):,} total, {int(remote_routes):,} remote ({100 * remote_routes / total_routes:.1f}%), "
      f"{int(total_routes - remote_routes):,} local; layers {nl}, experts {E}, topk {topk}")
print("per-GPU (as source/owner): layers owned, remote share of its routes")
for g in gpus:
    Ls = [L for L in range(nl) if owner[L] == g]
    c = counts[Ls]
    m = local_mask[Ls]
    tot = c.sum().item()
    rem = c[~m].sum().item()
    print(f"  cuda:{g}: layers {Ls[0]}..{Ls[-1]} ({len(Ls)}), shard experts {shard[g][0]}..{shard[g][0] + shard[g][1] - 1}, "
          f"routes {int(tot):,}, remote {100 * rem / max(tot, 1):.1f}%")

# ---------------------------------------------------------------- measured dispatch costs (EP trace)
# stamps on the owner (its own clock): 0 layer start, 1 after attention, 8 after hc_sub2, 9 after gate, 2 after multicast issue,
# 3 after own experts + shared, 4 after wait partials, 5 after hc_post; peers: 6 route received, 7 partial pushed
cost = {}
if p0.get("trace") is not None:
    tr = torch.cat([p["trace"] for p in profs if p.get("trace") is not None]).double()  # [steps, nd, nl, 16]
    tdev = p0["trace_devs"]
    parts = defaultdict(list)
    for L in range(nl):
        o = tdev.index(owner[L])
        t = tr[:, o, L]
        parts["attention+hc"].append((t[:, 1] - t[:, 0]).mean().item())
        parts["gate+topk"].append((t[:, 9] - t[:, 8]).mean().item())
        parts["multicast issue"].append((t[:, 2] - t[:, 9]).mean().item())
        parts["own experts || shared"].append((t[:, 3] - t[:, 2]).mean().item())
        parts["wait partials"].append((t[:, 4] - t[:, 3]).mean().item())
        parts["hc_post"].append((t[:, 5] - t[:, 4]).mean().item())
        parts["layer"].append((t[:, 5] - t[:, 0]).mean().item())
        pc = [(tr[:, i, L, 7] - tr[:, i, L, 6]).mean().item() for i in range(len(tdev)) if i != o]
        parts["peer compute+push (max)"].append(max(pc))
    print("EP timeline per layer (us, mean over layers and steps):")
    for k, v in parts.items():
        cost[k] = sum(v) / len(v) / 1000
        print(f"  {k:26s} {cost[k]:8.1f} us")
    remote_per_layer_token = remote_routes / total_routes * topk
    cost["remote_dispatch_us_per_layer_token"] = cost["multicast issue"] + cost["wait partials"]
    cost["remote_dispatch_us_per_remote_route"] = cost["remote_dispatch_us_per_layer_token"] / max(remote_per_layer_token, 1e-9)
    print(f"  remote dispatch (multicast + wait) {cost['remote_dispatch_us_per_layer_token']:.1f} us per layer-token, "
          f"{remote_per_layer_token:.2f} remote routes per layer-token -> {cost['remote_dispatch_us_per_remote_route']:.1f} us per remote route")
S_ref = p0["S"]
partial_bytes = S_ref * dim * (2 if p0.get("bf16_part") else 4)
bytes_saved_per_route = partial_bytes / topk  # attribution of one peer's partial return to the routes it carries
print(f"bytes attributed per remote route: {bytes_saved_per_route:.0f} (one bf16/fp32 partial of {partial_bytes} B per peer / topk); "
      f"note: the route packet ({S_ref * dim * 2 + 2 * S_ref * topk * 4} B) is multicast to every peer regardless of routing")

# ---------------------------------------------------------------- rankings
cand = []  # (count, L, e, g)
for L in range(nl):
    g = owner[L]
    for e in torch.nonzero(counts[L] > 0).flatten().tolist():
        if not local_mask[L, e]:
            cand.append((counts[L, e].item(), L, e, g))
cand.sort(reverse=True)
p2p_cost = cost.get("remote_dispatch_us_per_remote_route", 1.0)
print(f"\ntop {a.top} experts globally by route count ((layer, expert), remote?, count, share of layer's routes):")
allc = [(counts[L, e].item(), L, e) for L in range(nl) for e in torch.nonzero(counts[L] > 0).flatten().tolist()]
allc.sort(reverse=True)
per_layer_tot = counts.sum(1)
for c, L, e in allc[:a.top]:
    print(f"  L{L:2d} E{e:3d} {'remote' if not local_mask[L, e] else 'local ':6s} {int(c):7,d} {100 * c / per_layer_tot[L].item():5.1f}%")
byid = counts.sum(0)
top_ids = torch.argsort(byid, descending=True)[:20].tolist()
print("top 20 expert ids summed over layers:", ", ".join(f"E{e}={int(byid[e])}" for e in top_ids))
print(f"\ntop {a.per_gpu} replica candidates per GPU (by remote route count; score = count * bytes_saved * P2P us/route):")
for g in gpus:
    cg = [c for c in cand if c[3] == g][:a.per_gpu]
    print(f"  cuda:{g}:")
    for c, L, e, _ in cg:
        print(f"    L{L:2d} E{e:3d} count {int(c):7,d}  score {c * bytes_saved_per_route * p2p_cost / 1e6:10.3f}")

# ---------------------------------------------------------------- coverage vs budget
budgets = [float(v) for v in a.budgets.split(",")]
def plan_for(gib):
    slots = int(gib * 2**30 // bpe)
    plan = {g: [] for g in gpus}
    for c, L, e, g in cand:
        if len(plan[g]) < slots:
            plan[g].append((L, e, c))
    return plan, slots
print(f"\nreplica budget coverage (bytes per expert {bpe / 2**20:.2f} MiB):")
print(f"  {'GiB/GPU':>8s} {'slots/GPU':>9s} {'remote routes covered':>22s} {'layer-tokens fully local':>25s} {'tokens fully local (all layers)':>32s}")
for gib in budgets:
    plan, slots = plan_for(gib)
    rep = torch.zeros(nl, E, dtype=torch.bool)
    for g, lst in plan.items():
        for L, e, _ in lst:
            rep[L, e] = True
    covered = ev_counts[rep].sum().item()
    # per (step, row, layer): all remote experts replicated?
    full_layer = 0; n_layer_ev = 0; full_tok = 0; n_tok = 0
    for eid in events:
        steps, _, S, _ = eid.shape
        lm = local_mask.gather(1, eid.permute(1, 0, 2, 3).reshape(nl, -1)).reshape(nl, steps, S, topk)
        rm = rep.gather(1, eid.permute(1, 0, 2, 3).reshape(nl, -1)).reshape(nl, steps, S, topk)
        ok = (lm | rm).all(-1)  # [nl, steps, S]
        full_layer += ok.sum().item(); n_layer_ev += ok.numel()
        full_tok += ok.all(0).sum().item(); n_tok += ok[0].numel()
    ev_remote = ev_counts[~local_mask].sum().item()
    print(f"  {gib:8.1f} {slots:9d} {100 * covered / ev_remote:21.1f}% {100 * full_layer / n_layer_ev:24.1f}% {100 * full_tok / n_tok:31.1f}%")
# baseline: fraction of layer-tokens that need no peer at all today
lt0 = 0; n0 = 0
for eid in events:
    steps, _, S, _ = eid.shape
    lm = local_mask.gather(1, eid.permute(1, 0, 2, 3).reshape(nl, -1)).reshape(nl, steps, S, topk)
    lt0 += lm.all(-1).sum().item(); n0 += lm.all(-1).numel()
print(f"  (no replicas: {100 * lt0 / n0:.1f}% of layer-tokens have all {topk} experts on the owner)")

# ---------------------------------------------------------------- locality
def locality(kind):
    """kind = 'layer': predict experts of layer L+1 from the experts of layer L (same token);
    kind = 'time': predict experts of layer L at token t+1 from those at token t (same row). Train on the first
    (1 - holdout) steps of every profile, evaluate on the rest; baseline = unconditional top-k of the target."""
    cond = torch.zeros(nl, E, E, dtype=torch.float32)  # [layer of the condition, e_cond, e_target]
    marg = torch.zeros(nl, E, dtype=torch.float32)
    hits = {k: 0.0 for k in (1, 2, 4, 8)}; base = {k: 0.0 for k in (1, 2, 4, 8)}; n_eval = 0
    for eid in events:
        steps, _, S, _ = eid.shape
        cut = int(steps * (1 - a.holdout))
        if kind == "layer":
            pairs = [(eid[:, L], eid[:, L + 1], L) for L in range(nl - 1)]  # [steps, S, topk] each
        else:
            pairs = [(eid[:-1, L], eid[1:, L], L) for L in range(nl)]
        for src, dst, L in pairs:
            tr_s, tr_d = src[:cut].reshape(-1, topk), dst[:cut].reshape(-1, topk)
            for i in range(topk):
                for j in range(topk):
                    cond[L].view(-1).index_add_(0, (tr_s[:, i] * E + tr_d[:, j]), torch.ones(tr_s.shape[0]))
            marg[L].index_add_(0, tr_d.reshape(-1), torch.ones(tr_d.numel()))
        for src, dst, L in pairs:
            ev_s, ev_d = src[cut:].reshape(-1, topk), dst[cut:].reshape(-1, topk)
            if ev_s.shape[0] == 0:
                continue
            pred = cond[L][ev_s].sum(1)  # [n, E]
            mtop = torch.argsort(marg[L], descending=True)
            for k in hits:
                topp = torch.topk(pred, k, dim=1).indices  # [n, k]
                hit = (topp.unsqueeze(2) == ev_d.unsqueeze(1)).any(2).sum(1).float()  # predicted experts that are actual
                hits[k] += hit.sum().item()
                bh = (mtop[:k].view(1, k, 1) == ev_d.unsqueeze(1)).any(2).sum(1).float()
                base[k] += bh.sum().item()
            n_eval += ev_s.shape[0]
    print(f"\nrouting locality ({'P(E_L+1 | E_L)' if kind == 'layer' else 'P(E_t+1,L | E_t,L)'}), {n_eval:,} held-out events:")
    print(f"  {'k':>3s} {'precision@k (cond)':>18s} {'recall@k (cond)':>16s} {'precision@k (marginal)':>22s} {'recall@k (marginal)':>20s}")
    for k in hits:
        print(f"  {k:3d} {100 * hits[k] / (n_eval * k):17.1f}% {100 * hits[k] / (n_eval * topk):15.1f}% {100 * base[k] / (n_eval * k):21.1f}% {100 * base[k] / (n_eval * topk):19.1f}%")
    return hits, base, n_eval
loc_layer = locality("layer")
loc_time = locality("time")

# ---------------------------------------------------------------- optional plan
if a.plan:
    plan, slots = plan_for(a.plan_gib)
    js = {"plan_gib": a.plan_gib, "slots_per_gpu": slots, "bytes_per_expert": bpe, "profiles": a.profiles,
          "replicas": {str(g): [[L, e] for L, e, _ in lst] for g, lst in plan.items()}}
    json.dump(js, open(a.plan, "w"), indent=1)
    print(f"\nplan written to {a.plan}: " + ", ".join(f"cuda:{g} {len(lst)} replicas ({len(lst) * bpe / 2**30:.2f} GiB)" for g, lst in plan.items()))
