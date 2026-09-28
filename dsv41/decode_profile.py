"""Critical-path profile of one decode token on the EP runtime (baseline mode).

Records one decode step with the Kineto/CUPTI profiler (every kernel and copy inside the CUDA graphs, per device
and stream) together with the runtime's globaltimer stamps (DSV41_EP_TRACE). The stamp kernels (`p2p_stamp`) are
phase boundaries on each device's main stream, so every kernel/copy can be attributed to (device, layer, phase);
per phase the report gives wall-clock time (between the stamps), GPU kernel time (sum of durations of all kernels
of that device inside the window, all streams), overlap (kernel time beyond the union of busy intervals) and idle
gap (wall minus busy union). Kernels are also classified by name (attention, hyper-connection mixes, RMSNorm,
router, FP4 expert GEMMs, SwiGLU, shared expert, P2P messaging, partial sums, combine, head, ...).

usage: python -m dsv41.decode_profile --seqs 1 [--steps 3] [--out results/decode_profile_S1.json]
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict

os.environ.setdefault("DSV41_EP_TRACE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash-Abliterated")
ap.add_argument("--devices", default="2,3,0,1")
ap.add_argument("--ep-shards", default="")
ap.add_argument("--seqs", type=int, default=1)
ap.add_argument("--steps", type=int, default=3, help="profiled steps (the last one is analysed)")
ap.add_argument("--warmup", type=int, default=12)
ap.add_argument("--prompts", default="dsv41/batch_prompts64.txt")
ap.add_argument("--prompt-offset", type=int, default=0)
ap.add_argument("--max-seq-len", type=int, default=8192)
ap.add_argument("--out", default="")
a = ap.parse_args()

from transformers import AutoTokenizer
from dsv41.load import load_model
from dsv41.ep import EPRuntime, trace_report

tok = AutoTokenizer.from_pretrained(a.ckpt)
sys.path.insert(0, os.path.join(a.ckpt, "encoding"))
from encoding import encode_messages

S = a.seqs
devs = [int(d) for d in a.devices.split(",")]
shards = [int(v) for v in a.ep_shards.split(",")] if a.ep_shards else None
model = load_model(a.ckpt, devs, max_seq_len=a.max_seq_len, max_batch=S, max_seqs=S, engram=True, tokenizer=tok, ep=True, ep_shards=shards)
rt = EPRuntime(model, use_graphs=True)
rt.capture()
nl = len(model.blocks)
owner_of = [b.device.index for b in model.blocks]
lines = [l.strip() for l in open(a.prompts) if l.strip()]
lines = (lines[a.prompt_offset:] + lines[:a.prompt_offset])[:S]
prompts = [tok.encode(encode_messages([{"role": "user", "content": l}], thinking_mode="chat")) for l in lines]
p_last, nxt = [0] * S, [0] * S
for s in range(S - 1, -1, -1):
    logits = model.forward(torch.tensor([prompts[s]]), 0)
    if s > 0:
        rt.copy_seq(0, s)
    p_last[s] = len(prompts[s]) - 1
    nxt[s] = int(logits.argmax(-1))
torch.cuda.synchronize()

def step():
    global nxt
    poss = [p_last[s] + 1 for s in range(S)]
    t0 = time.perf_counter()
    logits = rt.step(nxt, poss, seq=list(range(S)), pmax=poss)
    t1 = time.perf_counter()
    am = logits.argmax(-1).tolist()
    t2 = time.perf_counter()
    for s in range(S):
        nxt[s] = am[s]
        p_last[s] += 1
    return (t1 - t0) * 1e6, (t2 - t1) * 1e6

for _ in range(a.warmup):
    step()
torch.cuda.synchronize()
host_times = []
prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=False)
prof.__enter__()
for _ in range(a.steps):
    torch.cuda.synchronize()
    host_times.append(step())
torch.cuda.synchronize()
prof.__exit__(None, None, None)
trace_path = (a.out or f"results/decode_profile_S{S}.json").replace(".json", "_trace.json")
os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
prof.export_chrome_trace(trace_path)
print(f"[profile] host: step (rt.step incl. final sync) {host_times[-1][0]:.0f} us, argmax+tolist {host_times[-1][1]:.0f} us; trace {trace_path}", flush=True)
print("[EP timeline per layer, last step]\n" + trace_report(rt), flush=True)

# ------------------------------------------------------------------ parse the trace
ev = json.load(open(trace_path))["traceEvents"]
gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "ts" in e and "dur" in e]
for e in gpu:
    e["dev"] = int(e["args"].get("device", -1))
    e["stream"] = int(e["args"].get("stream", -1))
# the last profiled step = the last block of stamps; find stamp kernels per device in time order
stamps = defaultdict(list)
for e in gpu:
    if e["name"].startswith("p2p_stamp"):
        stamps[e["dev"]].append(e)
for d in stamps:
    stamps[d].sort(key=lambda e: e["ts"])
# expected stamp sequence per device per token
OWNER_SEQ = [0, 1, 8, 9, 2, 10, 3, 4, 5]
PEER_SEQ = [6, 7]
def expected(dev):
    seq = []
    for L in range(nl):
        for k in (OWNER_SEQ if owner_of[L] == dev else PEER_SEQ):
            seq.append((L, k))
    return seq
per_dev = {}
for dev, st in stamps.items():
    exp = expected(dev)
    n_tok = len(st) // len(exp)
    assert n_tok >= 1 and len(st) % len(exp) == 0, (dev, len(st), len(exp))
    last = st[-len(exp):]  # the last profiled token
    per_dev[dev] = {(L, k): e["ts"] for (L, k), e in zip(exp, last)}
# kernels of the last token per device (between its first and last stamp)
tok_win = {dev: (min(v.values()), max(v.values())) for dev, v in per_dev.items()}
kern = defaultdict(list)
for e in gpu:
    if e["dev"] in tok_win and tok_win[e["dev"]][0] - 50 <= e["ts"] <= tok_win[e["dev"]][1] + 200:
        kern[e["dev"]].append(e)

def classify(name, cat):
    n = name.lower()
    if cat == "gpu_memcpy":
        return "memcpy (DMA route/partials/hops)"
    if cat == "gpu_memset":
        return "memset"
    if "p2p_stamp" in n:
        return "trace stamps"
    if "p2p_wait" in n:
        return "P2P wait (spin)"
    if "p2p_signal" in n or "p2p_multicast" in n or "p2p_copy" in n or "p2p_seq" in n:
        return "P2P send/signal"
    if "p2p_sum_rows" in n:
        return "partial sum (pairs -> partial)"
    if "fp4_gemm" in n:
        return "FP4 expert GEMM"
    if "swiglu" in n:
        return "SwiGLU + FP8 round"
    if "gate_topk" in n or "topk" in n:
        return "router top-k"
    if "hc_post" in n:
        return "combine (hc_post)"
    if "hc_mix" in n or "hc_pre" in n or "hc_sub" in n or "hc_" in n:
        return "hyper-connection mixes"
    if "rmsnorm" in n or "norm" in n:
        return "RMSNorm"
    if "sattn" in n or "attn" in n or "flash" in n or "softmax" in n:
        return "attention core"
    if "index" in n or "compress" in n or "rope" in n or "window" in n:
        return "attention: indexer/compressor/rope"
    if "fp8_gemv" in n or "fp8_gemm" in n or "w8" in n or "linear_w" in n:
        return "FP8 dense GEMV (attention proj / shared expert / gate)"
    if "gemm" in n or "gemv" in n or "cutlass" in n or "cublas" in n or "sgemm" in n or "hgemm" in n:
        return "other GEMM (cuBLAS)"
    if "engram" in n or "hash" in n or "gather" in n or "index_select" in n:
        return "engram / gathers"
    if "elementwise" in n or "vectorized" in n or "copy" in n or "fill" in n or "cat" in n or "reduce" in n or "add" in n or "mul" in n:
        return "torch elementwise / copies"
    if "sort" in n or "cumsum" in n or "scan" in n or "scatter" in n or "unique" in n:
        return "bucketing (sort/scan/scatter)"
    return "other: " + name[:60]

def union_len(intervals):
    intervals = sorted(intervals)
    tot = 0.0; cur_s = None; cur_e = None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot

PHASES = [("attention+hc (0->1)", 0, 1), ("hc_sub2 (1->8)", 1, 8), ("gate+topk (8->9)", 8, 9), ("multicast issue (9->2)", 9, 2),
          ("local expert GEMMs (2->10)", 2, 10), ("shared-expert join (10->3)", 10, 3), ("wait partials (3->4)", 3, 4), ("combine (4->5)", 4, 5)]
summary = {"S": S, "host_step_us": host_times[-1][0], "host_sample_us": host_times[-1][1], "layers": [], "phases": {}, "categories": {}, "devices": {}}
phase_acc = defaultdict(lambda: {"wall": 0.0, "busy": 0.0, "union": 0.0, "n": 0})
cat_acc = defaultdict(lambda: defaultdict(float))
print(f"\n[per-layer phases, S={S}: wall / kernel / overlap / idle us]")
print(f"  {'layer':>5s} {'dev':>3s} " + " ".join(f"{p.split(' (')[0][:14]:>14s}" for p, _, _ in PHASES) + "  layer_wall")
for L in range(nl):
    dev = owner_of[L]
    st = per_dev[dev]
    row = {"layer": L, "dev": dev, "phases": {}}
    cells = []
    for pname, k0, k1 in PHASES:
        t0, t1 = st[(L, k0)], st[(L, k1)]
        ks = [e for e in kern[dev] if t0 <= e["ts"] < t1 and not e["name"].startswith("p2p_stamp")]
        busy = sum(e["dur"] for e in ks)
        un = union_len([(e["ts"], e["ts"] + e["dur"]) for e in ks])
        wall = t1 - t0
        row["phases"][pname] = {"wall": wall, "kernel": busy, "overlap": busy - un, "idle": wall - un}
        pa = phase_acc[pname]; pa["wall"] += wall; pa["busy"] += busy; pa["union"] += un; pa["n"] += 1
        for e in ks:
            cat_acc[classify(e["name"], e["cat"])][pname] += e["dur"]
        cells.append(f"{wall:5.0f}/{busy:4.0f}/{busy - un:3.0f}/{wall - un:3.0f}")
    lw = st[(L, 5)] - st[(L, 0)]
    row["layer_wall"] = lw
    # peers: compute+push window per peer
    row["peer_windows"] = {}
    for pdev, pst in per_dev.items():
        if pdev != dev:
            t6, t7 = pst[(L, 6)], pst[(L, 7)]
            ks = [e for e in kern[pdev] if t6 <= e["ts"] < t7 and not e["name"].startswith("p2p_stamp")]
            row["peer_windows"][pdev] = {"wall": t7 - t6, "kernel": sum(e["dur"] for e in ks), "route_arrival_after_owner_gate": t6 - st[(L, 9)]}
            for e in ks:
                cat_acc["peer: " + classify(e["name"], e["cat"])]["peer compute+push (6->7)"] += e["dur"]
    summary["layers"].append(row)
    print(f"  {L:5d} {dev:3d} " + " ".join(f"{c:>14s}" for c in cells) + f"  {lw:7.0f}")
print("\n[phase totals per token: wall / kernel / overlap / idle us, summed over the 40 layers]")
tot_wall = 0
for pname, _, _ in PHASES:
    pa = phase_acc[pname]
    summary["phases"][pname] = {"wall": pa["wall"], "kernel": pa["busy"], "overlap": pa["busy"] - pa["union"], "idle": pa["wall"] - pa["union"]}
    tot_wall += pa["wall"]
    print(f"  {pname:32s} {pa['wall']:8.0f} {pa['busy']:8.0f} {pa['busy'] - pa['union']:8.0f} {pa['wall'] - pa['union']:8.0f}")
# hops and token-level accounting on the owner chain
chain = []
for L in range(nl):
    chain.append((per_dev[owner_of[L]][(L, 0)], per_dev[owner_of[L]][(L, 5)]))
hop_gaps = [chain[L + 1][0] - chain[L][1] for L in range(nl - 1)]
tok_span = chain[-1][1] - chain[0][0]
print(f"\n[critical path] layers (owner sections) {tot_wall:.0f} us, inter-layer gaps {sum(hop_gaps):.0f} us "
      f"(pipeline hops at layers {[L for L in range(nl - 1) if owner_of[L] != owner_of[L + 1]]}: {[round(hop_gaps[L]) for L in range(nl - 1) if owner_of[L] != owner_of[L + 1]]} us; "
      f"same-device gaps mean {sum(g for L, g in enumerate(hop_gaps) if owner_of[L] == owner_of[L + 1]) / max(1, sum(1 for L in range(nl - 1) if owner_of[L] == owner_of[L + 1])):.1f} us), "
      f"first-stamp to last-stamp {tok_span:.0f} us, host step {host_times[-1][0]:.0f} us (host-side launch/sync overhead {host_times[-1][0] - tok_span:.0f} us), sampling {host_times[-1][1]:.0f} us")
summary["critical_path"] = {"layers_wall": tot_wall, "gaps": sum(hop_gaps), "hop_gaps": hop_gaps, "token_span": tok_span, "host_step": host_times[-1][0], "host_sample": host_times[-1][1]}
# head/logits after the last layer
last_dev = owner_of[-1]
tail = [e for e in kern[last_dev] if e["ts"] >= per_dev[last_dev][(nl - 1, 5)] and not e["name"].startswith("p2p_stamp")]
summary["critical_path"]["head_kernels_us"] = sum(e["dur"] for e in tail)
print(f"  after the last layer (norm + head + logits): {sum(e['dur'] for e in tail):.0f} us of kernels")
# per-device busy
print("\n[per device, last token] kernel time (all streams) / busy union / span")
for dev in sorted(kern):
    ks = [e for e in kern[dev] if not e["name"].startswith("p2p_stamp")]
    busy = sum(e["dur"] for e in ks); un = union_len([(e["ts"], e["ts"] + e["dur"]) for e in ks])
    span = tok_win[dev][1] - tok_win[dev][0]
    wait = sum(e["dur"] for e in ks if "p2p_wait" in e["name"])
    summary["devices"][dev] = {"kernel": busy, "busy_union": un, "span": span, "wait_spin": wait, "n_kernels": len(ks)}
    print(f"  cuda:{dev}: {busy:7.0f} / {un:7.0f} / {span:7.0f} us, of which spin-wait kernels {wait:6.0f} us, {len(ks)} kernels")
print("\n[kernel categories, owner sections, us per token]")
cats = sorted(cat_acc.items(), key=lambda kv: -sum(kv[1].values()))
for cat, ph in cats:
    summary["categories"][cat] = dict(ph)
    print(f"  {sum(ph.values()):8.0f}  {cat:60s}  " + ", ".join(f"{p.split(' (')[0]}={v:.0f}" for p, v in sorted(ph.items(), key=lambda kv: -kv[1])[:3]))
# ---- upper bounds from measured phases (S=1 single stream)
ms_step = host_times[-1][0] / 1000
def bound(saved_us, label):
    t = host_times[-1][0] - saved_us
    print(f"  {label:70s} -> {t / 1000:6.2f} ms/step = {S * 1e6 / t:6.1f} tok/s (measured {S * 1e6 / host_times[-1][0]:.1f})")
    return S * 1e6 / t
wait_us = summary["phases"]["wait partials (3->4)"]["wall"]
mc_us = summary["phases"]["multicast issue (9->2)"]["wall"]
hops = sum(hop_gaps[L] for L in range(nl - 1) if owner_of[L] != owner_of[L + 1])
exp_us = summary["phases"]["local expert GEMMs (2->10)"]["wall"] + summary["phases"]["shared-expert join (10->3)"]["wall"]
print("\n[upper bounds, from this token's measured phases]")
summary["bounds"] = {
    "no_peer_waits": bound(wait_us, "all peer waits removed (wait partials = 0)"),
    "free_p2p": bound(wait_us + mc_us + hops, "all P2P activation traffic free (waits + multicast issue + pipeline hops = 0)"),
}
summary["bounds"]["expert_phase_us"] = exp_us
print(f"  (expert phase per token now {exp_us:.0f} us; the S=8 comparison for 'perfect batching' is done offline from the S=8 profile)")
out = a.out or f"results/decode_profile_S{S}.json"
json.dump(summary, open(out, "w"), indent=1)
print("[profile] summary saved", out)
