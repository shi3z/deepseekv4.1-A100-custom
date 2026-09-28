"""A/B benchmark of the static expert-replica cache (dsv41/replica.py) on the 4-GPU EP decode runtime.

One process = one configuration (the replica tensors and CUDA graphs are built at load): run it once per
(cache budget, streams) and compare the JSON lines it appends to --log. Measures tok/s and ms/token of plain
batched decode over --steps tokens (host timing across rt.step, one host sync per token like the server), the
EP per-layer device timeline (DSV41_EP_TRACE), the replica counters, GPU utilisation / memory-controller
utilisation / PCIe throughput sampled by `nvidia-smi dmon` while decoding, and NVLink byte counters before and
after when the driver exposes them. --check-tokens saves / compares the greedy tokens with the baseline run.

usage: python -m dsv41.bench_replica --seqs 1 --steps 200 [--plan plan.json --gib 4] [--no-skip-wait]
       [--save-tokens results/tok_base_S1.json | --check-tokens results/tok_base_S1.json] [--log results/replica_bench.jsonl]
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time, threading

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash-Abliterated")
ap.add_argument("--devices", default="2,3,0,1")
ap.add_argument("--ep-shards", default="")
ap.add_argument("--seqs", type=int, default=1)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--warmup", type=int, default=16)
ap.add_argument("--prompts", default="dsv41/batch_prompts64.txt")
ap.add_argument("--prompt-offset", type=int, default=0)
ap.add_argument("--max-seq-len", type=int, default=8192)
ap.add_argument("--plan", default="", help="replica plan JSON (enables DSV41_EXPERT_REPLICA_CACHE=1)")
ap.add_argument("--gib", type=float, default=0.0, help="cap the plan to this many GiB per GPU")
ap.add_argument("--no-skip-wait", action="store_true", help="replicas without the masked wait (peers always waited on)")
ap.add_argument("--trace", action="store_true", help="EP per-layer timeline (DSV41_EP_TRACE=1)")
ap.add_argument("--save-tokens", default="")
ap.add_argument("--check-tokens", default="")
ap.add_argument("--log", default="results/replica_bench.jsonl")
ap.add_argument("--label", default="")
a = ap.parse_args()

if a.plan:
    os.environ["DSV41_EXPERT_REPLICA_CACHE"] = "1"
    os.environ["DSV41_REPLICA_PLAN"] = a.plan
    if a.gib:
        os.environ["DSV41_REPLICA_GIB"] = str(a.gib)
    os.environ["DSV41_REPLICA_SKIP_WAIT"] = "0" if a.no_skip_wait else "1"
if a.trace:
    os.environ["DSV41_EP_TRACE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
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
free_vram = {}
for d in rt.devs:
    torch.cuda.synchronize(d)
    fr, tot = torch.cuda.mem_get_info(d)
    free_vram[d.index] = round(fr / 2**30, 2)
print("[bench] free VRAM after capture (GiB):", free_vram, flush=True)

lines = [l.strip() for l in open(a.prompts) if l.strip()]
lines = (lines[a.prompt_offset:] + lines[:a.prompt_offset])[:S]
prompts = [tok.encode(encode_messages([{"role": "user", "content": l}], thinking_mode="chat")) for l in lines]
p_last, nxt = [0] * S, [0] * S
first_logits = None
for s in range(S - 1, -1, -1):
    logits = model.forward(torch.tensor([prompts[s]]), 0)
    if s > 0:
        rt.copy_seq(0, s)
    p_last[s] = len(prompts[s]) - 1
    nxt[s] = int(logits.argmax(-1))
torch.cuda.synchronize()

# ---- warm-up, then the timed loop with nvidia-smi sampling
first_logits = None
first_h = None
first_hits = None
def run(n_steps, record):
    global first_logits
    toks = [[] for _ in range(S)]
    lat = []
    for _ in range(n_steps):
        poss = [p_last[s] + 1 for s in range(S)]
        t0 = time.perf_counter()
        logits = rt.step(nxt, poss, seq=list(range(S)), pmax=poss)
        am = logits.argmax(-1).tolist()
        lat.append(time.perf_counter() - t0)
        if first_logits is None:
            first_logits = logits.detach().float().cpu().clone()  # the very first decode step (before warm-up)
            if rt.dbg_h is not None:
                global first_h, first_hits
                first_h = torch.stack([rt.dbg_h[blk.device][L].float().cpu() for L, blk in enumerate(model.blocks)])  # [nl, B, hc, dim]
                if rt.replica is not None:
                    first_hits = torch.stack([rt.replica.hits[blk.device][L].cpu() for L, blk in enumerate(model.blocks)]).tolist()
        for s in range(S):
            if record:
                toks[s].append(am[s])
            nxt[s] = am[s]
            p_last[s] += 1
    return toks, lat
run(a.warmup, False)
if rt.replica is not None and getattr(rt, "replica_debug", False):
    run(1, False)
    bad = 0
    for d in rt.devs:
        need = rt.dbg_need[d].cpu(); rows = rt.dbg_rows[d].cpu()
        for L, blk in enumerate(model.blocks):
            if blk.device != d:
                continue
            for i in range(rt.nd):
                if need[L, i] == 0 and rows[L, i] > 0:
                    bad += 1
                    if bad <= 12:
                        print(f"[replica-debug] {d} L{L} slot {i} ({rt.devs[i]}) masked out but |row| sum = {rows[L, i]:.4f}; need row {need[L].tolist()} rows {[round(x, 3) for x in rows[L].tolist()]}", flush=True)
    print(f"[replica-debug] masked-out rows with non-zero content: {bad}", flush=True)
if rt.replica is not None:
    rt.replica.reset_counters()
dev_arg = ",".join(str(d) for d in devs)
samples = []
def sampler(stop):
    p = subprocess.Popen(["nvidia-smi", "dmon", "-i", dev_arg, "-s", "ut", "-d", "1"], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if stop.is_set():
            break
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) >= 7:
            try:
                samples.append({"gpu": int(f[0]), "sm": float(f[1]), "mem": float(f[2]), "rx_mb": float(f[5]), "tx_mb": float(f[6])})
            except ValueError:
                pass
    p.kill()
def nvlink_bytes():
    try:
        out = subprocess.run(["nvidia-smi", "nvlink", "-gt", "d"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    tot = {}
    cur = None
    for line in out.splitlines():
        if line.startswith("GPU"):
            cur = int(line.split()[1].rstrip(":"))
        elif "Data" in line and cur is not None and cur in devs:
            v = line.split(":")[-1].strip().split()
            try:
                kib = float(v[0]) * {"KiB": 1, "MiB": 1024, "GiB": 1024**2}.get(v[1], 1)
            except (ValueError, IndexError):
                continue
            tot[cur] = tot.get(cur, 0.0) + kib
    return tot or None
nv0 = nvlink_bytes()
stop = threading.Event()
th = threading.Thread(target=sampler, args=(stop,), daemon=True)
th.start()
torch.cuda.synchronize()
t0 = time.perf_counter()
toks, lat = run(a.steps, True)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
stop.set()
nv1 = nvlink_bytes()

n_tok = S * a.steps
ms_tok = dt / a.steps * 1000
lat_ms = sorted(l * 1000 for l in lat)
res = {
    "label": a.label or ("replica" if a.plan else "baseline"), "S": S, "steps": a.steps, "plan": a.plan, "gib": a.gib,
    "skip_wait": bool(a.plan) and not a.no_skip_wait,
    "tok_s": n_tok / dt, "ms_per_step": ms_tok, "ms_per_token_per_stream": ms_tok,
    "step_ms_p50": lat_ms[len(lat_ms) // 2], "step_ms_p90": lat_ms[int(len(lat_ms) * 0.9)],
    "free_vram_gib": free_vram,
}
if samples:
    by = {}
    for smp in samples:
        by.setdefault(smp["gpu"], []).append(smp)
    res["gpu"] = {g: {"sm_util": sum(x["sm"] for x in v) / len(v), "mem_util": sum(x["mem"] for x in v) / len(v),
                      "pcie_rx_mb_s": sum(x["rx_mb"] for x in v) / len(v), "pcie_tx_mb_s": sum(x["tx_mb"] for x in v) / len(v), "n": len(v)}
                  for g, v in by.items()}
if nv0 and nv1:
    res["nvlink_kib_delta"] = {g: nv1.get(g, 0) - nv0.get(g, 0) for g in nv1}
if rt.trace is not None:
    tr = {d: rt.trace[d].cpu() for d in rt.devs}
    agg = {}
    for L, blk in enumerate(model.blocks):
        t = tr[blk.device][L]
        pc = [tr[d][L, 7].item() - tr[d][L, 6].item() for d in rt.devs if d != blk.device]
        for k, v in {"attention_hc": t[1] - t[0], "gate_topk": t[9] - t[8], "multicast": t[2] - t[9], "local_expert_gemms": t[10] - t[2],
                     "own_experts_and_shared": t[3] - t[2], "wait_partials": t[4] - t[3], "hc_post": t[5] - t[4], "layer": t[5] - t[0],
                     "peer_compute_push_max": max(pc)}.items():
            agg.setdefault(k, []).append(float(v))
    res["trace_us"] = {k: sum(v) / len(v) / 1000 for k, v in agg.items()}
    res["trace_ms_per_token"] = {k: sum(v) / 1e6 for k, v in agg.items()}
    print("[EP timeline per layer, last step]\n" + trace_report(rt))
if rt.replica is not None:
    res["replica"] = rt.replica.report()
    tr_ = res.get("trace_ms_per_token", {})
    res["local_replica_gemm_ms"] = tr_.get("local_expert_gemms")
    res["remaining_remote_dispatch_ms"] = (tr_.get("multicast", 0) + tr_.get("wait_partials", 0)) if tr_ else None
if a.save_tokens:
    json.dump({"prompts": lines, "tokens": toks, "warmup": a.warmup}, open(a.save_tokens, "w"))
    torch.save(first_logits, a.save_tokens + ".logits.pt")
    if first_h is not None:
        torch.save(first_h, a.save_tokens + ".h.pt")
if a.check_tokens:
    refj = json.load(open(a.check_tokens))
    ref = refj["tokens"]
    same = [sum(1 for x, y in zip(ref[s], toks[s]) if x == y) for s in range(S)]
    first_diff = [next((i for i, (x, y) in enumerate(zip(ref[s], toks[s])) if x != y), None) for s in range(S)]
    res["token_match"] = {"same": same, "of": [len(r) for r in ref], "first_diff": first_diff, "same_warmup": refj.get("warmup") == a.warmup}
    if os.path.exists(a.check_tokens + ".logits.pt"):
        rl = torch.load(a.check_tokens + ".logits.pt")
        diff = (rl - first_logits).abs()
        if first_h is not None and os.path.exists(a.check_tokens + ".h.pt"):
            rh = torch.load(a.check_tokens + ".h.pt")
            per_layer = [(rh[L] - first_h[L]).abs().max().item() for L in range(rh.shape[0])]
            rel = [((rh[L] - first_h[L]).abs().max() / rh[L].abs().max()).item() for L in range(rh.shape[0])]
            first_div = next((L for L, v in enumerate(per_layer) if v > 0), None)
            res["first_step_hidden"] = {"first_divergent_layer": first_div, "max_abs_diff_per_layer": [round(v, 5) for v in per_layer],
                                        "rel_per_layer": [round(v, 5) for v in rel], "replica_hits_per_layer_step0": first_hits}
            print(f"[check] first divergent layer {first_div}; per-layer max|dh| {[round(v, 4) for v in per_layer]}", flush=True)
            if first_hits is not None:
                print(f"[check] replica hits per layer at step 0: {first_hits}", flush=True)
        res["first_step_logits"] = {"max_abs_diff": diff.max().item(), "mean_abs_diff": diff.mean().item(),
                                    "ref_absmax": rl.abs().max().item(), "argmax_same": bool((rl.argmax(-1) == first_logits.argmax(-1)).all()),
                                    "top1_margin_ref": [float(v) for v in (rl.topk(2, dim=-1).values[:, 0] - rl.topk(2, dim=-1).values[:, 1])]}
print(json.dumps(res, ensure_ascii=False), flush=True)
print(f"[bench] {res['label']} S={S}: {res['tok_s']:.1f} tok/s, {ms_tok:.2f} ms/step; sample: {tok.decode(toks[0][:30]).replace(chr(10), ' ')[:120]!r}")
os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
with open(a.log, "a") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
