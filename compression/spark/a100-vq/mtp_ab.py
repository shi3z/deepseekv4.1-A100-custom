"""Config Q (fp8 head) vs Config S (fp4 head) end to end, in one process.

Both arms share one engine, one arena, one warm cache and one prompt set; only `W.head` changes
between them. That is the only way to compare them on this box: a 0.3 GB difference in a resident
allocation moves every later one, and NOTES.md 2026-09-11 records that measuring the same head
question across two processes reported the opposite sign.

The verify block is the checkpoint's own: DSV41_BLOCK unset = 5 drafts + 1 = 6 positions.
"""
from __future__ import annotations

import collections
import json
import os
import statistics as stats
import sys
import time

import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))

# ---------------------------------------------------------------- Phase 1: sanity
DOTENV = {}
for line in open(os.path.join(WORK, ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        DOTENV[k.strip()] = v.strip()
WATCH = ("DSV41_DENSE_FP4", "DSV41_HEAD_FMT", "DSV41_FUSED_ATTN", "DSV41_BLOCK",
         "DSV41_STEP_TIMING", "EXPERT_FORMAT", "ARENA_GB", "TRANSIENT_SLOTS", "KEEP_FREE_GB")
print("=" * 78)
print("PHASE 1  environment sanity -- this process never sources .env")
print(f"{'key':20s} {'explicit env':>22s} {'.env says':>16s}   in effect")
for k in WATCH:
    e = os.environ.get(k)
    d = DOTENV.get(k)
    src = "explicit env" if e is not None else (".env (NOT read here)" if d else "code default")
    print(f"{k:20s} {str(e):>22s} {str(d):>16s}   {src}")

os.environ["DSV41_DENSE_FP4"] = "off"       # load fp8 dense; the swap below makes the arm
os.environ["DSV41_HEAD_FMT"] = "bf16"
from engine.v41_engine import V41Engine                        # noqa: E402
import v41_ref as R                                            # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402
from fp4_linear import quantize_fp8_to_fp4, quantize_to_fp4    # noqa: E402
from fp8_linear import FP8Weight, quantize_to_fp8              # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=float(os.environ.get("ARENA_GB", "80")),
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "64")),
                keep_free_gb=float(os.environ.get("KEEP_FREE_GB", "12")), prune_keep=None,
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
W, fd, m = eng.model.W, eng.fast, eng.model
cfg = eng.config()
n = 0
for b in list(W.layers) + list(W.mtp):
    for nm in ("wq_a", "wq_b", "wkv", "wo_b"):
        w = getattr(b, nm, None)
        if isinstance(w, FP8Weight):
            setattr(b, nm, quantize_fp8_to_fp4(w)); n += 1
assert n == 172, f"attn re-quantization converted {n}, expected 172"
head_bf16 = W.head.to(torch.bfloat16) if torch.is_tensor(W.head) else W.head
print(f"\nengine reports: dense_fp4(loaded)={cfg['dense_fp4']} -> attn re-quantized in place "
      f"({n} tensors)\n  head loaded {type(head_bf16).__name__}  FUSED_ATTN={FD.FUSED_ATTN}  "
      f"T_DRAFT={FD.T_DRAFT}  T_VERIFY={FD.T_VERIFY}\n  kernel={cfg['kernel']} "
      f"expert_format={cfg['expert_format']} arena_slots={cfg['arena_slots']} "
      f"transient={cfg['transient_slots']} topk={cfg['routed_topk']}", flush=True)
assert FD.FUSED_ATTN is False, "fused attention must be off"
assert FD.T_DRAFT == 5, f"expected the checkpoint's 5-draft block, got {FD.T_DRAFT}"

CONFIGS = {"Q": "fp8", "S": "fp4"}


def set_head(fmt):
    h = (head_bf16 if fmt == "bf16" else
         quantize_to_fp8(head_bf16) if fmt == "fp8" else quantize_to_fp4(head_bf16))
    W.head, fd.head_bf16 = h, h
    fd.graphs.clear()
    torch.cuda.synchronize()
    return h


# ---------------------------------------------------------------- prompts and warm-up
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
PROMPTS = [json.loads(l) for l in open(os.path.expanduser("~/prompts20.txt")) if l.strip()]
ENC = []
for p in PROMPTS:
    s = encode_messages([{"role": "user", "content": p}], thinking_mode="chat")
    ENC.append(eng.tokenizer.encode(s if isinstance(s, str) else s[0], add_special_tokens=False))
print(f"\n{len(ENC)} prompts (ja / en / code mixed)", flush=True)

set_head("fp8")
for ids in ENC:                                   # one full pass fills the arena for both arms
    list(eng.generate(ids, max_tokens=32, temperature=0.0, ignore_eos=True))
st = eng.model.store.stats
print(f"arena warmed: hit {st['hits']/max(st['hits']+st['misses'],1):.4f}", flush=True)

ids0 = ENC[0]
list(eng.generate(ids0, max_tokens=32, temperature=0.0, ignore_eos=True))
pos = m.c.len
t0tok = int(ids0[-1])
d, _ = fd.draft(t0tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}


def ev_time(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(n)]
    for a, b in ev:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return sorted(a.elapsed_time(b) for a, b in ev)


def verify_step():
    m.c.len = pos
    fd.step(block, pos, rows)


def draft_step():
    fd.draft(t0tok, pos - 1, 0.0)


# ---------------------------------------------------------------- Phase 2: greedy, MTP off
print("\n" + "=" * 78)
print("PHASE 2  greedy decode, no speculation: one token through the model and the head")
x1 = torch.zeros(1, eng.args.dim, dtype=torch.bfloat16, device="cuda")
greedy = {}
for c, fmt in CONFIGS.items():
    set_head(fmt)
    h = W.head

    def one_tok():
        m.c.len = pos
        m.forward(block[:1], pos, prefill=False, need_logits=True)

    def head_only():
        R.head_logits(x1, h)

    ms = ev_time(one_tok, 200)
    hm = ev_time(head_only, 200)
    greedy[c] = (ms[100], ms[190], ms[0], hm[100])
    print(f"  Config {c} (head {fmt}): median {ms[100]:7.3f} ms  p95 {ms[190]:7.3f}  "
          f"min {ms[0]:7.3f}  -> {1000/ms[100]:6.3f} tok/s   head alone {hm[100]:6.3f} ms",
          flush=True)

# ---------------------------------------------------------------- Phase 5: interleaved A/B
print("\n" + "=" * 78)
print("PHASE 5  same-process A/B, order Q S S Q Q S (order dependence check)")
order = ["Q", "S", "S", "Q", "Q", "S"]
step_ms, draft_ms = collections.defaultdict(list), collections.defaultdict(list)
for i, c in enumerate(order):
    set_head(CONFIGS[c])
    s = ev_time(verify_step, 200)
    dd = ev_time(draft_step, 100, warm=10)
    step_ms[c].append(s[100]); draft_ms[c].append(dd[50])
    print(f"  run{i+1} Config {c}: verify {s[100]:7.3f} ms   draft {dd[50]:6.3f} ms", flush=True)
for c in ("Q", "S"):
    sv, dv = step_ms[c], draft_ms[c]
    print(f"  Config {c}: verify mean {stats.mean(sv):7.3f} median {stats.median(sv):7.3f} "
          f"sd {stats.pstdev(sv):5.3f}   draft mean {stats.mean(dv):6.3f} sd {stats.pstdev(dv):5.3f}",
          flush=True)

# ---------------------------------------------------------------- Phase 3 + 4: MTP end to end
print("\n" + "=" * 78)
print("PHASE 3/4  MTP end to end over every prompt, and the acceptance it comes from")
e2e = {}
for c, fmt in CONFIGS.items():
    set_head(fmt)
    for ids in ENC[:3]:                            # settle the graphs for this head
        list(eng.generate(ids, max_tokens=32, temperature=0.0, ignore_eos=True))
    hist, tok_total, wall = [], 0, 0.0
    for j, ids in enumerate(ENC):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = [t for b in eng.generate(ids, max_tokens=200, temperature=0.0, ignore_eos=True)
               for t in b]
        torch.cuda.synchronize()
        wall += time.perf_counter() - t0
        tok_total += len(out)
        hist += eng.last_stats.get("accepted_hist", [])
    e2e[c] = (tok_total, wall, hist)
    print(f"  Config {c} (head {fmt}): {tok_total} tokens in {wall:7.2f} s  "
          f"-> effective {tok_total/wall:6.3f} tok/s   steps {len(hist)}  "
          f"hit {eng.model.store.hit_rate():.4f}", flush=True)

print("\n  acceptance, per drafted position (block = 5 drafts + 1)")
print(f"  {'metric':28s} {'Config Q (fp8)':>16s} {'Config S (fp4)':>16s} {'delta':>10s}")


def acc_metrics(hist):
    nh = len(hist)
    cnt = collections.Counter(hist)
    out = {"accept_len_mean": sum(hist) / nh + 1,
           "proposed/step": FD.T_DRAFT,
           "accepted/step": sum(hist) / nh,
           "rollback rate": sum(1 for a in hist if a < FD.T_DRAFT) / nh}
    for k in range(1, FD.T_DRAFT + 1):
        ge = sum(1 for a in hist if a >= k)
        gem1 = sum(1 for a in hist if a >= k - 1)
        out[f"P(accept pos{k})"] = ge / nh
        out[f"P(pos{k} | pos{k-1})"] = ge / gem1 if gem1 else float("nan")
    for k in range(0, FD.T_DRAFT + 1):
        out[f"dist accept_len={k+1}"] = cnt.get(k, 0) / nh
    return out


mq, msv = acc_metrics(e2e["Q"][2]), acc_metrics(e2e["S"][2])
for k in mq:
    a, b = mq[k], msv[k]
    print(f"  {k:28s} {a:16.4f} {b:16.4f} {b - a:+10.4f}")

# ---------------------------------------------------------------- Phase 6: bytes
print("\n" + "=" * 78)
print("PHASE 6  bytes per step and per output token")
GB = 1e9
BYTES = {"attention (fp4)": 1.75, "wo_a (fp8)": 1.26, "shared FFN (fp8)": 1.33,
         "other dense": 0.84, "routed experts (CB3)": 4.58}
HEAD = {"Q": 0.663, "S": 0.352}
print(f"  {'group':26s} {'Config Q GB/step':>18s} {'Config S GB/step':>18s}")
tot = {"Q": 0.0, "S": 0.0}
for g, v in BYTES.items():
    print(f"  {g:26s} {v:18.3f} {v:18.3f}")
    tot["Q"] += v; tot["S"] += v
for c in ("Q", "S"):
    tot[c] += 2 * HEAD[c]
print(f"  {'LM head (verify read)':26s} {HEAD['Q']:18.3f} {HEAD['S']:18.3f}")
print(f"  {'LM head (draft read)':26s} {HEAD['Q']:18.3f} {HEAD['S']:18.3f}")
print(f"  {'TOTAL':26s} {tot['Q']:18.3f} {tot['S']:18.3f}")
for c in ("Q", "S"):
    al = (mq if c == "Q" else msv)["accept_len_mean"]
    tk, wl, _ = e2e[c]
    sm = stats.mean(step_ms[c]) + stats.mean(draft_ms[c])
    print(f"  Config {c}: {tot[c]/al:.3f} GB per output token;  "
          f"{tot[c]/(sm/1000)/GB*1e9:6.1f} GB/s over verify+draft ({sm:.2f} ms)")
print("\nDONE", flush=True)
