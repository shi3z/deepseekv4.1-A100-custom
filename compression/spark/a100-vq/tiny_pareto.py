"""A Pareto of the small kernels a verify step runs, with the call site that asked for each.

Two passes over the same eager step: a dispatch mode that counts every aten call and the engine
frame that issued it, and torch.profiler for the CUDA time per op. Joining them gives calls/step
and ms/step per site, which is what decides whether a fusion is worth writing.

The eager number is an upper bound, not the prize: inside the captured graph a launch costs its
GPU time and not its dispatch, and the SwiGLU/merge fusions returned 57 % of what eager promised.
Both are printed.
"""
from __future__ import annotations

import collections
import os
import sys
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=float(os.environ.get("ARENA_GB", "80")), transient_slots=400,
                keep_free_gb=12.0, prune_keep=None, expert_format="cb3")
fd, m = eng.fast, eng.model
import elem_fused as EF                                        # noqa: E402
print(f"gate_kernel={FD.GATE_KERNEL} fuse_qkv={FD.FUSE_QKV} swiglu={EF.SWIGLU} "
      f"merge={EF.MERGE} T_VERIFY={FD.T_VERIFY}", flush=True)
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in range(2):
    list(eng.generate(ids, max_tokens=64, temperature=0.0, ignore_eos=True))
pos = m.c.len
t0 = int(ids[-1])
d, _ = fd.draft(t0, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0], device="cuda"), d.clone()])
hsh = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hsh[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}
ENGINE = (os.path.join(WORK, "engine"), os.path.join(WORK, "tools"))


def site():
    for fr in reversed(traceback.extract_stack()[:-3]):
        if fr.filename.startswith(ENGINE):
            return f"{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}"
    return "?"


class Count(TorchDispatchMode):
    def __init__(self):
        self.n = collections.Counter()
        self.on = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.on:
            self.n[(str(func).split("::")[-1].split(".")[0], site())] += 1
        return func(*args, **(kwargs or {}))


fd.use_graphs = False
m.c.len = pos
fd.step(block, pos, rows)
torch.cuda.synchronize()
c = Count()
with c:
    c.on = True
    m.c.len = pos
    fd.step(block, pos, rows)
    fd.draft(t0, pos - 1, 0.0)
    c.on = False
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity                     # noqa: E402
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    m.c.len = pos
    fd.step(block, pos, rows)
    fd.draft(t0, pos - 1, 0.0)
    torch.cuda.synchronize()
fd.use_graphs = True
per_op = {}
for e in prof.key_averages():
    if e.self_device_time_total > 0:
        per_op[e.key] = (e.self_device_time_total / 1e3, e.count)

BIG = ("_cb3v3_up_kernel", "_cb3v3_down_kernel", "_fp4_linear_kernel", "_fp8_linear_kernel",
       "_fp8_grouped_kernel", "_gate_kernel", "_moe_up_kernel", "_moe_down_kernel")
big_ms = sum(v[0] for k, v in per_op.items() if k in BIG)
tot_ms = sum(v[0] for v in per_op.values())
print(f"\nstep CUDA time {tot_ms:.3f} ms   heavy GEMM/expert kernels {big_ms:.3f} ms "
      f"({100*big_ms/tot_ms:.1f} %)   everything else {tot_ms-big_ms:.3f} ms", flush=True)

# per-site rollup: attribute each aten op's average CUDA cost to the sites that called it
op_ms = collections.Counter()
op_n = collections.Counter()
for (op, st_), n in c.n.items():
    op_n[op] += n
site_ms = collections.Counter()
site_n = collections.Counter()
for (op, st_), n in c.n.items():
    for k, (ms, cnt) in per_op.items():
        if k.startswith("aten::" + op) or k == op:
            site_ms[st_] += ms * n / max(op_n[op], 1)
            site_n[st_] += n
            break
print(f"\n{'call site':46s} {'calls':>7s} {'ms/step':>9s} {'us/call':>8s}")
for st_, ms in site_ms.most_common(22):
    n = site_n[st_]
    print(f"{st_[:46]:46s} {n:7d} {ms:9.3f} {ms*1000/max(n,1):8.2f}")
print(f"\n{'aten op':34s} {'calls':>7s} {'ms/step':>9s}")
agg = collections.Counter()
for k, (ms, cnt) in per_op.items():
    if k in BIG:
        continue
    agg[k] = ms
for k, ms in agg.most_common(18):
    print(f"{k[:34]:34s} {per_op[k][1]:7d} {ms:9.3f}")
