"""How many DISTINCT experts a verify step actually reads, and what that does to the byte count.

The verify block is 2 token positions and each routes its own top-6, so a layer reads between 6
and 12 experts, not 6. Every bandwidth figure in this study divided the measured kernel time by
6 experts per layer; if the real number is near 12 then CB3 is not at 148 GB/s and the question
"why is it below its microbenchmark" has no subject.
"""
from __future__ import annotations

import os
import sys
import time

import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
os.environ["DSV41_ROUTE_STATS"] = "1"
from engine.v41_engine import V41Engine                        # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=94.0, transient_slots=64, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
fd, m = eng.fast, eng.model
print(f"T_VERIFY={FD.T_VERIFY} topk={eng.config()['routed_topk']} "
      f"expert_mb={eng.config()['expert_mb']}", flush=True)
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in range(3):
    list(eng.generate(ids, max_tokens=160, temperature=0.0, ignore_eos=True))
print(f"hit {m.store.hit_rate():.4f}", flush=True)
pos = m.c.len
t0tok = int(ids[-1])
d, _ = fd.draft(t0tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}


def one():
    m.c.len = pos
    fd.step(block, pos, rows)


for _ in range(20):
    one()
torch.cuda.synchronize()
fd.route_stats_reset()
N = 100
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(N):
    one()
torch.cuda.synchronize()
verify_ms = (time.perf_counter() - t0) * 1000 / N
rs = fd.route_stats_report()
per = rs["per_layer"]
print(f"\ndistinct routed experts per backbone layer, {rs['steps']} verify steps "
      f"(T_VERIFY={FD.T_VERIFY}, topk={eng.config()['routed_topk']}, so 6..12 possible)")
print(f"  mean {rs['mean']:.3f}   total over 40 layers {rs['total']:.2f}")
print("  per layer: " + " ".join(f"{v:.1f}" for v in per))
MB = eng.config()["expert_mb"]
gb = rs["total"] * MB / 1024
print(f"\nexpert bytes per verify step: {rs['total']:.2f} experts x {MB} MB = {gb:.3f} GB")
print(f"  (the 6-per-layer assumption used so far: {40*6*MB/1024:.3f} GB -- "
      f"{gb/(40*6*MB/1024):.2f}x too low)")
print(f"\nverify wall {verify_ms:.3f} ms/step")
# the two CB3 kernels were 31.03 ms of the profiled pair; recompute with the measured expert count
CB3_MS = float(os.environ.get("CB3_MS", "31.03"))
print(f"CB3 up+down measured {CB3_MS} ms/pair -> {gb/(CB3_MS/1000):.1f} GB/s of CB3 bytes")
