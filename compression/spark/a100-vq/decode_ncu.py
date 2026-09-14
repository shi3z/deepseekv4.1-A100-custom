"""Device-side breakdown of one fully-resident decode step.

No pruning: the arena is warmed by generating the same prompt twice, which is the condition the
benchmark's warm rows measure (hit 1.0, no NVMe traffic). The step is the DSpark 6-token verify
block, replayed from its CUDA graph, so what the profiler sees is what the GPU actually runs.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))

from engine.v41_engine import V41Engine                      # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=float(os.environ.get("ARENA_GB", "94")),
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "64")),
                keep_free_gb=12.0, prune_keep=None,
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
print("config:", {k: eng.config()[k] for k in ("expert_format", "arena_slots", "kernel")}, flush=True)

sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                          # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in range(2):                       # warm the arena to hit 1.0
    for _ in eng.generate(ids, max_tokens=120, temperature=0.0):
        pass
st = eng.model.store.stats
print(f"store: hits {st['hits']} misses {st['misses']} -> hit {st['hits']/max(st['hits']+st['misses'],1):.4f}",
      flush=True)

m, fd = eng.model, eng.fast
pos = m.c.len
tok = int(ids[-1])
d, q = fd.draft(tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}


def one():
    m.c.len = pos
    fd.step(block, pos, rows)


N = 0
for _ in range(4):
    one()
torch.cuda.synchronize()

torch.cuda.synchronize()
print('warm done', flush=True)
