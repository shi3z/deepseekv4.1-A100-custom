"""Is the fused attention kernel safe in THIS configuration, and what is it worth?

fastdecode.py keeps DSV41_FUSED_ATTN off because, with fp4 dense projections and an fp8 head,
greedy decoding through it diverged and could fall into a repetition loop. This engine runs fp8
dense and a bf16 head, so the combination the note warns about is not the one in play. The fp32
attention einsums it replaces are 4.7 ms of a 86.9 ms step, and the `.float()` casts feeding them
are most of another 3.5.

Prints the generated token ids (so the two settings can be diffed exactly) and the step time.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
from engine.v41_engine import V41Engine                        # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=94.0, transient_slots=64, keep_free_gb=12.0, prune_keep=None,
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
import engine.fastdecode as FD                                 # noqa: E402
print(f"FUSED_ATTN={FD.FUSED_ATTN}  dense_fp4={eng.config()['dense_fp4']}  "
      f"kernel={eng.config()['kernel']}  arena_slots={eng.config()['arena_slots']}", flush=True)

sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)

tok = []
for _ in range(2):                      # warm the arena to hit 1.0; generate yields BURSTS of ids
    tok = [i for burst in eng.generate(ids, max_tokens=160, temperature=0.0, ignore_eos=True)
           for i in burst]
st = eng.model.store.stats
h = hashlib.md5(json.dumps(tok).encode()).hexdigest()[:16]
print(f"hit {st['hits']/max(st['hits']+st['misses'],1):.4f}  tokens {len(tok)}  md5 {h}", flush=True)
print("first 24 ids:", tok[:24], flush=True)

m, fd = eng.model, eng.fast
pos = m.c.len
t0tok = int(ids[-1])
d, q = fd.draft(t0tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}


def one():
    m.c.len = pos
    fd.step(block, pos, rows)


# the step's logits, for an exact numerical comparison between the two settings
m.c.len = pos
lg, _ = fd.step(block, pos, rows)
torch.save({"logits": lg.detach().float().cpu(), "tokens": tok},
           os.path.expanduser(f"~/dec_{os.environ.get('TAG', 'base')}.pt"))
print("saved logits", tuple(lg.shape), flush=True)

for _ in range(20):
    one()
torch.cuda.synchronize()
N = int(os.environ.get("STEPS", "200"))
ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(N)]
for a, b in ev:
    a.record(); one(); b.record()
torch.cuda.synchronize()
ms = sorted(a.elapsed_time(b) for a, b in ev)
print(f"step: median {ms[N//2]:.3f} ms  p95 {ms[int(N*0.95)]:.3f}  min {ms[0]:.3f}", flush=True)
