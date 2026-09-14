"""Every fp32 tensor a verify step produces, ranked by the bytes it costs.

A decode step at T=2 moves ~11 GB of weights, so an fp32 activation is only worth chasing when it
is a weight read or a per-layer buffer, not when it is a 24-element constant. TorchDispatchMode
sees every aten op the eager path runs -- the same code the graphs were captured from -- and
records the op, its output dtype and size, and the engine frame that asked for it.
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
                arena_gb=float(os.environ.get("ARENA_GB", "40")), transient_slots=400,
                keep_free_gb=12.0, prune_keep=None, expert_format="cb3")
fd, m = eng.fast, eng.model
print(f"dense_fp4={eng.config()['dense_fp4']} head={eng.config()['head_fmt']} "
      f"T_VERIFY={FD.T_VERIFY}", flush=True)
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402
pr = encode_messages([{"role": "user", "content":
                       "日本の四季について、それぞれの季節の気候と代表的な行事を交えて400字程度で説明してください。"}],
                     thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in range(2):
    list(eng.generate(ids, max_tokens=64, temperature=0.0, ignore_eos=True))
pos = m.c.len
t0tok = int(ids[-1])
d, _ = fd.draft(t0tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([t0tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}

ENGINE = (os.path.expanduser("~/dsv41-spark/work/engine"),
          os.path.expanduser("~/dsv41-spark/work/tools"))


def site():
    for fr in reversed(traceback.extract_stack()[:-3]):
        if fr.filename.startswith(ENGINE):
            return f"{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}"
    return "?"


class Census(TorchDispatchMode):
    def __init__(self):
        self.rows = collections.defaultdict(lambda: [0, 0])     # key -> [bytes, count]
        self.on = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if self.on:
            for t in (out if isinstance(out, (tuple, list)) else (out,)):
                if isinstance(t, torch.Tensor) and t.dtype == torch.float32 and t.is_cuda:
                    k = (str(func).split(".")[-2] if "." in str(func) else str(func),
                         tuple(t.shape), site())
                    r = self.rows[k]
                    r[0] += t.numel() * 4
                    r[1] += 1
        return out


fd.use_graphs = False
m.c.len = pos
fd.step(block, pos, rows)                       # warm the eager path
torch.cuda.synchronize()
c = Census()
with c:
    c.on = True
    m.c.len = pos
    fd.step(block, pos, rows)
    fd.draft(t0tok, pos - 1, 0.0)
    c.on = False
torch.cuda.synchronize()
fd.use_graphs = True

tot = sum(v[0] for v in c.rows.values())
print(f"\nfp32 CUDA tensors produced in one (verify+draft): "
      f"{tot/2**20:.2f} MB over {sum(v[1] for v in c.rows.values())} ops\n")
print(f"{'MB':>8s} {'n':>5s} {'op':22s} {'shape':22s} call site")
for (op, sh, st_), (b, n) in sorted(c.rows.items(), key=lambda x: -x[1][0])[:32]:
    print(f"{b/2**20:8.3f} {n:5d} {op[:22]:22s} {str(sh)[:22]:22s} {st_}")
