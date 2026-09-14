"""Where the 2,710 aten::copy_ calls of a verify step come from.

The first attempt keyed ops with str(func).split("::") -- an OpOverload prints as `aten.copy_.default`
with no "::", so every op collapsed to "aten" and the site table came out empty. This keys on the
overload packet and records the engine frame, the shape and the dtype pair, which is what says
whether a copy is a dtype cast, a ring store or a buffer shuffle.
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
COPYISH = {"copy_", "_to_copy", "clone", "cat", "index_put_", "contiguous", "to"}


def site():
    out = []
    for fr in reversed(traceback.extract_stack()[:-3]):
        if fr.filename.startswith(ENGINE):
            out.append(f"{os.path.basename(fr.filename)}:{fr.lineno}")
            if len(out) == 2:
                break
    return " <- ".join(out) if out else "?"


class C(TorchDispatchMode):
    def __init__(self):
        self.n = collections.Counter()
        self.b = collections.Counter()
        self.all = collections.Counter()
        self.on = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if self.on:
            name = getattr(func, "overloadpacket", func)
            name = getattr(name, "__name__", str(func)).lstrip("aten.")
            self.all[name] += 1
            if name in COPYISH:
                t = args[0] if args and isinstance(args[0], torch.Tensor) else None
                src = args[1] if len(args) > 1 and isinstance(args[1], torch.Tensor) else None
                sh = tuple(t.shape) if t is not None else ()
                dt = (f"{src.dtype}".replace("torch.", "") + "->" if src is not None else "") + \
                     (f"{t.dtype}".replace("torch.", "") if t is not None else "")
                k = (name, site(), sh, dt)
                self.n[k] += 1
                if t is not None:
                    self.b[k] += t.numel() * t.element_size()
        return out


fd.use_graphs = False
m.c.len = pos
fd.step(block, pos, rows)
torch.cuda.synchronize()
c = C()
with c:
    c.on = True
    m.c.len = pos
    fd.step(block, pos, rows)
    fd.draft(t0, pos - 1, 0.0)
    c.on = False
torch.cuda.synchronize()
fd.use_graphs = True

tot = sum(c.n.values())
print(f"\ncopy-like calls in one (verify+draft): {tot}   "
      f"total aten calls {sum(c.all.values())}")
print(f"by op: " + "  ".join(f"{k}={c.all[k]}" for k in sorted(COPYISH) if c.all[k]))
print(f"\n{'op':12s} {'calls':>6s} {'KB':>9s} {'shape':18s} {'dtype':18s} call site")
for (op, st_, sh, dt), n in c.n.most_common(26):
    print(f"{op:12s} {n:6d} {c.b[(op,st_,sh,dt)]/1024:9.1f} {str(sh)[:18]:18s} {dt[:18]:18s} {st_}")
