"""Decode tok/s and speculative accept length per dense-FP4 configuration, over several prompts.

ms/step alone cannot decide this: tok/s is accept_len / step_time, and a configuration that
degrades the model degrades its own DSpark draft acceptance too. `eng.last_stats` is the same
decode_tok_s the HTTP benchmark reports, so these numbers sit next to the 18.14 tok/s figure.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
from engine.v41_engine import V41Engine                        # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=94.0, transient_slots=64, keep_free_gb=12.0, prune_keep=None,
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
print(f"dense_fp4={eng.config()['dense_fp4']}  head={eng.config()['head_fmt']}", flush=True)

sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402

prompts = [json.loads(l) for l in open(os.environ.get("PROMPTS", os.path.expanduser("~/prompts.txt"))) if l.strip()]
enc = []
for p in prompts:
    s = encode_messages([{"role": "user", "content": p}], thinking_mode="chat")
    enc.append(eng.tokenizer.encode(s if isinstance(s, str) else s[0], add_special_tokens=False))

N = int(os.environ.get("GEN", "260"))
for p in range(int(os.environ.get("PASSES", "2"))):     # pass 0 warms the LRU; pass 1 is the number
    tps, acc = [], []
    for i, ids in enumerate(enc):
        tok = [t for burst in eng.generate(ids, max_tokens=N, temperature=0.0, ignore_eos=True)
               for t in burst]
        s = eng.last_stats
        tps.append(s["decode_tok_s"]); acc.append(s["accept_len_mean"])
        h = hashlib.md5(json.dumps(tok).encode()).hexdigest()[:12]
        print(f"pass{p} prompt{i+1}  {s['decode_tok_s']:6.2f} tok/s  accept {s['accept_len_mean']:.3f}  "
              f"steps {s['steps']:4d}  hit {s['expert_hit_rate']:.4f}  md5 {h}", flush=True)
    print(f"pass{p} MEAN   {sum(tps)/len(tps):6.3f} tok/s   accept {sum(acc)/len(acc):.3f}", flush=True)
