"""Greedy generation with and without speculation, same weights, same prompt.

Speculative decoding verifies every drafted token, so the two must agree exactly. NOTES.md
2026-09-11 records a configuration where they did not -- fused attention + dense fp4 + fp8 head
crossed the precision the verify step needed and greedy decoding fell into a repetition loop --
and teacher-forced loss, the gate used everywhere else, cannot see that class of fault.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
from engine.v41_engine import V41Engine                        # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
spec = os.environ.get("SPEC", "1") == "1"
eng = V41Engine(md, max_seq=8192, spec=spec,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=94.0, transient_slots=64, keep_free_gb=12.0, prune_keep=None,
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
c = eng.config()
print(f"spec={spec} dense_fp4={c['dense_fp4']} head={c['head_fmt']}", flush=True)
sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages                           # noqa: E402

out = {}
for i, p in enumerate([json.loads(l) for l in open(os.path.expanduser("~/prompts20.txt"))][:6]):
    s = encode_messages([{"role": "user", "content": p}], thinking_mode="chat")
    ids = eng.tokenizer.encode(s if isinstance(s, str) else s[0], add_special_tokens=False)
    tok = [t for b in eng.generate(ids, max_tokens=200, temperature=0.0, ignore_eos=True) for t in b]
    out[i] = tok
    # a run of one token repeated is the failure mode that matters, so measure it directly
    longest = best = 1
    for j in range(1, len(tok)):
        longest = longest + 1 if tok[j] == tok[j - 1] else 1
        best = max(best, longest)
    print(f"prompt{i+1} md5 {hashlib.md5(json.dumps(tok).encode()).hexdigest()[:12]} "
          f"n {len(tok)} distinct {len(set(tok))} longest_repeat {best}", flush=True)
json.dump(out, open(os.path.expanduser(f"~/gen_spec{int(spec)}.json"), "w"))
