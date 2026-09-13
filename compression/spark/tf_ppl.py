"""Teacher-forced NLL per category for one expert format, on the Spark.

Same engine, same corpus, same settings for every format: only EXPERT_FORMAT and DSV41_CB3_STORE
change, so the difference between two runs is the quantiser and nothing else.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))

from engine.v41_engine import V41Engine                        # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=False,
                arena_gb=float(os.environ.get("ARENA_GB", "94")),
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "64")),
                keep_free_gb=float(os.environ.get("KEEP_FREE_GB", "12")),
                prune_keep=None,
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
print("config:", {k: eng.config()[k] for k in ("expert_format", "prune_keep", "arena_slots")},
      flush=True)
out = eng.teacher_forced(os.path.expanduser("~/dsv41-spark/work/corpus/eval_ppl.jsonl"),
                         max_len=512)
summary = out[0] if isinstance(out, tuple) else out
print("RESULT " + json.dumps(summary))
import math
for k, v in summary.items():
    if isinstance(v, dict) and "mean_nll" in v:
        print(f"  {k:10s} nll {v['mean_nll']:.4f}  ppl {math.exp(v['mean_nll']):.4f}  "
              f"top1 {v['top1_acc']:.4f}  n {v['n']}")
