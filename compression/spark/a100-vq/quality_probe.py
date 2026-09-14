"""Is the model still the model? Teacher-forced NLL plus generation probes on the live config.

Every change adopted this session is either bit-exact (the fused qkv projection, the fused SwiGLU
and merge) or routing-preserving (the bf16-storage gate, 0 top-6 changes in 71,044 decisions), so
the teacher-forced numbers should land on the ones measured for this configuration BEFORE any of
them existed: wikitext 1.16853, code 0.53473. Anything else means one of those claims is wrong.
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
from engine.v41_engine import V41Engine                        # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=False,
                arena_gb=float(os.environ.get("ARENA_GB", "80")),
                transient_slots=400, keep_free_gb=12.0, prune_keep=None, io_threads=12,
                expert_format="cb3")
c = eng.config()
print(f"config: dense_fp4={c['dense_fp4']} head={c['head_fmt']} expert={c['expert_format']} "
      f"gate_kernel={FD.GATE_KERNEL} fused_qkv={hasattr(eng.model.W.layers[0], 'wqkv')}", flush=True)
import elem_fused as EF                                        # noqa: E402
print(f"        fused_swiglu={EF.SWIGLU} fused_merge={EF.MERGE}", flush=True)

out = eng.teacher_forced(os.path.expanduser("~/dsv41-spark/work/corpus/eval_ppl.jsonl"), max_len=512)
s = out[0] if isinstance(out, tuple) else out
REF = {"wikitext": 1.16853, "code": 0.53473}     # this config, measured before the optimisations
print("\nteacher-forced NLL (reference = same config, pre-optimisation)")
for k, v in s.items():
    if isinstance(v, dict) and "mean_nll" in v:
        r = REF.get(k)
        d = v["mean_nll"] - r if r else 0.0
        print(f"  {k:9s} nll {v['mean_nll']:.5f}  ppl {math.exp(v['mean_nll']):.4f}  "
              f"top1 {v['top1_acc']:.4f}  n {v['n']}   ref {r:.5f}  delta {d:+.5f} "
              f"({100*(math.exp(d)-1):+.3f} % ppl)", flush=True)
print("RESULT " + json.dumps(s))
