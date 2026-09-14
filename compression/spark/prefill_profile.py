"""Kernel breakdown of one steady-state prefill chunk.

`prefill_probe.py` showed prefill costs 12.8 s fixed plus 13.1 ms a token, and that swapping the
MoE prefill kernel for a four-times slower one moves the total by only 28 % -- so the MoE kernels
are not where prefill time goes. This says where it does go.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))

from engine.v41_engine import V41Engine                        # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=False,
                arena_gb=float(os.environ.get("ARENA_GB", "94")),
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "400")),
                keep_free_gb=12.0, prune_keep=None,
                io_threads=int(os.environ.get("IO_THREADS", "12")),
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
ids = []
for line in open(os.path.expanduser("~/dsv41-spark/work/corpus/eval_ppl.jsonl")):
    ids += eng.tokenizer.encode(json.loads(line)["text"], add_special_tokens=False)
N = int(os.environ.get("NTOK", "512"))
t = torch.tensor(ids[:N], dtype=torch.long, device=eng.device)

for _ in range(2):                      # settle the arena
    eng._reset(); eng.model.begin_prompt()
    eng.model.forward(t, 0, prefill=True, need_logits=False)
torch.cuda.synchronize()

from torch.profiler import ProfilerActivity, profile           # noqa: E402
eng._reset(); eng.model.begin_prompt()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    eng.model.forward(t, 0, prefill=True, need_logits=False)
    torch.cuda.synchronize()
tab = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=22)
open(os.path.expanduser("~/dsv41-spark/work/results/prefill_profile.txt"), "w").write(tab)
print(tab)
