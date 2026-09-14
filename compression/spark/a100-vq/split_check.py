"""Does resident-first execution give the same answer? Prints a checksum of the prefill logits."""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))

from engine.v41_engine import V41Engine                        # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=False, arena_gb=float(os.environ.get("ARENA_GB", "94")),
                transient_slots=400, keep_free_gb=12.0, prune_keep=None,
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
ids = eng.tokenizer.encode(json.loads(open(os.path.expanduser(
    "~/dsv41-spark/work/corpus/eval_ppl.jsonl")).readline())["text"], add_special_tokens=False)
t = torch.tensor(ids[:int(os.environ.get("NTOK", "256"))], dtype=torch.long, device=eng.device)
eng._reset(); eng.model.begin_prompt()
logits, _ = eng.model.forward(t, 0, prefill=True, need_logits=True)
lg = logits.float()
print(f"SPLIT={os.environ.get('DSV41_SPLIT_MOE', '0')}  "
      f"sum {float(lg.sum()):.4f}  absmax {float(lg.abs().max()):.5f}  "
      f"argmax[-1] {int(lg[-1].argmax())}  top5 {lg[-1].topk(5).indices.tolist()}")
