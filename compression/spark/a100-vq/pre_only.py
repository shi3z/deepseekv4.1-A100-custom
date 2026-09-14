"""Just the fixed 256-token prefill logits, for a config whose full NLL we already have.

tf_ppl.py grew the logits save after the FP8 baseline had already been measured, so the
baseline is the one config with NLL numbers but no ~/pre_*.pt. This runs only the prefill.
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
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "64")),
                keep_free_gb=float(os.environ.get("KEEP_FREE_GB", "12")),
                prune_keep=None,
                io_threads=int(os.environ.get("IO_THREADS", "12")),
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
print("config:", eng.config()["dense_fp4"], flush=True)
line = json.loads(open(os.path.expanduser("~/dsv41-spark/work/corpus/eval_ppl.jsonl")).readline())
ids = eng.tokenizer.encode(line["text"], add_special_tokens=False)[:256]
eng._reset(); eng.model.begin_prompt()
lg, _ = eng.model.forward(torch.tensor(ids, dtype=torch.long, device=eng.device), 0,
                          prefill=True, need_logits=True)
tag = os.environ.get("TAG", "base")
torch.save(lg.detach().float().cpu(), os.path.expanduser(f"~/pre_{tag}.pt"))
print(f"saved prefill logits {tuple(lg.shape)} -> ~/pre_{tag}.pt", flush=True)
