"""Where does a prefill chunk's time actually go: the MoE kernels, or reading experts off NVMe?

Runs the same 512-token prefill three times. The first pass pulls whatever the warm start did not
leave resident; by the third the arena holds the chunk's whole working set, so the difference is
the expert I/O and what is left is compute.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))

from engine.v41_engine import V41Engine                        # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=False,
                arena_gb=float(os.environ.get("ARENA_GB", "94")),
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "400")),
                keep_free_gb=float(os.environ.get("KEEP_FREE_GB", "12")),
                prune_keep=None,
                io_threads=int(os.environ.get("IO_THREADS", "12")),
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
print("config:", {k: eng.config()[k] for k in ("expert_format", "arena_slots", "io_threads", "read_threads")}, flush=True)

ids = []
for line in open(os.path.expanduser("~/dsv41-spark/work/corpus/eval_ppl.jsonl")):
    ids += eng.tokenizer.encode(json.loads(line)["text"], add_special_tokens=False)
N = int(os.environ.get("NTOK", "512"))
t = torch.tensor(ids[:N], dtype=torch.long, device=eng.device)
print(f"prefill of {t.numel()} tokens", flush=True)
store = eng.model.experts if hasattr(eng.model, "experts") else None


def diskstats():
    """sectors read on the NVMe, x512 bytes -- the ground truth for what a prefill actually reads."""
    for ln in open("/proc/diskstats"):
        f = ln.split()
        if f[2] == "nvme0n1":
            return int(f[5]) * 512
    return 0


def snap():
    a = getattr(eng, "expert_store", None) or getattr(eng.model, "expert_store", None)
    for obj in (a, store, getattr(eng, "experts", None)):
        if obj is not None and hasattr(obj, "stats"):
            return dict(obj.stats)
    return {}


for i in range(3):
    eng._reset()
    eng.model.begin_prompt()
    b0 = diskstats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    eng.model.forward(t, 0, prefill=True, need_logits=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gb = (diskstats() - b0) / 1e9
    print(f"pass {i+1}: {dt*1000:8.1f} ms   NVMe {gb:7.1f} GB   {gb/dt:5.2f} GB/s   "
          f"= {gb*1000/14.454784:.0f} expert records", flush=True)
    try:
        import cb3_store
        if cb3_store.PROBE is not None:
            print("   " + cb3_store.PROBE.report(), flush=True)
    except Exception:
        pass
