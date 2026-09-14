"""The ceiling on future-routing prefetch: record the routing, then replay it as an oracle.

Pass 1-2 settle the arena. Pass 3 is the control. Pass 4 runs with the routing of pass 3 handed to
the model, so every layer's experts start loading one layer early -- which is exactly what a
perfect predictor would achieve and no predictor can beat.
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


def diskstats():
    for ln in open("/proc/diskstats"):
        f = ln.split()
        if f[2] == "nvme0n1":
            return int(f[5]) * 512
    return 0


md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=False, arena_gb=float(os.environ.get("ARENA_GB", "94")),
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "400")),
                keep_free_gb=12.0, prune_keep=None,
                io_threads=int(os.environ.get("IO_THREADS", "12")),
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
ids = []
for line in open(os.path.expanduser("~/dsv41-spark/work/corpus/eval_ppl.jsonl")):
    ids += eng.tokenizer.encode(json.loads(line)["text"], add_special_tokens=False)
t = torch.tensor(ids[:int(os.environ.get("NTOK", "512"))], dtype=torch.long, device=eng.device)
m = eng.model
print(f"prefill of {t.numel()} tokens, oracle prefetch one layer ahead", flush=True)

route = {}


def tap(name, L, x):
    if name == "route_idx":
        route[L] = x.clone()


def run(label, oracle=None, record=False):
    m.oracle = oracle
    m.tap = tap if record else None
    eng._reset(); m.begin_prompt()
    b0 = diskstats(); torch.cuda.synchronize(); t0 = time.perf_counter()
    m.forward(t, 0, prefill=True, need_logits=False)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    gb = (diskstats() - b0) / 1e9
    m.tap = None; m.oracle = None
    print(f"  {label:22s} {dt*1000:8.1f} ms   NVMe {gb:6.1f} GB   {gb/dt:5.2f} GB/s", flush=True)
    return dt


run("warm 1")
run("warm 2")
base = run("control (record)", record=True)
print(f"  recorded routing for {len(route)} layers", flush=True)
orc = run("oracle prefetch", oracle=route)
print(f"  oracle vs control: {(orc-base)/base*100:+.1f} %", flush=True)
