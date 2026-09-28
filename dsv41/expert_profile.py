"""Expert-routing profiler for the 4-GPU EP decode runtime (Phase 1 of the replica experiment: instrumentation only).

Runs plain batched decode (no MTP) on S sequences and records, per step and per layer, the routed expert ids
and weights of every row (the runtime's DSV41_ROUTE_LOG buffers, written inside the CUDA graphs), the EP
per-layer device timeline (DSV41_EP_TRACE stamps) and the free VRAM after graph capture. Everything is saved
to one .pt that dsv41/replica_plan.py turns into rankings, coverage curves and locality statistics.

usage: python -m dsv41.expert_profile --seqs 1 --steps 256 [--ckpt ... --devices 2,3,0,1 --ep-shards 97,97,97,93]
"""
from __future__ import annotations

import argparse, os, sys, time

os.environ.setdefault("DSV41_ROUTE_LOG", "1")
os.environ.setdefault("DSV41_EP_TRACE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash-Abliterated")
ap.add_argument("--devices", default="2,3,0,1")
ap.add_argument("--ep-shards", default="")
ap.add_argument("--seqs", type=int, default=1)
ap.add_argument("--steps", type=int, default=256)
ap.add_argument("--prompts", default="dsv41/batch_prompts64.txt")
ap.add_argument("--prompt-offset", type=int, default=0, help="first prompt line to use (different rows for different S)")
ap.add_argument("--max-seq-len", type=int, default=8192)
ap.add_argument("--out", default="")
ap.add_argument("--warmup", type=int, default=8)
a = ap.parse_args()

from transformers import AutoTokenizer
from dsv41.load import load_model
from dsv41.ep import EPRuntime, trace_report

tok = AutoTokenizer.from_pretrained(a.ckpt)
sys.path.insert(0, os.path.join(a.ckpt, "encoding"))
from encoding import encode_messages

S = a.seqs
devs = [int(d) for d in a.devices.split(",")]
shards = [int(v) for v in a.ep_shards.split(",")] if a.ep_shards else None
model = load_model(a.ckpt, devs, max_seq_len=a.max_seq_len, max_batch=S, max_seqs=S, engram=True, tokenizer=tok, ep=True, ep_shards=shards)
rt = EPRuntime(model, use_graphs=True)
rt.capture()
last = model.blocks[-1].device
nl = len(model.blocks)
E = model.args.n_routed_experts
topk = model.args.n_activated_experts

# ---- static facts: layer owners, expert shards, replica cost per expert, free VRAM after capture
owner_of_layer = [b.device.index for b in model.blocks]
shard_of_dev = {d.index: (s, n) for d, (s, n) in rt.shard.items()}
sh0 = model.blocks[0].ffn.ep[0]
bytes_per_expert = sum(int(sh0[k][0].numel()) * sh0[k][0].element_size() for k in ("w13", "s13", "w2", "s2"))
free_vram = {}
for d in rt.devs:
    torch.cuda.synchronize(d)
    fr, tot = torch.cuda.mem_get_info(d)
    free_vram[d.index] = {"free": int(fr), "total": int(tot), "torch_reserved": int(torch.cuda.memory_reserved(d)), "torch_allocated": int(torch.cuda.memory_allocated(d))}
print("[profile] layer owners:", owner_of_layer)
print("[profile] shards:", shard_of_dev)
print(f"[profile] bytes per expert per layer: {bytes_per_expert:,} ({bytes_per_expert / 2**20:.2f} MiB)")
for d, v in free_vram.items():
    print(f"[profile] cuda:{d} free {v['free'] / 2**30:.2f} GiB of {v['total'] / 2**30:.2f} (torch reserved {v['torch_reserved'] / 2**30:.2f}, allocated {v['torch_allocated'] / 2**30:.2f})")

# ---- prompts: one per sequence, prefilled into slot 0 and copied to its slot (like mtp_run)
lines = [l.strip() for l in open(a.prompts) if l.strip()]
lines = (lines[a.prompt_offset:] + lines[:a.prompt_offset])[:S]
assert len(lines) == S, (len(lines), S)
prompts = [tok.encode(encode_messages([{"role": "user", "content": l}], thinking_mode="chat")) for l in lines]
p_last, nxt = [0] * S, [0] * S
for s in range(S - 1, -1, -1):
    ids = prompts[s]
    logits = model.forward(torch.tensor([ids]), 0)
    if s > 0:
        rt.copy_seq(0, s)
    p_last[s] = len(ids) - 1
    nxt[s] = int(logits.argmax(-1))
torch.cuda.synchronize()

# ---- decode: per step keep eid/wt of every layer and the EP trace of the token
eids, wts, traces, step_ms = [], [], [], []
generated = [[] for _ in range(S)]
done = [False] * S
n_steps = a.warmup + a.steps
t_all = time.perf_counter()
for step in range(n_steps):
    poss = [p_last[s] + 1 for s in range(S)]
    t0 = time.perf_counter()
    logits = rt.step(nxt, poss, seq=list(range(S)), pmax=poss)
    am = logits.argmax(-1).tolist()
    dt = time.perf_counter() - t0
    if step >= a.warmup:
        eid, wt = rt.route_snapshot()  # [layers, S, topk] (or [layers, topk] when S == 1)
        eids.append(eid.reshape(nl, S, topk).to(torch.int16).clone())
        wts.append(wt.reshape(nl, S, topk).clone())
        if rt.trace is not None:
            traces.append(torch.stack([rt.trace[d].cpu() for d in rt.devs]))  # [nd, layers, 16] ns
        step_ms.append(dt * 1000)
    for s in range(S):
        if not done[s]:
            generated[s].append(am[s])
            if am[s] == tok.eos_token_id:
                done[s] = True
        nxt[s] = am[s]
        p_last[s] += 1
torch.cuda.synchronize()
t_all = time.perf_counter() - t_all
tot_tokens = S * a.steps
ms = sum(step_ms) / len(step_ms)
print(f"[profile] S={S}: {a.steps} steps, {ms:.2f} ms/step (incl. logging sync), {S * 1000 / ms:.1f} tok/s; wall {t_all:.1f}s")
print("[profile] sample:", tok.decode(generated[0][:40]).replace("\n", " ")[:160])
if rt.trace is not None:
    print("[EP timeline per layer, last step]\n" + trace_report(rt))

out = a.out or f"results/expert_profile_S{S}.pt"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
torch.save({
    "S": S, "steps": a.steps, "topk": topk, "n_experts": E, "n_layers": nl, "devices": devs,
    "owner_of_layer": owner_of_layer, "shard_of_dev": shard_of_dev, "bytes_per_expert": bytes_per_expert,
    "free_vram": free_vram, "dim": model.args.dim,
    "eid": torch.stack(eids),  # [steps, layers, S, topk] int16
    "wt": torch.stack(wts),  # [steps, layers, S, topk] fp32
    "trace": torch.stack(traces) if traces else None,  # [steps, nd, layers, 16] ns (globaltimer)
    "trace_devs": [d.index for d in rt.devs],
    "step_ms": step_ms, "ms_per_step": ms,
    "relay": rt.relay, "dma_route": rt.dma_route, "dma_part": rt.dma_part, "bf16_part": rt.bf16_part,
    "prompts": lines,
}, out)
print("[profile] saved", out)
