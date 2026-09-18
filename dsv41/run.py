"""Generate text with DeepSeek-V4.1-Flash on A100s.
usage: .venv-lc/bin/python -m dsv41.run --devices 0,1,2,3,4 --prompt "..." [--max-new-tokens 64]"""
import argparse
import os
import sys
import time

import os as _os
_os.environ.setdefault("OMP_WAIT_POLICY", "active")  # CPU expert threads keep spinning between layers (libgomp reads this once)
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsv41.load import load_model  # noqa: E402

DEFAULT_CKPT = "/mnt/ssd/models/DeepSeek-V4.1-Flash-Abliterated" if os.path.exists("/mnt/ssd/models/DeepSeek-V4.1-Flash-Abliterated") else "/mnt/ssd/models/DeepSeek-V4.1-Flash"
CKPT = os.environ.get("DSV41_CKPT", DEFAULT_CKPT)


def sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--devices", default="2,3,0,1")
    ap.add_argument("--budgets", default="", help="per-device GB overrides, e.g. 0:60,1:35")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt with the chat template")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--n-layers", type=int, default=None, help="load only the first N layers (plumbing test)")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--batch", type=int, default=1, help="decode B copies of the prompt together (throughput test)")
    ap.add_argument("--batch-prompts", default="", help="file with one prompt per line: distinct prompts for the batch rows (truncated to a common token length)")
    ap.add_argument("--ep", action="store_true", help="expert parallelism over --devices (dense pipelined, experts sharded; dsv41/ep.py)")
    ap.add_argument("--ep-shards", default="", help="experts per device for --ep, e.g. 82,82,68,38,38,38,38 (default: equal)")
    ap.add_argument("--offload-experts", nargs="?", const="cpu", default=False, choices=["gpu", "cpu"], help="single-GPU mode: experts in host RAM; 'cpu' computes them on the CPU (default), 'gpu' streams them over PCIe")
    ap.add_argument("--profile", action="store_true", help="per-component timing of the decode steps")
    ap.add_argument("--decode", default="graph", choices=["eager", "static", "graph"], help="decode path")
    ap.add_argument("--kernel-profile", action="store_true", help="after generation, profile 8 decode steps and list the top CUDA kernels")
    ap.add_argument("--kernel-trace", default="", help="after generation, trace one decode step (use --decode static) and write the chronological kernel list with the op behind each launch")
    ap.add_argument("--route-stats", default="", help="write per-layer expert hit counts (decode only) to this .pt file")
    ap.add_argument("--hot-experts", type=int, default=0, help="cpu offload mode: experts per layer kept on the GPU (by usage stats)")
    ap.add_argument("--hot-stats", default="", help="route stats .pt used to pick the hot experts (default: results/route_stats.pt)")
    ap.add_argument("--jev", action="store_true", help="enable Jev mode parallel non-autoregressive structured extraction")
    ap.add_argument("--mode", default="", help="inference mode: 'jev' or standard autoregressive")
    ap.add_argument("--schema", default="", help="schema JSON string or path to schema JSON file")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    devices = [int(d) for d in a.devices.split(",")]
    budgets = {int(k): float(v) for k, v in (kv.split(":") for kv in a.budgets.split(",") if kv)} or None
    model = load_model(a.ckpt, devices, max_seq_len=a.max_seq_len, max_batch=a.batch, budgets_gb=budgets, n_layers=a.n_layers,
                       engram=not a.no_engram, tokenizer=tok, offload_experts=a.offload_experts, hot_experts=a.hot_experts, route_stats=a.hot_stats, ep=a.ep,
                       ep_shards=[int(v) for v in a.ep_shards.split(",")] if a.ep_shards else None)

    if a.jev or a.mode == "jev":
        import json
        from dsv41.jev import JevEngine
        if not a.schema:
            raw_schema = {
                "sentiment": ["positive", "neutral", "negative"],
                "churn_risk": ["low", "medium", "high"],
                "urgency": ["low", "medium", "high"],
                "needs_human": [True, False],
            }
        elif os.path.exists(a.schema):
            raw_schema = json.load(open(a.schema))
        else:
            raw_schema = json.loads(a.schema)

        jev_eng = JevEngine(model, tok)
        assembled, metrics = jev_eng.process_request(a.prompt, raw_schema)
        print("\n=== Jev Mode Structured Output ===", flush=True)
        print(json.dumps(assembled, indent=2, ensure_ascii=False), flush=True)
        print("\n=== Jev Performance Metrics ===", flush=True)
        for k, v in metrics.items():
            print(f"  {k}: {v}", flush=True)
        return

    if a.chat:
        sys.path.insert(0, os.path.join(a.ckpt, "encoding"))
        from encoding import encode_messages  # type: ignore
        text = encode_messages([{"role": "user", "content": a.prompt}], thinking_mode="chat")
        ids = tok.encode(text)
    else:
        ids = tok.encode(a.prompt)
    if a.batch_prompts:
        lines = [l.strip() for l in open(a.batch_prompts) if l.strip()][: a.batch]
        assert len(lines) == a.batch, f"need {a.batch} prompts in {a.batch_prompts}"
        if a.chat:
            rows = [tok.encode(encode_messages([{"role": "user", "content": l}], thinking_mode="chat")) for l in lines]
        else:
            rows = [tok.encode(l) for l in lines]
        T = min(len(r) for r in rows)
        rows = [r[:T] for r in rows]  # lockstep positions: a common prompt length
        ids = rows[0]
        print(f"batch prompts: {a.batch} distinct, truncated to {T} tokens each", flush=True)
        input_ids = torch.tensor(rows, dtype=torch.long)
    else:
        input_ids = torch.tensor([ids] * a.batch, dtype=torch.long)
    print(f"prompt tokens: {len(ids)}", flush=True)

    rt = None
    if a.decode != "eager":
        # capture before the prefill: the warm-up/capture runs scribble on the caches at position 0,
        # and the prefill rewrites everything they touched
        from dsv41.decode import DecodeRuntime, OffloadDecodeRuntime
        if a.ep:
            from dsv41.ep import EPRuntime
            rt = EPRuntime(model, use_graphs=(a.decode == "graph"))
        elif a.offload_experts:
            rt = OffloadDecodeRuntime(model, use_graphs=(a.decode == "graph"))
        else:
            rt = DecodeRuntime(model, use_graphs=(a.decode == "graph"))
        if a.decode == "graph":
            tc = time.time()
            rt.capture()
            print(f"[captured {len(rt.graphs)} CUDA graphs in {time.time() - tc:.1f}s]", flush=True)
    torch.cuda.synchronize()
    t0 = time.time()
    logits = model.forward(input_ids, 0)
    torch.cuda.synchronize()
    t_prefill = time.time() - t0
    out = []
    pos = input_ids.size(1)
    if a.profile:
        from collections import defaultdict
        import dsv41.model as M
        M.PROF = defaultdict(float)
    nxt = sample(logits, a.temperature)
    if a.route_stats:
        import dsv41.model as M
        M.ROUTE_STATS = {}
    t1 = time.time()
    B = a.batch
    uniq, pairs_n = [], 0
    for _ in range(a.max_new_tokens):
        if B > 1 and getattr(rt, "route_log", False) and len(out) > 2:
            eid, _ = rt.route_snapshot()  # [layers, B, topk]
            uniq.append(sum(len(set(eid[l].reshape(-1).tolist())) for l in range(eid.shape[0])) / eid.shape[0])
            pairs_n = eid.shape[1] * eid.shape[2]
        toks = nxt.view(-1).tolist()
        if B > 1 and not a.batch_prompts and len(set(toks)) > 1 and not getattr(main, "_diverged", False):
            main._diverged = True
            print(f"\n[batch rows diverged at step {len(out)}: {toks[:8]}]", flush=True)
        out.append(toks[0])
        print(tok.decode(out[-1:]), end="", flush=True)
        if out[-1] == tok.eos_token_id and B == 1:
            break
        if rt is not None:
            logits = rt.step(toks if B > 1 else toks[0], pos)
        else:
            logits = model.forward(nxt.view(B, 1), pos)
        pos += 1
        nxt = sample(logits, a.temperature)
    torch.cuda.synchronize()
    t_dec = time.time() - t1
    print(f"\n\nprefill {len(ids)} tok in {t_prefill:.2f}s ({len(ids)/t_prefill:.1f} tok/s); decode {len(out)} tok in {t_dec:.2f}s ({len(out)/max(t_dec,1e-9):.2f} tok/s)"
          + (f"; batch {B}: {len(out) * B / max(t_dec, 1e-9):.1f} tok/s aggregate, {t_dec / len(out) * 1000:.1f} ms/step" if B > 1 else ""))
    if B > 1:
        rows = logits.argmax(-1).view(-1).tolist()
        print(f"[batch rows agree on the last argmax: {len(set(rows)) == 1} {rows[:4]}]")
        if uniq:
            u = sum(uniq) / len(uniq)
            print(f"[routing: {pairs_n} (token, expert) pairs per layer, {u:.1f} unique experts per layer -> {pairs_n / u:.2f} tokens per expert read]")
    if a.route_stats:
        import dsv41.model as M
        torch.save({k: v for k, v in M.ROUTE_STATS.items()}, a.route_stats)
        print(f"[route stats for {len(out)} decode tokens saved to {a.route_stats}]")
    if a.kernel_profile and rt is not None:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for i in range(8):
                rt.step(out[-1], pos + i)
            torch.cuda.synchronize()
        rows = [(e.key, e.device_time_total, e.count) for e in prof.key_averages() if e.device_time_total > 0]
        tot = sum(r[1] for r in rows)
        print(f"\n[kernel profile: {tot / 8 / 1000:.1f} ms of GPU time per token]")
        for k, t, c in sorted(rows, key=lambda r: -r[1])[:22]:
            print(f"  {t / 8 / 1000:7.2f} ms/token  {c // 8:5d}/token  {k[:90]}")
    if a.ep and os.environ.get("DSV41_EP_TRACE") == "1" and rt is not None:
        from dsv41.ep import trace_report
        rt.step(out[-1], pos)
        torch.cuda.synchronize()
        print("[EP timeline per layer, averaged over the 40 layers]\n" + trace_report(rt))
    if a.kernel_trace and rt is not None:
        import json
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
            rt.step(out[-1], pos)
            torch.cuda.synchronize()
        tmp = a.kernel_trace + ".json"
        prof.export_chrome_trace(tmp)
        ev = json.load(open(tmp))["traceEvents"]
        ops = {}
        for e in ev:
            if e.get("cat") == "cpu_op" and "External id" in e.get("args", {}):
                ops.setdefault(e["args"]["External id"], e)  # outermost op for the id
        ks = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
        t0 = ks[0]["ts"] if ks else 0
        with open(a.kernel_trace, "w") as f:
            f.write(f"# {len(ks)} GPU launches in one decode step ({len(model.blocks)} layers)\n")
            for e in ks:
                op = ops.get(e.get("args", {}).get("External id"))
                shapes = op["args"].get("Input Dims", "") if op else ""
                f.write(f"{e['ts'] - t0:9.1f} us  {e['dur']:6.1f} us  {e['name'][:60]:60s}  {op['name'][:40] if op else '?':40s} {str(shapes)[:80]}\n")
        print(f"[kernel trace: {len(ks)} launches written to {a.kernel_trace}]")
    if a.profile and rt is not None and getattr(rt, "cpu_experts", False):
        from collections import defaultdict
        rt.prof = defaultdict(float)
        for i in range(8):
            rt.step(out[-1], pos + i)
        tot = sum(rt.prof.values())
        print(f"[offload decode stages, per token; cold experts per token: {rt.n_cold / max(len(out) + 8, 1):.1f} of {40 * 6}]")
        for k, v in sorted(rt.prof.items(), key=lambda kv: -kv[1]):
            print(f"  {k:26s} {v / 8 * 1000:7.1f} ms  ({v / tot * 100:4.1f}%)")
        if getattr(rt, "hotcache", None) is not None:
            print("  " + rt.hotcache.stats())
        ct = sorted(rt.call_times)
        if ct:
            q = lambda f: ct[min(int(len(ct) * f), len(ct) - 1)] * 1e3
            print(f"  cpumoe_forward per call: median {q(0.5):.2f} ms  p90 {q(0.9):.2f}  p99 {q(0.99):.2f}  max {ct[-1] * 1e3:.2f}")
    if a.profile:
        import dsv41.model as M
        tot = sum(M.PROF.values())
        for k, v in sorted(M.PROF.items(), key=lambda kv: -kv[1]):
            print(f"  {k:10s} {v / max(len(out), 1) * 1000:7.1f} ms/token  ({v / tot * 100:4.1f}%)")


if __name__ == "__main__":
    main()
