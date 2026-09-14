"""Shape-by-shape profile of the decode-sized fp8 dense projections.

The decode step issues 294 of these per step and they are 33 % of it at an effective 145 GB/s,
where the CB3 expert kernel on the same GPU reaches 187. The two dominant shapes are very
different -- wq_b [32768, 1280] launches 256 programs over a 10-step K loop, wo_b [5120, 8192]
launches 40 over a 64-step one -- so they are measured apart.
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
import torch
from fp8_linear import FP8Weight, fp8_linear
props = torch.cuda.get_device_properties(0)
print(f"device: {props.name}, {props.multi_processor_count} SMs, "
      f"{props.total_memory/2**30:.1f} GiB", flush=True)
import json, time
from safetensors import safe_open
md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
idx = json.load(open(f"{md}/model.safetensors.index.json"))["weight_map"]
f = safe_open(f"{md}/{idx['layers.0.attn.wq_b.weight']}", "pt", device="cpu")
torch.manual_seed(0)
for name in ("layers.0.attn.wq_b", "layers.0.attn.wo_b", "layers.0.attn.wq_a", "layers.0.ffn.shared_experts.w1", "layers.0.ffn.shared_experts.w2"):
    W = FP8Weight(f.get_tensor(name + ".weight").cuda(), f.get_tensor(name + ".scale").cuda())
    ref_w = W.dequant()
    for M in (6,):
        x = (torch.randn(M, W.K, device="cuda") * 0.5).to(torch.bfloat16)
        y = fp8_linear(x, W)
        r = torch.nn.functional.linear(x, ref_w)
        rel = float((y.float() - r.float()).norm() / r.float().norm())
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20): fp8_linear(x, W)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 20
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20): torch.nn.functional.linear(x, ref_w)
        torch.cuda.synchronize(); dt2 = (time.perf_counter() - t0) / 20
        print(f"{name:34s} N={W.N:6d} K={W.K:5d} M={M:3d}: rel {rel:.2e}  fp8 {dt*1e3:6.3f} ms ({W.N*W.K/dt/1e9:5.0f} GB/s of fp8)  bf16 {dt2*1e3:6.3f} ms ({W.N*W.K*2/dt2/1e9:5.0f} GB/s)")
