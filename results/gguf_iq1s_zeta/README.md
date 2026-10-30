# DeepSeek-V4.1-Flash-1017 as a 1-bit GGUF on one Blackwell card (2026-10-31)

What [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) does on a
single workstation, for comparison with this repo's own 4x A100 runtime. Quantization **UD-IQ1_S**, the smallest one
the repo offers (5 shards, 216.97 GiB), which is what unsloth's "run on a 128GB+ RAM device" line refers to.

## Machine and build

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q, 97,887 MiB, compute 12.0, driver 580.178.04 |
| CPU | AMD Ryzen 9 9950X, 16 cores (benchmarks run with `-t 16`) |
| RAM | 123 GiB |
| model storage | NVMe, model read through mmap |
| llama.cpp | mainline `5047748` (version 0.8.0-dev), built with CUDA 12.9, `-DCMAKE_CUDA_ARCHITECTURES=120` |

Mainline llama.cpp already carries the `deepseek41` architecture, its engram tables, the DSpark draft model and a
dedicated KV cache (`src/llama-kv-cache-dsv4.h`), so no fork is needed.

Model metadata: 748.49 B parameters, 16 B active (`748B-A16B`), 40 blocks, hidden 5120, 64 attention heads with 1 KV
head, 384 experts of which 6 are used per token, expert FFN width 2304, context length 1,048,576, engram vocabulary
16,000,000 (compressed 99,092), DSpark draft model with 128 experts / 3 used.

**The model does not fit.** 216.97 GiB is 233 GB; VRAM plus RAM is 221 GB. Every configuration below is therefore
mmap'd, and whatever does not fit is read from NVMe as the tokens touch it. That the model still runs at tens of
tokens per second is a property of the architecture: only 6 of 384 experts fire per layer per token, and the 16M-entry
engram table is a sparse lookup.

## Throughput and memory (llama-bench, `-p 512 -n 128 -r 2`)

`-ncmoe K` keeps the expert tensors of the first K of 40 layers on the CPU side; everything else goes to the GPU.

| configuration | prefill pp512 (t/s) | decode tg128 (t/s) | peak VRAM | peak RSS |
|---|---|---|---|---|
| `-ngl 0` (no GPU, tg32) | — | 3.31 | — | 116 GB |
| `-ncmoe 40` (every expert on CPU) | 78.3 ± 9.3 | 14.51 ± 0.18 | 8.3 GB | 105 GB |
| `-ncmoe 24` | 157.4 ± 17.0 | 22.55 ± 0.60 | 56.4 GB | 115 GB |
| `-ncmoe 18` | 187.0 ± 16.5 | 25.56 ± 0.51 | 72.4 GB | 114 GB |
| `-ncmoe 14` | 217.6 ± 9.4 | 29.37 ± 0.35 | 82.5 GB | 116 GB |
| `-ncmoe 10` | 220.4 ± 7.6 | 34.92 ± 1.55 | 92.7 GB | 116 GB |
| **`-ncmoe 9` (best that fits)** | 213.3 ± 9.3 | **37.20 ± 1.23** | **95.2 GB** | 116 GB |
| `-ncmoe 8` | out of memory | — | 96.5 GB at failure | — |

Prefill scaling at `-ncmoe 10`, one repetition: 206 t/s at 512 tokens, 332 at 2048, 345 at 8192.

One real generation (`llama-cli -ncmoe 9 -c 4096 -n 256 --temp 0.3`): 33.3 t/s prompt, **35.0 t/s generation**, peak
VRAM 95.2 GB, peak RSS 116.3 GB. The answer was coherent, with the model's reasoning block intact, so the 1-bit
quantization is not obviously broken at this size.

## What the numbers say

- A 748B model generates at 37 t/s on one card plus 123 GB of host RAM. unsloth's "128GB+ RAM" claim holds, and the
  98 GB of VRAM here buys a 2.6x decode speedup over keeping every expert on the CPU (37.2 vs 14.5) and 11x over no
  GPU at all (37.2 vs 3.3).
- Moving experts to the GPU helps decode far more than prefill: prefill is flat from `-ncmoe 14` on (218 to 220 to
  213) while decode keeps climbing (29.4 to 34.9 to 37.2). Prefill is compute-bound, decode is memory-bound, which is
  the same split this repo's own runtime shows.
- RSS sits at 114-116 GB in every configuration that uses the GPU: the host side fills whatever RAM is free and the
  rest streams from NVMe. VRAM is the binding constraint, not RAM: 9 CPU layers is the limit, 8 fails to allocate.
- Prefill at 345 t/s for long prompts is where the comparison with this repo's runtime matters: the 4x A100 runtime
  does 250-300 t/s of cold prefill at 512K context with the full-precision model, so the single-card 1-bit build is
  in the same range for prefill while decoding faster (37 vs 45-55 t/s is actually slower, but on one card and at
  1/8 of the memory).

## Reproducing

Scripts are in `tools/gguf-bench/` of this repo (copied to `~/bench-gguf/` on the machine under test). Download is
`hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "UD-IQ1_S/*"`, 217 GiB at about 100 MB/s on zeta, so
roughly 40 minutes. Raw logs of every run in this directory: `results.log` (the sweep), `results2.log` (VRAM edge and
prefill scaling), `results3.log` (generation), and one `bench_*.log` per configuration.
