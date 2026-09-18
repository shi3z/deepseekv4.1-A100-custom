# DeepSeek-V4.1 Performance & Benchmark Suite

This directory contains standalone benchmark tools to evaluate inference performance, decoding speed, prefix cache reuse, and parallel non-autoregressive extraction on DeepSeek-V4.1.

---

## 📊 Benchmark Tools

### 1. Structured Output Benchmark (`bench_jev_structured.py`)
Compares standard autoregressive JSON decoding against **Jev Mode** parallel candidate scoring across 4 schema complexity cases (3, 10, 30, and 100 fields).

```bash
# Run all 4 cases
python3 examples/benchmarks/bench_jev_structured.py

# Run specific cases
python3 examples/benchmarks/bench_jev_structured.py --cases "Case A" "Case B"
```

**What it measures:**
- Normal JSON round-trip time vs Jev Mode round-trip time.
- Speedup multiplier ($84\times$ to $500\times$).
- Output tokens and effective token generation throughput.

---

### 2. Autoregressive Decode Throughput (`bench_decode_throughput.py`)
Measures single-stream generation throughput (tokens/sec) across different generated token lengths (64, 128, 256, 512 tokens).

```bash
python3 examples/benchmarks/bench_decode_throughput.py --lengths 64 128 256 512
```

**What it measures:**
- Decode latency (seconds) and token generation speed (~34–39 tok/s single-stream decode).
- Impact of short vs medium prompt prefill overhead on overall throughput.

---

### 3. Prefix Cache Hit Acceleration (`bench_prefix_cache_hit.py`)
Demonstrates the performance impact of Longest Common Prefix (LCP) caching during multi-turn developer interactions (such as Claude Code sessions or multi-step agent tool loops).

```bash
python3 examples/benchmarks/bench_prefix_cache_hit.py
```

**What it measures:**
- **Turn 1 (Cold Prefill / Cache MISS)**: Time to evaluate and store a large system prompt (~1,500 tokens).
- **Turn 2 (Shared Prefix / Cache HIT)**: Latency when reusing the GPU-cached prefix ($3\times$ to $6\times$ faster TTFT).
- **Turn 3 (Multi-Turn Continuation)**: Incremental cache hit as conversation grows.

---

### 4. Server Concurrency & Load Test (`bench_concurrency.py`)
Tests how the server scales under concurrent client loads (1, 2, and 4 parallel workers).

```bash
python3 examples/benchmarks/bench_concurrency.py --workers 1 2 4 --requests-per-worker 2
```

**What it measures:**
- Server batch scheduling efficiency.
- Aggregate throughput (combined tokens/sec across workers).
- Average client latency under concurrent queue pressure.

---

## ⚙️ Environment Variables

All benchmark scripts respect the following environment configuration:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DSV41_SERVER_URL` | `http://127.0.0.1:8000` | Target DeepSeek-V4.1 OpenAI-compatible server URL |
