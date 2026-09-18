# DeepSeek-V4.1 Examples & Integration Guides

This directory contains production integration recipes, interactive demonstrations, and benchmark suites for DeepSeek-V4.1.

---

## 📁 Directory Structure

```
examples/
├── claude_code/             # Running Anthropic Claude Code with DeepSeek-V4.1
│   ├── dsv41-litellm.yaml   # LiteLLM proxy translation configuration
│   ├── claude-dsv41.sh      # Launch wrapper with environment variables & healthchecks
│   ├── verify_litellm.py    # Standalone verification suite for LiteLLM + DSV4.1
│   └── README.md            # Complete architecture & integration guide
│
├── jev_mode/                # High-Speed Non-Autoregressive Structured Output Engine
│   ├── demo_customer_feedback.py # Customer feedback analysis (sentiment, churn, urgency)
│   ├── demo_batch_extraction.py  # Batch ticket triage demonstrating GPU schema prefix caching
│   ├── demo_interactive.py       # Interactive CLI for instant extraction
│   └── README.md                 # 5-level prefix caching architecture guide
│
└── benchmarks/              # Performance Measurement & Load Testing
    ├── bench_jev_structured.py   # Normal JSON vs Jev Mode (84x-500x speedup)
    ├── bench_decode_throughput.py# Single-stream decode speed (34-39 tok/s)
    ├── bench_prefix_cache_hit.py # LCP cache acceleration test (cold vs warm prefill)
    ├── bench_concurrency.py      # Concurrent client load & scheduling benchmark
    └── README.md                 # Benchmark instructions & metric explanations
```

---

## 🚀 Quick Highlights

### 1. Claude Code Backend via LiteLLM
Connect Anthropic's official `claude` CLI directly to your local DeepSeek-V4.1 cluster:
```bash
# Verify connection
python3 examples/claude_code/verify_litellm.py

# Launch Claude Code
bash examples/claude_code/claude-dsv41.sh
```

### 2. Jev Mode: Parallel Structured Extraction
Extract complex schemas in **~24 ms** instead of ~4,200 ms:
```bash
# Run real-time customer feedback demo
python3 examples/jev_mode/demo_customer_feedback.py

# Run batch triage showing Level 2 GPU schema caching
python3 examples/jev_mode/demo_batch_extraction.py
```

### 3. Benchmarks
Evaluate performance across all dimensions:
```bash
# Run structured output comparison
python3 examples/benchmarks/bench_jev_structured.py

# Run decode speed throughput test
python3 examples/benchmarks/bench_decode_throughput.py
```
