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
└── benchmarks/              # Performance Measurement & Load Testing
    ├── bench_decode_throughput.py# Single-stream decode speed (34-39 tok/s)
    ├── bench_code_generation.py  # Python coding speed (52-53 tok/s) & test pass rate
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

### 2. Benchmarks
Evaluate performance across all dimensions:
```bash
# Run decode speed throughput test
python3 examples/benchmarks/bench_decode_throughput.py
```
