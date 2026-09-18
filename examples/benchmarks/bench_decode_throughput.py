#!/usr/bin/env python3
"""Benchmark Suite: Autoregressive Decode Throughput.

Measures DeepSeek-V4.1 generation speed (tok/s) across multiple token generation
lengths (64, 128, 256, 512 tokens) and prompt lengths (short vs medium).
"""

import argparse
import json
import os
import sys
import time
import urllib.request

SERVER_URL = os.environ.get("DSV41_SERVER_URL", "http://127.0.0.1:8000")


def generate(prompt: str, max_tokens: int) -> dict:
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    total_time = time.perf_counter() - t0

    usage = res.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    compl_tokens = usage.get("completion_tokens", 0)
    tok_s = (compl_tokens / total_time) if total_time > 0 else 0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": compl_tokens,
        "total_time_s": total_time,
        "tok_s": tok_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    args = parser.parse_args()

    print("=" * 75)
    print("DeepSeek-V4.1 Generation Throughput Benchmark (Single Stream)")
    print(f"Target Server: {SERVER_URL}")
    print("=" * 75)

    # Warmup
    print("Warming up server...")
    try:
        generate("Warmup test.", 16)
    except Exception as e:
        print(f"Warmup error: {e}")

    results = []

    # 1. Short Prompt Benchmarks
    short_prompt = "Write a comprehensive essay describing the history and architecture of relational databases."
    print(f"\n[Test Group 1: Short Prompt ({len(short_prompt.split())} words)]")
    for max_tok in args.lengths:
        print(f"  Requesting {max_tok} tokens max...")
        res = generate(short_prompt, max_tok)
        print(f"    Generated {res['completion_tokens']} tokens in {res['total_time_s']:.2f}s -> {res['tok_s']:.2f} tok/s")
        results.append({
            "group": "Short Prompt",
            "requested": max_tok,
            "generated": res["completion_tokens"],
            "time_s": res["total_time_s"],
            "tok_s": res["tok_s"],
        })

    # 2. Medium Prompt Benchmarks
    medium_prompt = (
        "Here is a detailed system specification for an e-commerce microservices platform:\n"
        + "1. User authentication service using OAuth2 and JWT tokens with Redis session storage.\n"
        + "2. Product catalog service backed by PostgreSQL with Elasticsearch full-text search.\n"
        + "3. Order processing service with Kafka message queue and distributed saga orchestrator.\n"
        + "4. Payment gateway integration supporting Stripe and PayPal webhooks with idempotency keys.\n"
        + "5. Notification service delivering real-time WebSockets and transactional email via SES.\n"
        + "Please write an in-depth technical analysis explaining failover mechanisms, fault tolerance, and disaster recovery strategies for this architecture."
    )
    print(f"\n[Test Group 2: Medium Prompt (~100 words)]")
    for max_tok in [128, 256]:
        print(f"  Requesting {max_tok} tokens max...")
        res = generate(medium_prompt, max_tok)
        print(f"    Generated {res['completion_tokens']} tokens in {res['total_time_s']:.2f}s -> {res['tok_s']:.2f} tok/s")
        results.append({
            "group": "Medium Prompt",
            "requested": max_tok,
            "generated": res["completion_tokens"],
            "time_s": res["total_time_s"],
            "tok_s": res["tok_s"],
        })

    # Summary
    print("\n" + "=" * 75)
    print("THROUGHPUT BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Prompt Type':<16} | {'Target Tok':<12} | {'Actual Tok':<12} | {'Time (s)':<10} | {'Throughput'}")
    print("-" * 75)
    for r in results:
        print(f"{r['group']:<16} | {r['requested']:<12} | {r['generated']:<12} | {r['time_s']:>6.2f}s    | {r['tok_s']:>5.1f} tok/s")
    print("=" * 75)


if __name__ == "__main__":
    main()
