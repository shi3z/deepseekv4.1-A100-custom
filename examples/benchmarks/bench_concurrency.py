#!/usr/bin/env python3
"""Benchmark Suite: Concurrent Request Throughput.

Measures DeepSeek-V4.1 server concurrency scaling across multiple parallel clients
using concurrent.futures.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.request

SERVER_URL = os.environ.get("DSV41_SERVER_URL", "http://127.0.0.1:8000")


def send_single_query(req_id: int, max_tokens: int = 32) -> dict:
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": [
            {"role": "user", "content": f"Provide exactly 3 concise tips for software engineering topic #{req_id}."},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:
            if attempt == 2:
                raise exc
            time.sleep(0.1 * (attempt + 1))
    elapsed = time.perf_counter() - t0

    usage = res.get("usage", {})
    tokens = usage.get("completion_tokens", 0)
    return {
        "req_id": req_id,
        "elapsed_s": elapsed,
        "tokens": tokens,
        "tok_s": (tokens / elapsed) if elapsed > 0 else 0,
    }


def run_concurrency_test(num_workers: int, total_requests: int, max_tokens: int = 32):
    print(f"\n--- Testing Concurrency Level: {num_workers} Workers ({total_requests} total requests) ---")
    t0 = time.perf_counter()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(send_single_query, i + 1, max_tokens)
            for i in range(total_requests)
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    total_wall_time = time.perf_counter() - t0
    total_tokens = sum(r["tokens"] for r in results)
    avg_latency = sum(r["elapsed_s"] for r in results) / len(results)
    aggregate_tps = total_tokens / total_wall_time if total_wall_time > 0 else 0

    print(f"  Total Wall Time     : {total_wall_time:.2f} s")
    print(f"  Total Tokens        : {total_tokens} tokens")
    print(f"  Average Request Time: {avg_latency:.2f} s")
    print(f"  Aggregate Throughput: {aggregate_tps:.2f} tokens/second")

    return {
        "workers": num_workers,
        "requests": total_requests,
        "wall_time_s": total_wall_time,
        "total_tokens": total_tokens,
        "avg_latency_s": avg_latency,
        "aggregate_tps": aggregate_tps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--requests-per-worker", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()

    print("=" * 75)
    print("DeepSeek-V4.1 Server Concurrency Benchmark")
    print(f"Target Server: {SERVER_URL}")
    print("=" * 75)

    summary = []
    for w in args.workers:
        tot_req = w * args.requests_per_worker
        res = run_concurrency_test(w, tot_req, args.tokens)
        summary.append(res)

    print("\n" + "=" * 75)
    print("CONCURRENCY BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"{'Concurrency':<12} | {'Requests':<10} | {'Wall Time':<12} | {'Avg Latency':<14} | {'Aggregate TPS'}")
    print("-" * 75)
    for s in summary:
        print(
            f"{s['workers']:<12} | {s['requests']:<10} | {s['wall_time_s']:>6.2f} s    | "
            f"{s['avg_latency_s']:>6.2f} s       | {s['aggregate_tps']:>6.1f} tok/s"
        )
    print("=" * 75)


if __name__ == "__main__":
    main()
