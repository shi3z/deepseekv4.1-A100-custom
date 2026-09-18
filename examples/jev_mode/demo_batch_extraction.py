#!/usr/bin/env python3
"""Jev Mode Demo: Batch Support Ticket Triage & Schema Reuse.

Demonstrates:
1. Schema Prefix Caching (Level 2 GPU cache):
   - First ticket pays ~1s schema prefill.
   - Subsequent tickets hit the cached GPU schema prefix instantly (0 schema prefill!).
2. Instant classification of complex tickets into priority, department, SLA breach risk, and root cause.
"""

import json
import os
import sys
import time
import urllib.request

SERVER_URL = os.environ.get("DSV41_SERVER_URL", "http://127.0.0.1:8000")

TICKETS = [
    {
        "id": "TICK-101",
        "text": "Production database cluster node db-03 disk space is at 98%. WAL archiving is falling behind.",
    },
    {
        "id": "TICK-102",
        "text": "User Jane Doe in marketing cannot access the analytics dashboard after SSO password reset.",
    },
    {
        "id": "TICK-103",
        "text": "Billing invoice #4928 shows double-charge for monthly enterprise seat expansion.",
    },
    {
        "id": "TICK-104",
        "text": "Security vulnerability scanner detected outdated OpenSSL on public edge reverse proxy.",
    },
    {
        "id": "TICK-105",
        "text": "Requesting additional 16GB RAM upgrade for staging Kubernetes worker node pool.",
    },
]

SCHEMA = {
    "priority": ["P0_CRITICAL", "P1_HIGH", "P2_MEDIUM", "P3_LOW"],
    "department": ["infrastructure", "security", "identity", "billing", "it_support"],
    "sla_breach_risk": ["critical", "moderate", "none"],
    "automated_action": ["page_oncall", "create_jira", "route_to_finance", "provision_resource"],
}


def send_jev_request(text: str, schema: dict) -> dict:
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": "deepseek-v4.1-flash",
        "prompt": text,
        "jev": True,
        "schema": schema,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return res, elapsed_ms


def main():
    print("=" * 75)
    print("DeepSeek-V4.1 Jev Mode: Batch Triage & Schema Cache Demonstration")
    print(f"Target Server: {SERVER_URL}")
    print(f"Total Tickets: {len(TICKETS)}")
    print("=" * 75)

    results = []
    total_tokens = 0
    total_time_ms = 0.0

    for i, t in enumerate(TICKETS, 1):
        print(f"\n[{i}/{len(TICKETS)}] Processing {t['id']}...")
        print(f"  Summary: \"{t['text'][:65]}...\"")
        
        res, elapsed_ms = send_jev_request(t["text"], SCHEMA)
        total_time_ms += elapsed_ms

        content = ""
        if "choices" in res and res["choices"]:
            content = res["choices"][0]["message"]["content"]
        elif "result" in res:
            content = json.dumps(res["result"])

        usage = res.get("usage", {})
        out_toks = usage.get("completion_tokens", 0)
        total_tokens += out_toks

        # Look for jev telemetry headers / metrics if provided
        jev_metrics = res.get("jev_metrics", {})
        cache_hit = jev_metrics.get("cache_hit", False)

        try:
            parsed = json.loads(content)
            results.append((t["id"], parsed, elapsed_ms, cache_hit))
            print(f"  Result: {json.dumps(parsed)}")
        except Exception:
            results.append((t["id"], content, elapsed_ms, cache_hit))
            print(f"  Raw: {content}")

        print(f"  Latency: {elapsed_ms:.1f} ms | Schema Cache Hit: {cache_hit}")

    print("\n" + "=" * 75)
    print("Batch Execution Summary")
    print("=" * 75)
    print(f"{'Ticket ID':<12} | {'Priority':<12} | {'Department':<14} | {'Latency':<10} | {'Schema Cache'}")
    print("-" * 75)
    for tid, parsed, lat, hit in results:
        if isinstance(parsed, dict):
            pri = str(parsed.get("priority", "N/A"))
            dep = str(parsed.get("department", "N/A"))
        else:
            pri, dep = "ERR", "ERR"
        hit_str = "HIT (Fast)" if hit else "MISS (Warmup)"
        print(f"{tid:<12} | {pri:<12} | {dep:<14} | {lat:>6.1f} ms | {hit_str}")
    print("-" * 75)
    print(f"Total Elapsed Time: {total_time_ms:.1f} ms")
    print(f"Average Latency   : {total_time_ms / len(TICKETS):.1f} ms / request")
    if total_tokens > 0:
        overall_tps = total_tokens / (total_time_ms / 1000.0)
        print(f"Effective Throughput: {overall_tps:.1f} tok/s")
    print("=" * 75)


if __name__ == "__main__":
    main()
