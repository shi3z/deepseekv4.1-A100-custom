#!/usr/bin/env python3
"""Jev Mode Demo: Real-Time Customer Feedback Analysis.

Demonstrates parallel non-autoregressive structured output extraction.
Extracts multiple typed fields (sentiment, churn risk, urgency, needs_human)
in a single parallel scoring pass with ZERO autoregressive decode tokens.
"""

import json
import os
import sys
import time
import urllib.request

SERVER_URL = os.environ.get("DSV41_SERVER_URL", "http://127.0.0.1:8000")


def run_jev_analysis(text: str, schema: dict) -> dict:
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
    print("=" * 70)
    print("DeepSeek-V4.1 Jev Mode: Structured Output Extraction Demo")
    print(f"Target Server: {SERVER_URL}")
    print("=" * 70)

    # Sample customer review
    review = (
        "I have been a paying Pro user for 2 years. After the latest update, "
        "the export function completely crashes my browser. We have a major client "
        "presentation tomorrow morning and cannot export our work! Fix this immediately "
        "or we are canceling our enterprise migration."
    )

    # Typed Schema with candidate values
    schema = {
        "sentiment": ["positive", "neutral", "negative"],
        "churn_risk": ["low", "medium", "high"],
        "urgency": ["low", "medium", "high"],
        "category": ["billing", "bug", "feature_request", "performance", "usability"],
        "customer_tier": ["free", "pro", "enterprise"],
        "needs_human_manager": [True, False],
        "suggested_action": ["refund", "tech_escalation", "retention_call", "feature_log"],
    }

    print("\n[Input Customer Message]")
    print(f'"{review}"\n')

    print("[Defined Schema]")
    for k, v in schema.items():
        print(f"  - {k:<20}: {v}")

    print("\nSending Jev Mode request (parallel candidate scoring)...")
    res, total_ms = run_jev_analysis(review, schema)

    print("\n[Extraction Result]")
    if "choices" in res and res["choices"]:
        content = res["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except Exception:
            print(content)
    elif "result" in res:
        print(json.dumps(res["result"], indent=2, ensure_ascii=False))
    else:
        print(json.dumps(res, indent=2, ensure_ascii=False))

    usage = res.get("usage", {})
    prefill_tokens = usage.get("prompt_tokens", 0)
    compl_tokens = usage.get("completion_tokens", 0)

    print("\n[Performance Telemetry]")
    print(f"  - Total Round-Trip Time : {total_ms:.2f} ms")
    print(f"  - Prefill Tokens        : {prefill_tokens} tokens")
    print(f"  - Decode Output Tokens  : {compl_tokens} tokens (non-autoregressive)")
    if total_ms > 0 and compl_tokens > 0:
        eff_tps = compl_tokens / (total_ms / 1000.0)
        print(f"  - Effective Throughput  : {eff_tps:.1f} tokens/second")
    print("=" * 70)


if __name__ == "__main__":
    main()
