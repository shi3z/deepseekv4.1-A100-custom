#!/usr/bin/env python3
"""Benchmark Suite: Normal Autoregressive JSON vs Jev Mode Parallel Extraction.

Evaluates 4 schema complexity tiers against DeepSeek-V4.1:
  Case A: 3 fields
  Case B: 10 fields
  Case C: 30 fields
  Case D: 100 fields

Measures:
  - Latency (ms)
  - Completion tokens
  - Generation throughput (tok/s)
  - Speedup factor
  - Prefix cache hit latency
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

SERVER_URL = os.environ.get("DSV41_SERVER_URL", "http://127.0.0.1:8000")

BASE_PROMPT = (
    "The customer says the service is too expensive and they are "
    "considering cancelling unless somebody contacts them today."
)

CASES = {
    "Case A (3 fields)": {
        "sentiment": ["positive", "neutral", "negative"],
        "churn_risk": ["low", "medium", "high"],
        "urgency": ["low", "medium", "high"],
    },
    "Case B (10 fields)": {
        "sentiment": ["positive", "neutral", "negative"],
        "churn_risk": ["low", "medium", "high"],
        "urgency": ["low", "medium", "high"],
        "needs_human": [True, False],
        "priority": ["p0", "p1", "p2", "p3"],
        "category": ["billing", "technical", "sales", "account", "feedback"],
        "customer_tier": ["free", "pro", "enterprise"],
        "escalation_needed": [True, False],
        "resolution_difficulty": ["easy", "medium", "hard"],
        "contact_channel": ["email", "chat", "phone", "ticket"],
    },
}

# Construct Case C (30 fields)
case_c = dict(CASES["Case B (10 fields)"])
extra_20 = {
    "department": ["support", "finance", "legal", "product", "security"],
    "language": ["english", "spanish", "japanese", "german", "french"],
    "product_area": ["pricing", "ui", "api", "infra", "auth"],
    "root_cause": ["cost", "bug", "performance", "missing_feature", "usability"],
    "satisfaction": ["very_low", "low", "neutral", "high", "very_high"],
    "competitor_mentioned": [True, False],
    "feature_request": [True, False],
    "pricing_issue": [True, False],
    "contract_type": ["monthly", "annual", "custom"],
    "churn_timeframe": ["immediate", "within_month", "future", "none"],
    "action_required": [True, False],
    "assigned_team": ["retention", "billing_ops", "tier2_support", "account_exec"],
    "follow_up_method": ["call", "email", "in_app_message"],
    "retention_offer": ["discount_10", "discount_20", "free_month", "none"],
    "customer_tenure": ["new", "medium", "long", "veteran"],
    "nps_category": ["promoter", "passive", "detractor"],
    "executive_sponsor": [True, False],
    "usage_trend": ["increasing", "flat", "declining"],
    "support_history": ["clean", "occasional_issues", "frequent_issues"],
    "renewal_risk": ["low", "medium", "high"],
}
case_c.update(extra_20)
CASES["Case C (30 fields)"] = case_c

# Construct Case D (100 fields)
case_d = dict(case_c)
for i in range(1, 71):
    case_d[f"extra_attr_{i:02d}"] = ["val_a", "val_b", "val_c", "val_d"]
CASES["Case D (100 fields)"] = case_d


def query_normal_json(prompt: str, schema: dict) -> dict:
    url = f"{SERVER_URL}/v1/chat/completions"
    system_msg = (
        "You are an extraction assistant. Extract information strictly as valid JSON "
        f"matching this schema: {json.dumps(schema)}. Do not output any markdown or thinking, only JSON."
    )
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1500,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    usage = res.get("usage", {})
    compl_tokens = usage.get("completion_tokens", 0)
    tok_s = (compl_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0

    return {
        "elapsed_ms": elapsed_ms,
        "completion_tokens": compl_tokens,
        "tok_s": tok_s,
        "content": res["choices"][0]["message"]["content"],
    }


def query_jev_mode(prompt: str, schema: dict) -> dict:
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": "deepseek-v4.1-flash",
        "prompt": prompt,
        "jev": True,
        "schema": schema,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    usage = res.get("usage", {})
    compl_tokens = usage.get("completion_tokens", 0)
    tok_s = (compl_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0

    return {
        "elapsed_ms": elapsed_ms,
        "completion_tokens": compl_tokens,
        "tok_s": tok_s,
        "jev_metrics": res.get("jev_metrics", {}),
        "jev_result": res.get("jev_result", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["Case A", "Case B", "Case C", "Case D"])
    parser.add_argument("--warmup", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 80)
    print("DeepSeek-V4.1 Structured Output Benchmark: Normal JSON vs Jev Mode")
    print(f"Target Server: {SERVER_URL}")
    print("=" * 80)

    results = []

    for name, schema in CASES.items():
        matched = any(c.lower() in name.lower() for c in args.cases)
        if not matched:
            continue

        num_fields = len(schema)
        print(f"\n>>> Running {name} ({num_fields} fields)...")

        # 1. Normal Autoregressive JSON
        print("  Evaluating Normal Autoregressive JSON...")
        try:
            norm = query_normal_json(BASE_PROMPT, schema)
            print(f"    Normal JSON: {norm['elapsed_ms']:.1f} ms | {norm['completion_tokens']} tok | {norm['tok_s']:.1f} tok/s")
        except Exception as e:
            print(f"    Normal JSON failed: {e}")
            norm = {"elapsed_ms": 0, "completion_tokens": 0, "tok_s": 0}

        # 2. Warmup Jev (prefill schema)
        print("  Warming up Jev Mode schema cache...")
        try:
            query_jev_mode(BASE_PROMPT, schema)
        except Exception as e:
            print(f"    Warmup failed: {e}")

        # 3. Measured Jev Mode (GPU Cache Hit)
        print("  Evaluating Jev Mode (Cached GPU Schema)...")
        try:
            jev = query_jev_mode(BASE_PROMPT, schema)
            print(f"    Jev Mode   : {jev['elapsed_ms']:.1f} ms | {jev['completion_tokens']} tok | {jev['tok_s']:.1f} tok/s")
        except Exception as e:
            print(f"    Jev Mode failed: {e}")
            jev = {"elapsed_ms": 0, "completion_tokens": 0, "tok_s": 0}

        speedup = (norm["elapsed_ms"] / jev["elapsed_ms"]) if jev["elapsed_ms"] > 0 else 0
        results.append({
            "name": name,
            "fields": num_fields,
            "norm_ms": norm["elapsed_ms"],
            "norm_tok": norm["completion_tokens"],
            "norm_tps": norm["tok_s"],
            "jev_ms": jev["elapsed_ms"],
            "jev_tok": jev["completion_tokens"],
            "jev_tps": jev["tok_s"],
            "speedup": speedup,
        })

    # Summary Table
    print("\n" + "=" * 95)
    print("BENCHMARK RESULTS SUMMARY: Normal JSON vs Jev Mode")
    print("=" * 95)
    header = f"{'Case':<20} | {'Normal JSON Latency':<20} | {'Jev Mode Latency':<18} | {'Speedup':<10} | {'Jev tok/s'}"
    print(header)
    print("-" * 95)
    for r in results:
        norm_str = f"{r['norm_ms']:.1f} ms ({r['norm_tok']} tok)"
        jev_str = f"{r['jev_ms']:.1f} ms ({r['jev_tok']} tok)"
        sp_str = f"{r['speedup']:.1f}×"
        tps_str = f"{r['jev_tps']:.1f} tok/s"
        print(f"{r['name']:<20} | {norm_str:<20} | {jev_str:<18} | {sp_str:<10} | {tps_str}")
    print("=" * 95)


if __name__ == "__main__":
    main()
