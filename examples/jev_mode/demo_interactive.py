#!/usr/bin/env python3
"""Jev Mode Demo: Interactive CLI for Instant Structured Output.

Allows typing arbitrary text and immediately extracting structured JSON
via parallel non-autoregressive candidate evaluation.
"""

import json
import os
import sys
import time
import urllib.request

SERVER_URL = os.environ.get("DSV41_SERVER_URL", "http://127.0.0.1:8000")

DEFAULT_SCHEMA = {
    "intent": ["question", "bug_report", "feature_request", "praise", "complaint", "other"],
    "urgency": ["low", "medium", "high", "critical"],
    "sentiment": ["positive", "neutral", "negative"],
    "requires_followup": [True, False],
}


def query_jev(text: str, schema: dict) -> tuple[dict, float]:
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
    print("DeepSeek-V4.1 Jev Mode: Interactive CLI")
    print(f"Target Server: {SERVER_URL}")
    print("=" * 70)
    print("Active Schema:")
    print(json.dumps(DEFAULT_SCHEMA, indent=2))
    print("\nType your text below and press Enter to extract.")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("Text > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                break

            res, ms = query_jev(user_input, DEFAULT_SCHEMA)
            
            content = None
            if "choices" in res and res["choices"]:
                content = res["choices"][0]["message"]["content"]
            elif "result" in res:
                content = json.dumps(res["result"])

            print(f"\n[Result in {ms:.1f} ms]:")
            try:
                parsed = json.loads(content)
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
            except Exception:
                print(content)

            usage = res.get("usage", {})
            toks = usage.get("completion_tokens", 0)
            if toks and ms > 0:
                print(f"Tokens: {toks} | Speed: {toks / (ms / 1000.0):.1f} tok/s")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
