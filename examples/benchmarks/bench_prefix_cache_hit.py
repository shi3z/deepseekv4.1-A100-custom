#!/usr/bin/env python3
"""Benchmark Suite: Prefix Cache Hit vs Cold Prefill (LCP Cache Acceleration).

Demonstrates DeepSeek-V4.1's prefix caching engine:
  - Turn 1 (Cold Prefill): First request must prefill entire system prompt / document.
  - Turn 2 (Cache Hit): Subsequent request reuses the GPU-cached prefix, prefilling only new tokens.
  - Turn 3 (Incremental Cache): Multi-turn conversation preserves the growing prefix tree.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

SERVER_URL = os.environ.get("DSV41_SERVER_URL", "http://127.0.0.1:8000")

# 1,200+ word shared system context (simulates a Claude Code session context with tools, codebase rules, and files)
SHARED_SYSTEM_CONTEXT = (
    "You are an expert software engineer operating in an interactive agentic development environment.\n"
    "Below are the codebase architecture guidelines, API specifications, and operational invariants:\n\n"
    + ("# SECTION 1: SYSTEM INVARIANTS\n"
       "- All operations must maintain memory safety and avoid unconstrained tensor growth.\n"
       "- The system utilizes a multi-GPU pipelined Expert Parallel (EP) architecture with 4 A100 GPUs.\n"
       "- Attention layers execute dynamic compressed KV caching with ratios ranging from 1:1 to 1:2.\n"
       "- When executing prefix matching, the longest common prefix (LCP) is identified against the global prefix store.\n"
       "- Memory management utilizes pinned host RAM tables for Engram embeddings with fast host-to-device transfers.\n\n") * 6
    + ("# SECTION 2: API & TOOL DEFINITIONS\n"
       "- execute_bash(cmd: str) -> CommandResult: Runs shell commands within the current worktree.\n"
       "- read_file(path: str, offset: int = 0, limit: int = 1000) -> str: Reads file contents.\n"
       "- edit_file(path: str, target: str, replacement: str) -> bool: Applies surgical modifications.\n"
       "- git_commit(message: str) -> str: Creates atomic git commits adhering to semantic convention.\n\n") * 6
)


def send_chat_request(messages: list[dict], max_tokens: int = 30) -> dict:
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
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
    elapsed_time = time.perf_counter() - t0

    usage = res.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    compl_tokens = usage.get("completion_tokens", 0)

    return {
        "elapsed_s": elapsed_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": compl_tokens,
        "response": res["choices"][0]["message"]["content"],
    }


def main():
    print("=" * 80)
    print("DeepSeek-V4.1 Prefix Cache Acceleration Benchmark")
    print(f"Target Server: {SERVER_URL}")
    print("=" * 80)

    print("\nSimulating a multi-turn developer session with a large shared system context...")
    
    # Turn 1: Cold Cache (First time seeing the large context)
    print("\n[Turn 1: Cold Prefill (Cache MISS)]")
    msgs_turn1 = [
        {"role": "system", "content": SHARED_SYSTEM_CONTEXT},
        {"role": "user", "content": "What is the primary constraint mentioned in Section 1?"},
    ]
    t1 = send_chat_request(msgs_turn1, max_tokens=25)
    print(f"  Prompt Tokens    : {t1['prompt_tokens']}")
    print(f"  Total Round-Trip : {t1['elapsed_s']:.3f} s")
    print(f"  Output Tokens    : {t1['completion_tokens']}")
    print(f"  Answer Snippet   : {t1['response'].strip()[:60]}...")

    # Turn 2: Warm Cache (Exact same system context, different follow-up question)
    print("\n[Turn 2: Shared Prefix Reuse (Cache HIT)]")
    msgs_turn2 = [
        {"role": "system", "content": SHARED_SYSTEM_CONTEXT},
        {"role": "user", "content": "List the tools available in Section 2."},
    ]
    t2 = send_chat_request(msgs_turn2, max_tokens=25)
    print(f"  Prompt Tokens    : {t2['prompt_tokens']}")
    print(f"  Total Round-Trip : {t2['elapsed_s']:.3f} s")
    print(f"  Output Tokens    : {t2['completion_tokens']}")
    print(f"  Answer Snippet   : {t2['response'].strip()[:60]}...")

    # Turn 3: Multi-turn Conversation Continuation
    print("\n[Turn 3: Multi-Turn Continuation (Incremental HIT)]")
    msgs_turn3 = [
        {"role": "system", "content": SHARED_SYSTEM_CONTEXT},
        {"role": "user", "content": "What is the primary constraint mentioned in Section 1?"},
        {"role": "assistant", "content": t1["response"]},
        {"role": "user", "content": "How does it interact with the git_commit tool?"},
    ]
    t3 = send_chat_request(msgs_turn3, max_tokens=25)
    print(f"  Prompt Tokens    : {t3['prompt_tokens']}")
    print(f"  Total Round-Trip : {t3['elapsed_s']:.3f} s")
    print(f"  Output Tokens    : {t3['completion_tokens']}")
    print(f"  Answer Snippet   : {t3['response'].strip()[:60]}...")

    # Summary
    print("\n" + "=" * 80)
    print("PREFIX CACHING SPEEDUP SUMMARY")
    print("=" * 80)
    print(f"Turn 1 (Cold Prefill, Cache MISS) : {t1['elapsed_s']:.3f} s (Built prefix cache)")
    print(f"Turn 2 (Warm Shared Prefix, HIT)  : {t2['elapsed_s']:.3f} s ({t1['elapsed_s'] / max(t2['elapsed_s'], 1e-4):.2f}× faster)")
    print(f"Turn 3 (Multi-Turn Continuation)  : {t3['elapsed_s']:.3f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
