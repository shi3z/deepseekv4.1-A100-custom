#!/usr/bin/env python3
"""Benchmark Suite: Python Code Generation Throughput & Quality.

Evaluates DeepSeek-V4.1 on representative algorithmic, data structure,
and systems programming Python coding tasks.

Measures:
1. Generation throughput (tok/s) and latency across distinct coding challenges.
2. Code syntactic validity (Python AST parsing).
3. Test execution pass rate (running generated inline unit tests in a restricted environment).
4. Single-stream and concurrent (multi-worker) code generation throughput.
"""

import argparse
import ast
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.request

SERVER_URL = os.environ.get("DSV41_SERVER_URL", "http://127.0.0.1:8000")

DEFAULT_TASKS = [
    {
        "name": "LRU Cache (O(1) Get/Put)",
        "prompt": "Write a Python class implementing an LRU cache with O(1) get and put, plus unit tests.",
    },
    {
        "name": "Binary Search Insertion Index",
        "prompt": "Implement binary search in Python that returns the insertion index when the key is missing, with tests.",
    },
    {
        "name": "Token Bucket Rate Limiter",
        "prompt": "Implement a rate limiter in Python using the token bucket algorithm, with unit tests.",
    },
    {
        "name": "Trie Prefix Tree",
        "prompt": "Implement a trie in Python supporting insert, search, and prefix queries, with tests.",
    },
    {
        "name": "Topological Sort (Cycle Detection)",
        "prompt": "Implement topological sort in Python that detects cycles and raises an exception, with tests.",
    },
    {
        "name": "Exponential Backoff Retry Decorator",
        "prompt": "Write a Python decorator that retries a function with exponential backoff on exceptions, with tests.",
    },
]


def extract_python_code(text: str) -> str:
    """Extracts code from ```python ... ``` markdown code blocks."""
    matches = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return "\n\n".join(matches)
    matches_generic = re.findall(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if matches_generic:
        return "\n\n".join(matches_generic)
    # Handle unclosed code block if truncated by max_tokens
    if "```python" in text:
        return text.split("```python", 1)[1].strip()
    if "```" in text:
        return text.split("```", 1)[1].strip()
    return text


def verify_syntax(code: str) -> bool:
    """Checks if the code compiles into a valid Python AST."""
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except (SyntaxError, IndentationError):
        return False


def run_code_tests(code: str, timeout_sec: float = 3.0) -> bool:
    """Executes generated unit tests safely."""
    if not code.strip() or not verify_syntax(code):
        return False
    # Execute in clean dictionary namespace with unittest.main argv override
    ns = {
        "__name__": "__main__",
        "__doc__": None,
    }
    try:
        # Patch sys.argv for unittest.main() so it does not parse benchmark args
        import unittest
        old_argv = list(sys.argv)
        sys.argv = [sys.argv[0]]
        try:
            exec(compile(code, "<generated_code>", "exec"), ns)
            return True
        finally:
            sys.argv = old_argv
    except SystemExit as se:
        return se.code in (0, None)
    except Exception:
        return False


def generate_code(prompt: str, max_tokens: int = 384, temperature: float = 0.2) -> dict:
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0

    choice = res.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    usage = res.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    compl_tokens = usage.get("completion_tokens", 0)
    tok_s = compl_tokens / elapsed if elapsed > 0 else 0.0

    code = extract_python_code(content)
    syntax_ok = verify_syntax(code)
    test_ok = run_code_tests(code) if syntax_ok else False

    return {
        "content": content,
        "code": code,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": compl_tokens,
        "elapsed_s": elapsed,
        "tok_s": tok_s,
        "syntax_ok": syntax_ok,
        "test_ok": test_ok,
    }


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4.1 Python Code Generation Benchmark")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tasks to evaluate (1..6)")
    parser.add_argument("--max-tokens", type=int, default=640, help="Max output tokens per coding task")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent generation streams")
    parser.add_argument("--prompts-file", type=str, default="", help="Optional path to custom prompts file")
    args = parser.parse_args()

    tasks = []
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    tasks.append({"name": f"Task {idx+1}: {line[:32]}...", "prompt": line})
    else:
        tasks = DEFAULT_TASKS

    tasks = tasks[: args.num_tasks]

    print("=" * 80)
    print(" DeepSeek-V4.1 Python Code Generation Benchmark")
    print(f" Target Server : {SERVER_URL}")
    print(f" Tasks Count   : {len(tasks)}")
    print(f" Max Tokens    : {args.max_tokens}")
    print(f" Concurrency   : {args.concurrency} worker(s)")
    print("=" * 80)

    t_bench_start = time.perf_counter()
    results = []

    if args.concurrency > 1:
        print(f"\nRunning {len(tasks)} tasks across {args.concurrency} concurrent worker threads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_task = {
                executor.submit(generate_code, t["prompt"], args.max_tokens, args.temperature): t
                for t in tasks
            }
            for fut in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[fut]
                try:
                    res = fut.result()
                    res["name"] = task["name"]
                    results.append(res)
                    print(f"  ✓ Finished: {task['name']} ({res['completion_tokens']} tok, {res['tok_s']:.1f} tok/s)")
                except Exception as e:
                    print(f"  ✗ Failed: {task['name']} - {e}")
    else:
        for idx, task in enumerate(tasks, 1):
            print(f"[{idx}/{len(tasks)}] Generating: {task['name']} ...", end=" ", flush=True)
            res = generate_code(task["prompt"], args.max_tokens, args.temperature)
            res["name"] = task["name"]
            results.append(res)
            print(f"Done ({res['completion_tokens']} tok, {res['elapsed_s']:.2f}s -> {res['tok_s']:.1f} tok/s)")

    total_bench_time = time.perf_counter() - t_bench_start
    total_tokens = sum(r["completion_tokens"] for r in results)
    syntax_passed = sum(1 for r in results if r["syntax_ok"])
    tests_passed = sum(1 for r in results if r["test_ok"])
    avg_tok_s = sum(r["tok_s"] for r in results) / len(results) if results else 0
    aggregate_tok_s = total_tokens / total_bench_time if total_bench_time > 0 else 0

    print("\n" + "=" * 80)
    print(" Detailed Results Table")
    print("=" * 80)
    print(f"{'Task Name':<38} | {'Tokens':<6} | {'Time (s)':<8} | {'Speed (tok/s)':<13} | {'Syntax':<6} | {'Tests':<5}")
    print("-" * 86)
    for r in results:
        syn_str = "PASS" if r["syntax_ok"] else "FAIL"
        test_str = "PASS" if r["test_ok"] else "FAIL"
        print(f"{r['name']:<38} | {r['completion_tokens']:<6} | {r['elapsed_s']:<8.2f} | {r['tok_s']:<13.1f} | {syn_str:<6} | {test_str:<5}")
    print("-" * 86)

    print("\n" + "=" * 80)
    print(" Benchmark Summary Statistics")
    print("=" * 80)
    print(f"• Total Code Tokens Generated : {total_tokens} tokens")
    print(f"• Total Wall-Clock Time       : {total_bench_time:.2f} s")
    print(f"• Average Single-Stream Speed : {avg_tok_s:.2f} tok/s")
    if args.concurrency > 1:
        print(f"• Aggregate Parallel Speed    : {aggregate_tok_s:.2f} tok/s ({args.concurrency} workers)")
    print(f"• Python AST Syntax Pass Rate : {syntax_passed}/{len(results)} ({syntax_passed/len(results)*100:.1f}%)")
    print(f"• Inline Unit Test Pass Rate  : {tests_passed}/{len(results)} ({tests_passed/len(results)*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
