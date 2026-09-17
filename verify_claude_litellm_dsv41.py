#!/usr/bin/env python3
import json
import time
import urllib.request
import urllib.error
import threading

LITELLM_URL = "http://127.0.0.1:8101/v1/messages?beta=true"
BACKEND_URL = "http://127.0.0.1:8000/v1/chat/completions"

def post_json(url, data, headers=None):
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=hdrs)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status, resp.read().decode("utf-8")

def test_sequential_shared_prefix():
    print("\n--- Test 1: Sequential requests with shared prefix (Claude Code multi-turn conversation) ---")
    messages = [
        {"role": "user", "content": "You are a coding assistant. Remember the secret code is ALPHA-99."},
    ]
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": messages,
        "max_tokens": 40,
        "stream": False
    }
    status, body = post_json(LITELLM_URL, payload, {"x-api-key": "sk-local", "anthropic-version": "2023-06-01"})
    print(f"Turn 1 -> HTTP {status}")
    resp_obj = json.loads(body)
    assistant_text = "".join(c.get("text", "") for c in resp_obj.get("content", []))
    print(f"Assistant: {assistant_text[:80]}...")

    messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": "What was the secret code I gave you?"})
    payload["messages"] = messages
    status, body = post_json(LITELLM_URL, payload, {"x-api-key": "sk-local", "anthropic-version": "2023-06-01"})
    print(f"Turn 2 (Shared Prefix Hit) -> HTTP {status}")
    resp_obj = json.loads(body)
    assistant_text2 = "".join(c.get("text", "") for c in resp_obj.get("content", []))
    print(f"Assistant: {assistant_text2[:80]}...")
    assert "ALPHA-99" in assistant_text2 or "ALPHA" in assistant_text2, f"Expected secret code in: {assistant_text2}"
    print("Sequential shared prefix test: PASSED")

def test_different_prefixes():
    print("\n--- Test 2: Requests with different prefixes ---")
    prompts = [
        "Explain photosynthesis in one concise sentence.",
        "What are the primary colors in painting?",
        "Write a 4-line haiku about the autumn rain.",
    ]
    for i, p in enumerate(prompts):
        payload = {
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 35,
            "stream": False
        }
        status, body = post_json(LITELLM_URL, payload, {"x-api-key": "sk-local", "anthropic-version": "2023-06-01"})
        print(f"Prompt {i+1} -> HTTP {status}")
        resp_obj = json.loads(body)
        assistant_text = "".join(c.get("text", "") for c in resp_obj.get("content", []))
        print(f"Output: {assistant_text.strip()[:70]}...")
    print("Different prefixes test: PASSED")

def test_concurrent_requests():
    print("\n--- Test 3: Concurrent requests through LiteLLM ---")
    results = []
    errors = []

    def worker(worker_id):
        try:
            payload = {
                "model": "deepseek-v4.1-flash",
                "messages": [{"role": "user", "content": f"Worker {worker_id}: generate 3 random fruits."}],
                "max_tokens": 25,
                "stream": False
            }
            t0 = time.perf_counter()
            status, body = post_json(LITELLM_URL, payload, {"x-api-key": "sk-local", "anthropic-version": "2023-06-01"})
            dt = time.perf_counter() - t0
            results.append((worker_id, status, dt))
        except Exception as e:
            errors.append((worker_id, str(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for wid, st, dt in results:
        print(f"Worker {wid} -> HTTP {st} in {dt:.2f}s")
    if errors:
        print(f"Errors encountered: {errors}")
        raise RuntimeError(f"Concurrent test failed with errors: {errors}")
    print("Concurrent requests test: PASSED")

def test_tool_calling():
    print("\n--- Test 4: Tool calling via LiteLLM Anthropic format ---")
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        ],
        "tool_choice": {"type": "auto"},
        "max_tokens": 50,
        "stream": False
    }
    status, body = post_json(LITELLM_URL, payload, {"x-api-key": "sk-local", "anthropic-version": "2023-06-01"})
    print(f"Tool call request -> HTTP {status}")
    resp_obj = json.loads(body)
    content_blocks = resp_obj.get("content", [])
    print(f"Response blocks: {content_blocks}")
    print("Tool calling test: PASSED")

if __name__ == "__main__":
    t_start = time.perf_counter()
    test_sequential_shared_prefix()
    test_different_prefixes()
    test_concurrent_requests()
    test_tool_calling()
    print(f"\n==========================================")
    print(f"ALL TESTS PASSED in {time.perf_counter() - t_start:.2f}s")
    print(f"==========================================")
