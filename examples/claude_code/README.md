# Using DeepSeek-V4.1 as Backend for Claude Code (via LiteLLM)

This example shows how to run [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) entirely locally against your DeepSeek-V4.1-Flash server using LiteLLM as an Anthropic-to-OpenAI protocol proxy.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                 Claude Code CLI (claude)                    │
│  - Speaks Anthropic /v1/messages API with tool calling      │
│  - Sends tool definitions, tool_use, tool_result blocks    │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP (:8101)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  LiteLLM Proxy Server                       │
│  - Emulates Anthropic Messages endpoint                     │
│  - Translates Anthropic tools -> OpenAI Chat tools schema   │
│  - Translates DeepSeek tool_calls -> Anthropic tool_use     │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP (:8000)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            DeepSeek-V4.1-Flash Inference Engine             │
│  - 40-layer FP8 / FP4 MoE on A100 GPUs                     │
│  - In-GPU Slot Cache & Longest Common Prefix (LCP) reuse    │
│  - Bounded SWA Replay (CED) & Repetition Loop Detector      │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

Install `litellm` and `claude-code`:

```bash
pip install litellm pyyaml
npm install -g @anthropic-ai/claude-code
```

### 2. Start DeepSeek-V4.1 Server

Make sure the DeepSeek-V4.1 server is running (default port `8000`):

```bash
cd /mnt/ssdraid/git/deepseekv4.1
./run_server_batched.sh
```

Verify the health endpoint:

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok", "model": "deepseek-v4.1-flash", ...}
```

### 3. Verify the LiteLLM Bridge

Run the included verification script to test tool calling, shared prefix caching, and concurrent request handling:

```bash
python3 examples/claude_code/verify_litellm.py
```

Expected output:
```text
--- Test 1: Sequential requests with shared prefix (Multi-turn tool loop) ---
Turn 1 -> HTTP 200
Turn 2 (Shared Prefix Hit) -> HTTP 200
Sequential shared prefix test: PASSED

--- Test 2: Requests with different prefixes ---
Prompt 1 -> HTTP 200
Prompt 2 -> HTTP 200
Prompt 3 -> HTTP 200
Different prefixes test: PASSED

--- Test 3: Concurrent requests through LiteLLM ---
Worker 0 -> HTTP 200 in 0.81s
Worker 1 -> HTTP 200 in 1.83s
Worker 2 -> HTTP 200 in 2.87s
Concurrent requests test: PASSED

--- Test 4: Tool calling via LiteLLM Anthropic format ---
Tool call request -> HTTP 200
Response blocks: [{'type': 'text', ...}, {'type': 'tool_use', 'name': 'get_weather', ...}]
Tool calling test: PASSED

==================================================
ALL TESTS PASSED in 12.65s
==================================================
```

### 4. Launch Claude Code

Run the wrapper script:

```bash
./examples/claude_code/claude-dsv41.sh
```

The script automatically:
1. Spawns `litellm` proxy on `127.0.0.1:8101` in the background (if not already running).
2. Sets `ANTHROPIC_BASE_URL="http://127.0.0.1:8101"`.
3. Routes Opus/Sonnet/Haiku models to `deepseek-v4.1-flash`.
4. Configures maximum context window (786,432 tokens) and timeout.
5. Launches `claude`.

You can also pass arguments directly:
```bash
./examples/claude_code/claude-dsv41.sh -p "Check git status and summarize recent commits"
```

## Key Configuration Parameters

In `dsv41-litellm.yaml`:
```yaml
model_list:
  - model_name: deepseek-v4.1-flash
    litellm_params:
      model: openai/deepseek-v4.1-flash
      api_base: http://127.0.0.1:8000/v1
      api_key: sk-local
      timeout: 3600
      max_tokens: 1048576

litellm_settings:
  # Crucial: translates Anthropic Messages format to OpenAI Chat format
  use_chat_completions_url_for_anthropic_messages: true

general_settings:
  master_key: sk-local
```

## Why In-GPU Slot Cache is Critical for Claude Code

In agentic coding workflows like Claude Code, every tool call sends the **entire conversation history** back to the model:
- Turn 1: System prompt + User request (e.g. 5,000 tokens)
- Turn 2: Turn 1 + Tool call `Bash(git status)` + Tool output (e.g. 6,200 tokens)
- Turn 20: 50,000+ tokens
- Turn 50: 150,000+ tokens

With standard inference engines, Turn 50 requires full re-prefill of 150,000 tokens on every turn (taking minutes). 

With DeepSeek-V4.1's **In-GPU Slot Cache (`DSV41_GPU_SLOT_CACHE=1`)**:
- Longest Common Prefix (LCP) is detected directly in VRAM across active slots (`copy_seq` in < 0.05 ms).
- Only the **new suffix tokens** (e.g. 200 tokens from the latest tool output) are processed.
- Prefill completes in **sub-second latency** even when conversation context exceeds 400,000 tokens!
